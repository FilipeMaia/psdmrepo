<?php

/*
 * This script will process an action on a file restore request
 * from HPSS to an OFFLINE disk storage managed by the Data Management System.

 * The operation is defined by the following parameter:
 * 
 *   <action>
 * 
 * A file is defined by:
 * 
 *   <exper_id> <run_number> <storage_class> <file_type> <file_name>|<irods_filepath> [<force>]
 * 
 * If the optional '<force>' is provided the new request will substitute the old one.
 *
 * When finished the script will return a JSOB object with the completion
 * status of the operation.
 */

require_once('logbook/logbook.inc.php');
require_once('filemgr/filemgr.inc.php');
require_once('dataportal/dataportal.inc.php');

use LogBook\LogBook;

use LusiTime\LusiTime;

use FileMgr\RestRequest;
use FileMgr\FileMgrIrodsDb;

use DataPortal\Config;

function report_success($result) {
    
    $updated_str = json_encode(LusiTime::now()->toStringShort());
    
    header('Content-type: application/json');
    header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
    header("Expires: Sat, 26 Jul 1997 05:00:00 GMT");   // Date in the past
    
    print json_encode(
        array_merge(
            array('status' => 'success', 'updated' => $updated_str),
            $result
       )
   );
    exit;
}

function report_error($msg, $result=array()) {
    
    header('Content-type: application/json');
    header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
    header("Expires: Sat, 26 Jul 1997 05:00:00 GMT");   // Date in the past

    print json_encode(
        array_merge(
            array('status' => 'error', 'message' => $msg ),
            $result
       )
   );
    exit;
}

/* Parse mandatory parameters of the script
 */
if (!isset($_GET['action'])) report_error("no action parameter found in the request");
$action = strtolower(trim($_GET['action']));
switch ($action) {
    case 'resubmit': break;
    case 'cancel':   break;
    default:         report_error("invalid action parameter found in the request");
}
if (!isset($_GET['exper_id'])) report_error("no experiment identifier parameter found in the request");
$exper_id = intval(trim($_GET['exper_id']));
if ($exper_id == 0) report_error("invalid experiment identifier found in the request");

if (!isset($_GET['runnum'])) report_error("no run number parameter found in the request");
$runnum = intval(trim($_GET['runnum']));
if ($runnum == 0) report_error("invalid run number found in the request");

if (!isset($_GET['type'])) report_error("no file type parameter found in the request");
$type = strtolower(trim($_GET['type']));
if ($type == '') report_error("invalid file type found in the request");

if (!isset($_GET['storage'])) report_error("no storage class parameter found in the request");
$storage = strtoupper(trim($_GET['storage']));
if ($storage == '') report_error("invalid storage class found in the request");

$file_name = null;
$irods_filepath = null;
if (isset($_GET['file_name'])) {
    $file_name = trim($_GET['file_name']);
    if ($file_name == '') $file_name = null;
} elseif (isset($_GET['irods_filepath'])) {
    $irods_filepath = trim($_GET['irods_filepath']);
    if ($irods_filepath == '') $irods_filepath = null;
    else {
        $irods_filepath_split = explode('/', $irods_filepath);
        $file_name = $irods_filepath_split[count($irods_filepath_split)-1];
        if ($file_name == '')
            report_error("invalid iRODS file path in the request");
    }
}

if (is_null($file_name))
    report_error("no file name or iRODS file path parameter found in the request");

$force = isset($_GET['force']);

/*
 * Analyze and process the request
 */
try {

    LogBook::instance()->begin();
    Config::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $medium_quota_used_gb = Config::instance()->calculate_medium_quota($exper_id);

    $experiment = LogBook::instance()->find_experiment_by_id($exper_id);
    if (is_null($experiment))
        report_error(
            'No such experiment',
            array('medium_quota_used_gb' => $medium_quota_used_gb));

    $run = $experiment->find_run_by_num($runnum);
    if (is_null($run))
        report_error(
            "No such run in the experiment",
            array('medium_quota_used_gb' => $medium_quota_used_gb));

    // Find the file to be restored. And check if the file is already on disk.
    // Also find the file path in iRODS if none was provided as a parameter to the script.
    //
    $known_file = false;
    $on_tape = false;
    $on_disk = false;
    foreach (FileMgrIrodsDb::instance()->find_file(
        $experiment->instrument()->name(),
        $experiment->name(),
        $type,
        $file_name) as $r) {
        if ($r->run == $runnum) {
            foreach ($r->files as $f) {
                $known_file = true;
                if ($f->resource == 'hpss-resc' ) $on_tape = true;
                if ($f->resource == 'lustre-resc') $on_disk = true;
                if (is_null($irods_filepath)) $irods_filepath = "{$f->collName}/{$f->name}";
            }
        }
    }
    if (!$known_file)
        report_error(
            'no such file found for the experiment',
            array('medium_quota_used_gb' => $medium_quota_used_gb));

    if (!$on_tape)
        report_error(
            'the file is not on tape',
            array('medium_quota_used_gb' => $medium_quota_used_gb));

    if (!$on_disk) {
    
        // Check if there is any outstandig tape request
        //
        $request_spec = array(
            'exper_id'  => $exper_id,
            'runnum'    => $runnum,
            'file_type' => $type,
            'irods_filepath'     => $irods_filepath,
            'irods_src_resource' => 'hpss-resc',
            'irods_dst_resource' => 'lustre-resc'
        );
        $file_restore_request = Config::instance()->find_file_restore_request($request_spec);
        
        // The rest of the operation depends on the requested action.
        //
        if ($action == 'cancel') {
            if (!is_null($file_restore_request))
                Config::instance()->delete_file_restore_request($request_spec);
            
        } elseif ($action == 'resubmit') {
       
            // Make sure there are no outstanding restore requests for that file yet
            // unless the 'force' option is provide.
            //
            if (!is_null($file_restore_request)) {
                if ($force)
                    Config::instance()->delete_file_restore_request($request_spec);
                else
                    report_error(
                        'there is an outstanding file restore request for the file',
                        array('medium_quota_used_gb' => $medium_quota_used_gb));
            }

//            // Restore the file by its full path and a source resource
//            //
//            $request = new RestRequest(
//                "/files{$irods_filepath}",
//                'POST',
//                array(
//                    'src_resource' => 'hpss-resc',
//                    'dst_resource' => 'lustre-resc'
//               ),
//                true  /* to package parameters into the POST body */
//            );
//            $request->execute();
//            $responseInfo = $request->getResponseInfo();
//            $http_code = intval($responseInfo['http_code']);
//            switch($http_code) {
//                case 200: break;
//                case 404:
//                    report_error(
//                        "file '{$file->name}' doesn't exist",
//                        array('medium_quota_used_gb' => $medium_quota_used_gb));
//                default:
//                    report_error(
//                        "failed to restore file '{$file->name}' because of HTTP error {$http_code}",
//                        array('medium_quota_used_gb' => $medium_quota_used_gb));
//            }
            Config::instance()->add_file_restore_request(
                array(
                    'exper_id'  => $exper_id,
                    'runnum'    => $runnum,
                    'file_type' => $type,
                    'irods_filepath'     => $irods_filepath,
                    'irods_src_resource' => 'hpss-resc',
                    'irods_dst_resource' => 'lustre-resc'
               )
           );
        }
    }
    LogBook::instance()->commit();
    Config::instance()->commit();
    FileMgrIrodsDb::instance()->begin();

    report_success(array('medium_quota_used_gb' => $medium_quota_used_gb));

} catch(Exception $e) { report_error($e.'<pre>'.print_r($e->getTrace(), true).'</pre>'); }

?>
