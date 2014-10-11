<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );


use AuthDB\AuthDB;

use DataPortal\Config;

use FileMgr\RestRequest;
use FileMgr\FileMgrIrodsDb;

use LogBook\LogBook;
use LusiTime\LusiTime;

/*
 * This script will process a request for deleting a set of files
 * from an OFFLINE disk storage managed by the Data Management System.
 * A data set is defined by the following parameters:
 * 
 *   <exper_id> <run_number> <storage_class> <file_type>
 * 
 * When finished the script will return a JSOB object with the completion
 * status of the operation.
 */
header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_success($result) {    
    print json_encode(
        array_merge(
            array( 'status' => 'success', 'updated' => LusiTime::now()->toStringShort()),
            $result
        )
    );
    exit;
}
function report_error( $msg, $result=array()) {
    print json_encode(
        array_merge(
            array( 'status' => 'error', 'message' => $msg  ),
            $result
        )
    );
    exit;
}

/* Parse mandatory parameters of the script
 */
if( !isset( $_GET['exper_id'] )) report_error( "no experiment identifier parameter found in the request" );
$exper_id = intval( trim( $_GET['exper_id'] ));
if( $exper_id == 0 ) report_error( "invalid experiment identifier found in the request" );

if( !isset( $_GET['runnum'] )) report_error( "no run number parameter found in the request" );
$runnum = intval( trim( $_GET['runnum'] ));
if( $runnum == 0 ) report_error( "invalid run number found in the request" );

if( !isset( $_GET['type'] )) report_error( "no file type parameter found in the request" );
$type = strtolower( trim( $_GET['type'] ));
if( $type == '' ) report_error( "invalid file type found in the request" );

if( !isset( $_GET['storage'] )) report_error( "no storage class parameter found in the request" );
$storage = strtoupper( trim( $_GET['storage'] ));
switch( $storage ) {
    case 'SHORT-TERM':
    case 'MEDIUM-TERM':
        break;
    default:
        report_error( "inappropriate storage class of files in the request" );
}

/*
 * Analyze and process the request
 */
try {

    AuthDB::instance()->begin();
    if (!AuthDB::instance()->hasPrivilege(AuthDB::instance()->authName(), null, 'StoragePolicyMgr', 'edit'))
        report_error ("sorry, you don't possess sufficient privileges for this operation");

    LogBook::instance()->begin();
    Config::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id($exper_id);
    if( is_null($experiment)) report_error("No such experiment");

    $run = $experiment->find_run_by_num($runnum);
    if( is_null($run))
        report_error(
            "No such run in the experiment",
            array( 'medium_quota_used_gb' => Config::instance()->calculate_medium_quota($exper_id)));

    // Find the files to be deleted. Only consider file which are associated
    // with the specified storage class.
    //
    $files = array();
    foreach( FileMgrIrodsDb::instance()->runs(
        $experiment->instrument()->name(),
        $experiment->name(),
        $type,
        $runnum,
        $runnum ) as $run )
        foreach( $run->files as $file )
            if( $file->resource == 'lustre-resc' ) {
                $request = Config::instance()->find_medium_store_file(
                    array(
                        'exper_id'       => $exper_id,
                        'runnum'         => $runnum,
                        'file_type'      => $type,
                        'irods_filepath' => "{$file->collName}/{$file->name}",
                        'irods_resource' => 'lustre-resc'
                    )
                );
                switch( $storage ) {
                    case 'SHORT-TERM' : if(  is_null($request)) array_push($files, $file); break;
                    case 'MEDIUM-TERM': if( !is_null($request)) array_push($files, $file); break;
                }
            }

    // Proceed with the operation
    //
    foreach( $files as $file ) {

        $irods_filepath = "{$file->collName}/{$file->name}";

        // Delete selected files by their full path and a replica.
        //
        $request = new RestRequest(
            "/replica{$irods_filepath}/{$file->replica}",
            'DELETE'
        );
        $request->execute();
        $responseInfo = $request->getResponseInfo();
        $http_code = intval($responseInfo['http_code']);
        switch($http_code) {
            case 200: break;
            case 404:
               report_error(
                   "file '{$file->name}' doesn't exist",
                   array( 'medium_quota_used_gb' => Config::instance()->calculate_medium_quota($exper_id)));
            default:
                report_error(
                    "failed to delete file '{$file->name}' because of HTTP error {$http_code}",
                    array( 'medium_quota_used_gb' => Config::instance()->calculate_medium_quota($exper_id)));
        }

        if ($storage == 'MEDIUM-TERM')
            Config::instance()->remove_medium_store_file ( $exper_id, $runnum, $type, $irods_filepath );

        // Delete selected files by their full path and a replica. Also make sure
        // no old entries remain in the file restore queue. Otherwise it would
        // (falsefully) appear as if the deleted file is being restored.
        //
        Config::instance()->delete_file_restore_request(
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

    $medium_quota_used_gb = Config::instance()->calculate_medium_quota($exper_id);

    AuthDB::instance()->commit();
    LogBook::instance()->commit();
    Config::instance()->commit();
    FileMgrIrodsDb::instance()->commit();

    report_success( array( 'medium_quota_used_gb' => $medium_quota_used_gb ));

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
