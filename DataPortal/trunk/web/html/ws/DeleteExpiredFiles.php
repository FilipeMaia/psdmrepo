<?php

/*
 * This script will process a request for deleting expired files
 * from an OFFLINE disk storage managed by the Data Management System.
 * A data set is defined by the following parameters:
 * 
 *   <exper_id> <storage_class>
 * 
 * When finished the script will return a JSOB object with the completion
 * status of the operation.
 */

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LusiTime\LusiTime;
use FileMgr\RestRequest;
use FileMgr\FileMgrIrodsDb;
use DataPortal\Config;

function report_success($result=array()) {
    
    $updated_str = json_encode( LusiTime::now()->toStringShort());
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    print json_encode(
        array_merge(
            array( 'status' => 'success', 'updated' => $updated_str ),
            $result
        )
    );
    exit;
}

function report_error( $msg, $result=array()) {
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

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

if( !isset( $_GET['storage'] )) report_error( "no storage class parameter found in the request" );
$storage = strtoupper( trim( $_GET['storage'] ));
if( $storage == '' ) report_error( "invalid storage class found in the request" );
if( $storage !== 'SHORT-TERM') report_error( "unsupported storage class: '{$storage}'" );

function is_expired ($ctime_time, $retention_months, $deadline_time) {
    $expiration_time = new LusiTime($ctime_time->sec + 31 * 24 * 3600 * $retention_months);
    return !is_null($expiration_time) && $expiration_time->less($deadline_time);
}

try {

    LogBook::instance()->begin();
    Config::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id($exper_id);
    if( is_null($experiment)) report_error("No such experiment");

    $default_short_ctime_time_str =        Config::instance()->get_policy_param('SHORT-TERM',  'CTIME');
    $default_short_retention      = intval(Config::instance()->get_policy_param('SHORT-TERM',  'RETENTION'));

    $short_quota_ctime            = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-CTIME' );
    $short_quota_ctime_time_str   = is_null( $short_quota_ctime ) ? '' : $short_quota_ctime->value();
    $short_quota_ctime_time       = LusiTime::parse($short_quota_ctime_time_str ? $short_quota_ctime_time_str : $default_short_ctime_time_str);

    $short_retention              = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-RETENTION' );
    $short_retention_months       = is_null( $short_retention ) ? 0 : intval($short_retention->value());
    $short_retention_months       = $short_retention_months ? $short_retention_months : $default_short_retention;

    $deadline_time = LusiTime::now();
 
    // Get all files of the experiment into a dictionary
    //
    $all_files = array();
    foreach (array('xtc','hdf5') as $type)
        foreach (FileMgrIrodsDb::instance()->runs ($experiment->instrument()->name(), $experiment->name(), $type) as $r)
            foreach ($r->files as $f)
                if ($f->resource == 'lustre-resc') {
                    $irods_filepath = "{$f->collName}/{$f->name}";
                    $all_files[$irods_filepath] =
                        array(
                            'run'  => $r->run,
                            'type' => $type,
                            'file' => $f
                        );
                }

    $total_files = count($all_files);

    // Clean the dictionary by removing files which are in the MEDIUM-TERM storage
    //
    foreach (Config::instance()->medium_store_files($experiment->id()) as $f)
        unset( $all_files[$f['irods_filepath']]);

    $short_term_files = count($all_files);

    // Clean the dictionary by removing files which are not yet expired
    //
    foreach ($all_files as $irods_filepath => $f) {
        $ctime_time = new LusiTime(intval($f['file']->ctime));
        if (!is_null($short_quota_ctime_time) && $ctime_time->less($short_quota_ctime_time)) {
            $ctime_time = $short_quota_ctime_time;
        }
        if (!is_expired (
            $ctime_time,
            $short_retention_months,
            $deadline_time)) unset( $all_files[$irods_filepath]);
    }

    $expired_files = count($all_files);

//    report_error("total files: {$total_files}, SHORT-TERM files: {$short_term_files}, expired files: {$expired_files}");
//    return;

    // Proceed with the operation
    //
    foreach ($all_files as $irods_filepath => $f) {

        for ($tries = 2; $tries > 0 ; $tries--) {

            // Delete selected files by their full path and a replica. Also make sure
            // no old entries remain in the file restore queue. Otherwise it would
            // (falsefully) appear as if the deleted file is being restored.
            //
            Config::instance()->delete_file_restore_request(
                array(
                    'exper_id'  => $exper_id,
                    'runnum'    => $f['run'],
                    'file_type' => $f['type'],
                    'irods_filepath'     => $irods_filepath,
                    'irods_src_resource' => 'hpss-resc',
                    'irods_dst_resource' => 'lustre-resc'
                )
            );

            $replica = $f['file']->replica;
            $request = new RestRequest(
                "/replica{$irods_filepath}/{$replica}",
                'DELETE'
            );
            $request->execute();
            $responseInfo = $request->getResponseInfo();
            $http_code = intval($responseInfo['http_code']);
            switch($http_code) {
                case 200:
                    $tries = 0 ;
                    break;
                case 404:
                    report_error("file '{$irods_filepath}' doesn't exist");
                default :
                    if (!$tries) report_error("failed to delete file '{$irods_filepath}' because of HTTP error {$http_code}");
                    break;
            }
        }
    }
    FileMgrIrodsDb::instance()->commit();
    Config::instance()->commit();
    LogBook::instance()->commit();

    report_success();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
