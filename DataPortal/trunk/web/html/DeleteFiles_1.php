<?php

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

require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\RestRequest;
use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use DataPortal\Config;
use DataPortal\DataPortalException;

function report_success() {
    
    $updated_str = json_encode( LusiTime::now()->toStringShort());
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    print json_encode( array( 'status' => 'success', 'updated' => $updated_str ));

    exit;
}

function report_error($msg) {
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print json_encode( array( 'status' => 'error', 'message' => $msg ));

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
if( $storage == '' ) report_error( "invalid storage class found in the request" );


/*
 * Analyze and process the request
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $config = Config::instance();
    $config->begin();

    $experiment = $logbook->find_experiment_by_id($exper_id);
    if( is_null($experiment)) report_error("No such experiment");

    $run = $experiment->find_run_by_num($runnum);
    if( is_null($run)) report_error("No such run in the experiment");

    $runs2files = array();

    // Find the files to be deleted
    //
    $runs = null;

    FileMgrIrodsWs::runs(
        $runs,
        $experiment->instrument()->name(),
        $experiment->name(),
        $type,
        $runnum.'-'.$runnum );

    if( is_null($runs )) report_error('server encountered an internal error when trying to get a list of files for thr run');

    $files = array();
    foreach( $runs as $run )
        foreach( $run->files as $file )
            if( $file->resource == 'lustre-resc' ) array_push($files, $file);

    // Delete selected files by their full path and a replica. Also make sure
    // no old entries remain in the file restore queue. Otherwise it would
    // (falsfully) appear as if the deleted file is being restored.
    //
    foreach( $files as $file ) {
        $irods_filepath = "{$file->collName}/{$file->name}";
        $config->delete_file_restore_request(
            array(
                'exper_id'  => $exper_id,
                'runnum'    => $runnum,
                'file_type' => $type,
                'irods_filepath'     => $irods_filepath,
                'irods_src_resource' => 'hpss-resc',
                'irods_dst_resource' => 'lustre-resc'
            )
        );
        $request = new RestRequest(
            "/replica{$file->collName}/{$file->name}/{$file->replica}",
            'DELETE'
        );
        $request->execute();
        $responseInfo = $request->getResponseInfo();
        $http_code = intval($responseInfo['http_code']);
        switch($http_code) {
            case 200: break;
            case 404: report_error("file '{$file->name}' doesn't exist");
            default:  report_error("failed to delete file '{$file->name}' because of HTTP error {$http_code}");
        }
    }
    $logbook->commit();
    $config->commit();

    report_success();

} catch( LogBookException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( LusiTimeException   $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( FileMgrException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( DataPortalException $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
?>
