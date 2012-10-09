<?php

/*
 * This script will process a request for moving a set of files
 * from the SHORT-TERM store to the MEDIUM-TERM data store. This operation
 * only applies to files managed by the Data MAnagement System.
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

function report_success($result) {
    
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

if( !isset( $_GET['runnum'] )) report_error( "no run number parameter found in the request" );
$runnum = intval( trim( $_GET['runnum'] ));
if( $runnum == 0 ) report_error( "invalid run number found in the request" );

if( !isset( $_GET['type'] )) report_error( "no file type parameter found in the request" );
$type = strtolower( trim( $_GET['type'] ));
if( $type == '' ) report_error( "invalid file type found in the request" );

if( !isset( $_GET['storage'] )) report_error( "no storage class parameter found in the request" );
$storage = strtoupper( trim( $_GET['storage'] ));
if( $storage == '' ) report_error( "invalid storage class found in the request" );
if( $storage != 'SHORT-TERM' ) report_error( "inappropriate storage class of files in the request" );

/*
 * Analyze and process the request
 */
try {

    LogBook::instance()->begin();
    Config::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id($exper_id);
    if( is_null($experiment)) report_error("No such experiment");

    $run = $experiment->find_run_by_num($runnum);
    if( is_null($run))
        report_error(
            "No such run in the experiment",
            array( 'medium_quota_used_gb' => Config::instance()->calculate_medium_quota($exper_id)));

    // Find the files to be moved. And skip files which are already at their
    // intended destination.
    //
    $runs = null;

    FileMgrIrodsWs::runs(
        $runs,
        $experiment->instrument()->name(),
        $experiment->name(),
        $type,
        $runnum.'-'.$runnum );

    if( is_null($runs )) report_error(
        'server encountered an internal error when trying to get a list of files for the run',
        array( 'medium_quota_used_gb' => Config::instance()->calculate_medium_quota($exper_id)));

    $files = array();
    foreach( $runs as $run ) {
        foreach( $run->files as $file ) {
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
                if( is_null($request))
                    array_push($files, $file);
            }
        }
    }

    // Evaluate quota availability (if the one applies for the experiment)
    //
    $medium_quota_gb = intval(Config::instance()->get_policy_param('MEDIUM-TERM',   'QUOTA'));
    $param = $experiment->regdb_experiment()->find_param_by_name('MEDIUM-TERM-DISK-QUOTA');
    if ($param) {
        $gb = intval($param->value());
        if ($gb) $medium_quota_gb = $gb;
    }
    if( $medium_quota_gb ) {
        $medium_quota_used_gb = Config::instance()->calculate_medium_quota($exper_id);
        $size_gb = 0;
        $bytes_in_gb = 1024.0 * 1024.0 * 1024.0;
        foreach( $files as $file ) {
            $size_gb += $file->size / $bytes_in_gb;
        }
        $size_gb = intval($size_gb);
        if( $medium_quota_used_gb + $size_gb > $medium_quota_gb )
            report_error(
              'You can no longer move files into the MEDIUM-TERM storage '.
              'either because the experiment has run out of its storage quota or because the amount of data '.
              'involved in this request would result in exceeding  quota limit of the experiment.',
              array( 'medium_quota_used_gb' => Config::instance()->calculate_medium_quota($exper_id)));
    }

    // Register selected files in the MEDIUM-TERM storage
    //
    foreach( $files as $file ) {
        Config::instance()->add_medium_store_file (
            array(
                'exper_id'       => $exper_id,
                'runnum'         => $runnum,
                'file_type'      => $type,
                'irods_filepath' => "{$file->collName}/{$file->name}",
                'irods_resource' => 'lustre-resc',
                'irods_size'     => $file->size
            )
        );
    }

    // Update quota usage for the experiment
    //
    $medium_quota_used_gb = Config::instance()->calculate_medium_quota($exper_id);

    LogBook::instance()->commit();
    Config::instance()->commit();

    report_success( array( 'medium_quota_used_gb' => $medium_quota_used_gb ));

} catch( LogBookException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( LusiTimeException   $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( FileMgrException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( DataPortalException $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
