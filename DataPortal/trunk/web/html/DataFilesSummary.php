<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    print <<< HERE
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
HERE;
    exit;
}

/*
 * This script will process a request for data files in a context of
 * the specified experiment.
 */
if( !isset( $_GET['exper_id'] )) report_error( "no experiemnt identifier found among script parameters" );
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' )  report_error( "experiment identifier can't be empty" );

/*
 * Return JSON object with the summary data for the experiment. All known runs and files
 * will be included into the report.
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $exper_id ) or report_error( "no such experiment" );

    /* Get the stats for data files
     */
    $num_runs = 0;
    $xtc_size       = 0.0;
    $xtc_num_files  = 0;
    $xtc_archived   = 0;
    $xtc_local_copy = 0;

    $hdf5_size       = 0.0;
    $hdf5_num_files  = 0;
    $hdf5_archived   = 0;
    $hdf5_local_copy = 0;

    $range = FileMgrIrodsWs::max_run_range( $experiment->instrument()->name(), $experiment->name(), array('xtc','hdf5'));

    $num_runs = $range['total'];
    $min_run  = $range['min'];
    $max_run  = $range['max'];
    $range_of_runs = $min_run.'-'.$max_run;

    $runs2files = array();

    $xtc_runs = null;
    FileMgrIrodsWs::runs( $xtc_runs, $experiment->instrument()->name(), $experiment->name(), 'xtc', $range_of_runs );
    foreach( $xtc_runs as $run ) {
        $unique_files = array();  // per this run
        $files = $run->files;
        foreach( $files as $file ) {
            if( !array_key_exists( $file->name, $unique_files )) {
                $unique_files[$file->name] = $run->run;
                $xtc_num_files++;
                $xtc_size += $file->size / (1024.0 * 1024.0 * 1024.0);
                if( !array_key_exists( $run->run, $runs2files )) $runs2files[$run->run] = array('run'=>$run->run,'xtc'=>array(),'hdf5'=>array());
                array_push( $runs2files[$run->run]['xtc'], $file );
            }
            if( $file->resource == 'hpss-resc'   ) $xtc_archived++;
            if( $file->resource == 'lustre-resc' ) $xtc_local_copy++;
        }
    }

    $hdf5_runs = null;
    FileMgrIrodsWs::runs( $hdf5_runs, $experiment->instrument()->name(), $experiment->name(), 'hdf5', $range_of_runs );
    foreach( $hdf5_runs as $run ) {
        $unique_files = array();  // per this run
        $files = $run->files;
        foreach( $files as $file ) {
            if( !array_key_exists( $file->name, $unique_files )) {
                $unique_files[$file->name] = $run->run;
                $hdf5_num_files++;
                $hdf5_size += $file->size / (1024.0 * 1024.0 * 1024.0);
                if( !array_key_exists( $run->run, $runs2files )) $runs2files[$run->run] = array('run'=>$run->run,'xtc'=>array(),'hdf5'=>array());
                array_push( $runs2files[$run->run]['hdf5'], $file );
            }
            if( $file->resource == 'hpss-resc'   ) $hdf5_archived++;
            if( $file->resource == 'lustre-resc' ) $hdf5_local_copy++;
        }
    }
    $success_encoded = json_encode( "success" );
    $updated_str     = json_encode( LusiTime::now()->toStringShort());
    $xtc_size_str    = json_encode( sprintf( "%.0f", $xtc_size ));
    $hdf5_size_str   = json_encode( sprintf( "%.0f", $hdf5_size ));
    $runs_encoded    = json_encode( $runs2files );

    $xtc_archived_html = json_encode(
    	$xtc_num_files == 0 ?
    	'n/a' : (	$xtc_num_files == $xtc_archived ?
    				'100%' :
    				'<span style="color:red;">'.$xtc_archived.'</span> / '.$xtc_num_files )
   	);
    $xtc_local_copy_html = json_encode(
    	$xtc_num_files == 0 ?
    	'n/a' : (	$xtc_num_files == $xtc_local_copy ?
			    	'100%' :
    				'<span style="color:red;">'.sprintf("%2.0f",100.0*$xtc_local_copy/$xtc_num_files).'%</span> ( '.$xtc_local_copy.' / '.$xtc_num_files.' )' )
   	);
    $hdf5_archived_html = json_encode(
    	$hdf5_num_files == 0 ?
   		'n/a' : (	$hdf5_num_files == $hdf5_archived ?
    				'100%' :
    				'<span style="color:red;">'.$hdf5_archived.'</span> / '.$hdf5_num_files )
   	);
    $hdf5_local_copy_html = json_encode(
    	$hdf5_num_files == 0 ?
   		'n/a' : (	$hdf5_num_files == $hdf5_local_copy ?
   					'100%' :
   					'<span style="color:red;">'.sprintf("%2.0f",100.0*$hdf5_local_copy/$hdf5_num_files).'%</span> ( '.$hdf5_local_copy.' / '.$hdf5_num_files.' )' )
    );

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{ "Status": {$success_encoded},
  "updated": {$updated_str},
  "summary": {
    "runs": {$num_runs},
    "min_run" : {$min_run},
    "max_run" : {$max_run},
    "xtc" : { "size": {$xtc_size_str},  "files": {$xtc_num_files},  "archived": {$xtc_archived},  "archived_html": {$xtc_archived_html},  "disk": {$xtc_local_copy},  "disk_html": {$xtc_local_copy_html} },
    "hdf5": { "size": {$hdf5_size_str}, "files": {$hdf5_num_files}, "archived": {$hdf5_archived}, "archived_html": {$hdf5_archived_html}, "disk": {$hdf5_local_copy}, "disk_html": {$hdf5_local_copy_html} }
  }
}
HERE;

    $logbook->commit();

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException  $e ) { report_error( $e->toHtml()); }

?>
