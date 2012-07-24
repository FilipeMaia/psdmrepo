<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

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
 * This script will process a request for retreiving all runs of the experiments.
 * Return a JSON object with the descriptions of runs.
 * Othersise return another JSON object with an explanation of a problem.
 */
if( !isset( $_GET['exper_id'] )) report_error('no experiment identifier parameter found in the request');
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' ) report_error( 'experiment identifier found in the request is empty' );

$range_of_runs = '';
if( isset( $_GET['range_of_runs'] )) {
    $range_of_runs = trim( $_GET['range_of_runs'] );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $exper_id ) or report_error( 'no such experiment' );

    if( !LogBookAuth::instance()->canRead( $experiment->id())) report_error( 'You are not authorized to access any information about the experiment' );

    $first_run_num = null;
    $last_run_num  = $first_run_num;

    if( '' != $range_of_runs ) {

        /* Pasrse the run numbers first. If the parse succeeds and no last run
	   	 * is provided then assume the second run as the last one.
	   	 */
	   	if( strpos($range_of_runs, '-') === false ) {
	   		$r1 = $range_of_runs;
	   		$r2 = '';
	   	} else {
			list($r1,$r2) = explode( '-', $range_of_runs, 2 );
	   	}
		$r1 = trim( $r1 );
		$r2 = trim( $r2 );
		if( '' == $r1 ) report_error( "syntax error in the range of runs" );

		$first_run_num = null;
		if(( 1 != sscanf( $r1, "%d", $first_run_num )) or ( $first_run_num <= 0 ))
			report_error( "syntax error in the first run number of the range" );

		$last_run_num = $first_run_num;
		if( '' != $r2 )
			if(( 1 != sscanf( $r2, "%d", $last_run_num )) or ( $last_run_num <= 0 ))
				report_error( "syntax error in the last run number of the range" );

		if( $last_run_num < $first_run_num ) report_error( "last run in the range can't be less than the first one" );

		$first_run = $experiment->find_run_by_num( $first_run_num );
		if( is_null( $first_run )) report_error( "run {$first_run_num} can't be found" );
		$last_run = $experiment->find_run_by_num( $last_run_num );
		if( is_null( $last_run )) report_error( "run {$last_run_num} can't be found" );
    }

    $max_total_seconds = 1;
    $runs = array();
	foreach( $experiment->runs() as $r ) {

        // Skip runs which are not allowed by the filter (if any provid3ed)
        //
        if( !is_null($first_run_num)) {
            if(( $r->num() < $first_run_num ) || ( $r->num() > $last_run_num )) continue;
        }

		$total_seconds = is_null($r->end_time()) ? 0 : $r->end_time()->sec - $r->begin_time()->sec;
		if( $total_seconds > $max_total_seconds ) $max_total_seconds = $total_seconds;
		$durat = '';
		if( $total_seconds ) {
			$seconds_left = $total_seconds;

			$day          = floor( $seconds_left / ( 24 * 3600 ));
			$seconds_left = $seconds_left % ( 24 * 3600 );

			$hour         = floor( $seconds_left / 3600 );
			$seconds_left = $seconds_left % 3600;

			$min          = floor( $seconds_left / 60 );
			$seconds_left = $seconds_left % 60;

			$sec          = $seconds_left;

			$durat = sprintf( "%02d:%02d.%02d", $hour, $min, $sec );
		}
	   	array_push(
	   		$runs,
	   		array(
	   			'id'    => $r->id(),
	   			'num'   => $r->num(),
	   			'day'   => $r->begin_time()->toStringDay(),
	   			'ival'  => $r->begin_time()->toStringHMS().(is_null($r->end_time()) ? ' - <span style="color:red; font-weight:bold;">on-going</span>' : ' - '.$r->end_time()->toStringHMS()),
	   			'durat' => $durat,
	   			'sec'   => $total_seconds
	   		)
	   	);
    }

    $status_encoded = json_encode( "success" );
   	$updated_encoded = json_encode( LusiTime::now()->toStringShort());
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

   	print <<< HERE
{
  "Status": {$status_encoded},
  "Updated": {$updated_encoded},
  "Runs": [

HERE;


    $first = true;
    foreach( array_reverse($runs) as $r ) {
		if( $first ) $first = false;
		else echo ',';
		echo json_encode( $r );
    }
    print <<< HERE
  ],
  "MaxSeconds" : {$max_total_seconds}
}
HERE;

    $logbook->commit();

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }

?>
