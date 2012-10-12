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
 * This script will process a request for retreiving all shifts of the experiments.
 * Return a JSON object with the descriptions of shifts.
 * Othersise return another JSON object with an explanation of a problem.
 */
if( !isset( $_GET['exper_id'] )) report_error('no experiment identifier parameter found in the request');
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' ) report_error( 'experiment identifier found in the request is empty' );

try {

    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $exper_id ) or report_error( 'no such experiment' );

    if( !LogBookAuth::instance()->canRead( $experiment->id())) report_error( 'You are not authorized to access any information about the experiment' );

    $max_total_seconds = 1;
    $shifts = array();
	foreach( $experiment->shifts() as $s ) {
		$total_seconds = ( is_null($s->end_time()) ? LusiTime::now()->sec : $s->end_time()->sec ) - $s->begin_time()->sec;
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

			$durat = sprintf( "%03d days, %02d:%02d.%02d", $day, $hour, $min, $sec );
		}

		// See if the 'Goals' record is found among messages
    	//
    	$entries = $experiment->search (
        	$s->id(),   // $shift_id=
        	null,           // $run_id=
        	'',             // $text2search
        	false,          // $search_in_messages=
        	true,           // $search_in_tags=
        	false,          // $search_in_values=
        	false,          // $posted_at_experiment=
        	true,           // $posted_at_shifts=
        	false,          // $posted_at_runs=
        	null,           // $begin=
        	null,           // $end=
        	'SHIFT_GOALS',  // $tag=
        	'',             // $author=
        	null            // $since=
    	);
    	$goals = '';
    	foreach( $entries as $e )
        	$goals .= $e->content();

        $runs = array();
		foreach( $s->runs() as $r )
			array_push(
		   		$runs,
	   			array(
	   				'id'  => $r->id(),
	   				'num' => $r->num()
	   			)
	   		);
	   	array_push(
	   		$shifts,
	   		array(
	   			'id'        => $s->id(),
	   			'begin'     => $s->begin_time()->toStringShort(),
	   			'begin_sec' => $s->begin_time()->sec,
	   			'end'       => is_null($s->end_time()) ? '<span style="color:red; font-weight:bold;">on-going</span>' : $s->end_time()->toStringShort(),
	   			'durat'     => $durat,
	   			'sec'       => $total_seconds,
	   			'leader'    => $s->leader(),
	   			'crew'      => $s->crew(),
	   			'goals'		=> $goals,
	   			'num_runs'  => $s->num_runs(),
	   			'runs'      => $runs
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
  "Shifts": [

HERE;


    $first = true;
    foreach( array_reverse($shifts) as $s ) {
		if( $first ) $first = false;
		else echo ',';
		echo json_encode( $s );
    }
    print <<< HERE
  ],
  "MaxSeconds" : {$max_total_seconds}
}
HERE;

    LogBook::instance()->commit();

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }

?>
