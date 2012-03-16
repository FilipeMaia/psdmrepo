<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\SysMon;
use DataPortal\DataPortalException;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

/**
 * This service will return a data structure representing LCLS beam
 * time usage for the specified shift. The shift is defined as
 * a 24 hours interval staring at midnight of the shift day.
 * 
 * The shift is expcted to have the following format (no hours, minutes or
 * seconds, etc. after a day):
 * 
 *   2012-03-21
 */
function report_error($msg) {
	return_result(
        array(
            'status' => 'error',
            'message' => $msg
        )
    );
}
function report_success($result) {
    $result['status'] = 'success';
  	return_result($result);
}
function return_result($result) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

    echo json_encode($result);
	exit;
}

try {

    // Process input parameters first
    //
    if( !isset($_GET[ 'shift'] )) report_error( 'no shift parameter found' );
    $shift = LusiTime::parse(trim( $_GET[ 'shift'].' 00:00:00' ));

    $delta_sec = 0;
    if( isset($_GET[ 'delta'] )) $delta_sec = (int)trim( $_GET[ 'delta']);

    $start = new LusiTime($shift->sec + $delta_sec, $shift->nsec);
    $stop  = new LusiTime($start->sec + 24 * 3600,  $start->nsec);

    $start_sec = $start->to_float();
    $stop_sec  = $stop->to_float();
 
    // Fetch data from the database, then filter and extract the information
    // which is expected to be returned to a caller.
    //
    $logbook = new LogBook();
	$logbook->begin();

	$regdb = new RegDB();
	$regdb->begin();

    $sysmon = SysMon::instance();
	$sysmon->begin();

    $instrument_names = array('AMO', 'XPP', 'SXR', 'CXI', 'XCS', 'MEC');
    /*
	foreach( $logbook->regdb()->instruments() as $instrument )
		if( !$instrument->is_location())
			array_push( $instrument_names, $instrument->name());

	sort( $instrument_names );
    */
    // Fetch and process runs intersecting the requested interval
    //
    $runs = array();
    foreach( $sysmon->beamline_runs($start,$stop) as $run )
        array_push(
            $runs,
            array(
                'begin_rel2start_sec' => $run->begin_time()->to_float() - $start_sec,
                'end_rel2start_sec'   => $run->end_time()->to_float()   - $start_sec,
                'exper_id'            => $run->exper_id(),
                'runnum'              => $run->runnum(),
                'instr_name'          => $run->instr_name(),
                'exper_name'          => $run->exper_name()));

    // Process gaps between intersecting the requested interval.
    // Also calculate the total duration of gaps within the interval.
    //
    $gaps = array();
    $gaps_duration_sec = 0.0;
    foreach( $sysmon->beamline_gaps($start,$stop) as $gap ) {
        $comment = $sysmon->beamtime_comment_at($gap->begin_time());
        $comment_info = is_null($comment) ?
            array('available'     => 0) :
            array('available'     => 1,
                  'comment'       => $comment->comment(),
                  'system'        => $comment->system(),
                  'posted_by_uid' => $comment->posted_by_uid(),
                  'post_time'     => $comment->post_time()->toStringShort());
        $gap_begin_sec = $gap->begin_time()->to_float();
        $gap_end_sec   = $gap->end_time()->to_float();
        array_push(
            $gaps,
            array(
                'begin_time_64'       => $gap->begin_time()->to64(),
                'begin_rel2start_sec' => $gap_begin_sec - $start_sec,
                'end_rel2start_sec'   => $gap_end_sec   - $start_sec,
                'comment'             => $comment_info
            )
        );
        $gaps_duration_sec += $gap_end_sec - $gap_begin_sec;
    }

    $systems = $sysmon->beamline_systems();
    sort($systems);

    // TODO: Fetch this information from the database as soon as it will
    //       be available.
    //
    $beam = '100.0';

    $logbook->commit();
    $regdb->commit();
	$sysmon->commit();

    report_success( array(
        'shift'            => $start->toStringDay(),
        'instrument_names' => $instrument_names,
        'runs'      => $runs,
        'gaps'      => $gaps,
        'systems'   => $systems,
        'beam'      => $beam,
        'usage'     => sprintf("%5.1f", 100. * (1. - ( $gaps_duration_sec / ( $stop_sec - $start_sec)))),
        'start_sec' => $start_sec,
        'stop_sec'  => $stop_sec
    ));

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( DataPortalException $e ) { report_error( $e->toHtml()); }
  catch( LogBookException    $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( RegDBException      $e ) { report_error( $e->toHtml()); }
  catch( Exception           $e ) { report_error( "{$e}" );      }
  
?>
