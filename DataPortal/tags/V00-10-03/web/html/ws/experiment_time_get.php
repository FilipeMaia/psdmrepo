<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use DataPortal\ExpTimeMon;

use LogBook\LogBook;

use LusiTime\LusiTime;

use RegDB\RegDB;

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

    $start = null;
    $stop  = null;

    // Process input parameters first
    //
    if( isset($_GET[ 'shift'] )) {
        $shift = LusiTime::parse(trim( $_GET[ 'shift'].' 00:00:00' ));

        $delta_sec = 0;
        if( isset($_GET[ 'delta'] )) $delta_sec = (int)trim( $_GET[ 'delta']);

        $start = new LusiTime($shift->sec + $delta_sec, $shift->nsec);
        $stop  = new LusiTime($start->sec + 24 * 3600,  $start->nsec);

    } else {
        if( !isset($_GET[ 'start'] )) report_error( "parameter 'start' not found" );
        $start = LusiTime::parse(trim( $_GET[ 'start']));
        if( is_null($start)) report_error( "parameter 'start' has invalid value" );
        if( isset($_GET[ 'stop'] )) {
            $stop = LusiTime::parse(trim( $_GET[ 'stop']));
            if( is_null($stop)) report_error( "parameter 'stop' has invalid value" );
        } else {
            $stop = LusiTime::now();
        }
        if( $start->greaterOrEqual($stop)) report_error( "parameter 'start' can't have a value greater or equal to the one of 'stop'" );
    }
    $start_sec = $start->to_float();
    $stop_sec  = $stop->to_float();
 
    $now_sec = LusiTime::now()->to_float();

    // Fetch data from the database, then filter and extract the information
    // which is expected to be returned to a caller.
    //
    LogBook::instance()->begin();
    RegDB::instance()->begin();
    ExpTimeMon::instance()->begin();

    // Repopulate the database if the requested day is today
    //
    $shift_is4today = $start->toStringDay() == LusiTime::today()->toStringDay();
    if( $shift_is4today ) {
        ExpTimeMon::instance()->populate('XRAY_DESTINATIONS');
    }

    // Instrument specific statistics stored in the following dictionary
    //
    $instruments = array();
    foreach( ExpTimeMon::instrument_names() as $instr ) {
        $instruments[$instr] = array(
            'runs'                      => array(),
            'runs_duration_sec'         => 0.0,
            'beam_status'               => array(),
            'total_beam_time_sec'       => 0.0,
            'total_beam_time'           => '00 hr 00 min',
            'gaps'                      => array(),
            'gaps_duration_sec'         => 0.0,
            'total_data_taking'         => '00 hr 00 min',
            'total_data_taking_percent' => '0.0'
        );
    }

    // Beam destination statistics
    //
    $beam_destinations = array();
    foreach( ExpTimeMon::beam_destinations() as $name ) {
        $beam_destinations[$name] = array(
            'beam_status' => array()
        );
    }
    $total_beam_destinations_sec = 0.0;
 
    // LCLS statistics
    //
    $lcls_status         = array();
    $total_lcls_time_sec = 0.0;

    // Fetch and process runs intersecting the requested interval.
    // 
    // NOTE: runs are reported in a scope of the corresponding
    //       instruments.
    //
    foreach( ExpTimeMon::instance()->beamtime_runs($start,$stop) as $run ) {
        $instr_name = $run->instr_name();
        array_push(
            $instruments[$instr_name]['runs'],
            array(
                'begin_rel2start_sec' => $run->begin_time()->to_float() - $start_sec,
                'end_rel2start_sec'   => $run->end_time()->to_float()   - $start_sec,
                'exper_id'            => $run->exper_id(),
                'runnum'              => $run->runnum(),
                'instr_name'          => $instr_name,
                'exper_name'          => $run->exper_name()));

        $instruments[$instr_name]['runs_duration_sec'] += $run->end_time()->to_float() - $run->begin_time()->to_float();
    }

    // Process gaps intersecting the requested interval.
    // Also calculate the total duration of gaps within the interval.
    // 
    // NOTE: gaps are reported in a scope of the corresponding
    //       instruments.
    //
    foreach( ExpTimeMon::instance()->beamtime_gaps($start,$stop) as $gap ) {

        $instr = $gap->instr_name();

        $comment      = ExpTimeMon::instance()->beamtime_comment_at($gap->begin_time(), $instr);
        $comment_info = is_null($comment) ?
            array('available'     => 0) :
            array('available'     => 1,
                  'instr_name'    => $comment->instr_name(),
                  'comment'       => $comment->comment(),
                  'system'        => $comment->system(),
                  'posted_by_uid' => $comment->posted_by_uid(),
                  'post_time'     => $comment->post_time()->toStringShort());

        $gap_begin_sec = $gap->begin_time()->to_float();
        $gap_end_sec   = $gap->end_time()->to_float();
        array_push(
            $instruments[$instr]['gaps'],
            array(
                'begin_time_64'       => $gap->begin_time()->to64(),
                'begin_rel2start_sec' => $gap_begin_sec - $start_sec,
                'end_rel2start_sec'   => $gap_end_sec   - $start_sec,
                'comment'             => $comment_info ));
        $instruments[$instr]['gaps_duration_sec'] += $gap_end_sec - $gap_begin_sec;
    }

    // This is just a dictionary of known categories which have been
    // found in justtifications for gaps. The categories can be used
    // by any instruments.
    //
    $systems = ExpTimeMon::instance()->beamtime_systems();
    sort($systems);

    // Retreive and process beam-time and LCLS records
    // 
    // Please, note two exceptions:
    // 
    // - for any future shifts we're taking a side track as a workaround
    //   of a false positive beam status info reported by the API. The API
    //   always extends the very last record into the future w/o worring
    //   about a context in which this information is being requested.
    //
    // - we cut short the remainig of the today's shift after the current
    //   time as if there were no beam after now. And this is based on the
    //   same rationale as explained above for the future shifts.
    //
    if( $now_sec > $start_sec ) {

        // Beam time statistics
        //
        // NOTE: gaps are reported in a scope of the corresponding
        //       instruments.
        //
        foreach( ExpTimeMon::instance()->beamtime_beam_status('XRAY_DESTINATIONS',$start,$stop) as $ival ) {

            $ival_begin_sec = $ival['begin_time']->to_float();
            $ival_end_sec   = $ival['end_time'  ]->to_float();

            // Check if we are generating this report for today, and if we should
            // cut this report short for the rest of today.
            //
            $cut_short = false;
            if( $shift_is4today ) {
                if(( $ival_begin_sec <= $now_sec ) && ( $now_sec < $ival_end_sec )) {
                    $ival_end_sec = $now_sec;
                    $cut_short = true;
                }
            }
            foreach( ExpTimeMon::beam_destinations() as $name ) {

                $status = $ival['status'] & ExpTimeMon::beam_destination_mask($name);
                if( $status ) {
                    if( ExpTimeMon::is_instrument_name($name)) {
                        $instruments[$name]['total_beam_time_sec'] += $ival_end_sec - $ival_begin_sec;
                        array_push(
                            $instruments[$name]['beam_status'],
                            array(
                                'status' => $status,
                                'begin_rel2start_sec' => $ival_begin_sec - $start_sec,
                                'end_rel2start_sec'   => $ival_end_sec   - $start_sec));
                    }
                    $total_beam_destinations_sec += $ival_end_sec - $ival_begin_sec;
                    array_push(
                        $beam_destinations[$name]['beam_status'],
                        array(
                            'status' => $status,
                            'begin_rel2start_sec' => $ival_begin_sec - $start_sec,
                            'end_rel2start_sec'   => $ival_end_sec   - $start_sec));
                }
            }
            if( $shift_is4today && $cut_short ) break;  // Finish here  since we don't know the beam status
                                                        // for rest of today's day (starting from now).
        }

        // LCLS statistics
        //
        foreach( ExpTimeMon::instance()->beamtime_beam_status('LIGHT:LCLS:STATE',$start,$stop) as $ival ) {

            $ival_begin_sec = $ival['begin_time']->to_float();
            $ival_end_sec   = $ival['end_time'  ]->to_float();

            // Check if we are generating this report for today, and if we should
            // cut this report short for the rest of today.
            //
            $cut_short = false;
            if( $shift_is4today ) {
                if(( $ival_begin_sec <= $now_sec ) && ( $now_sec < $ival_end_sec )) {
                    $ival_end_sec = $now_sec;
                    $cut_short = true;
                }
            }

            $status = $ival['status'];
            if( $status ) {
                array_push(
                    $lcls_status,
                    array(
                        'status' => $status,
                        'begin_rel2start_sec' => $ival_begin_sec - $start_sec,
                        'end_rel2start_sec'   => $ival_end_sec   - $start_sec));

                $total_lcls_time_sec += $ival_end_sec - $ival_begin_sec;
            }
            if( $shift_is4today && $cut_short ) break;  // Finish here since we don't know the beam status
                                                        // for the rest of today's day (starting from now).
        }
    }
    $total_beam_time_sec     = 0.0;
    $total_runs_duration_sec = 0.0;
    foreach( ExpTimeMon::instrument_names() as $instr ) {

        $total_hutch_beam_time_sec = $instruments[$instr]['total_beam_time_sec'];
        $total_beam_time_sec += $total_hutch_beam_time_sec;

        $hours   = floor(  $total_hutch_beam_time_sec / 3600. );
        $minutes = floor(( $total_hutch_beam_time_sec % 3600. ) / 60. );
        $instruments[$instr]['total_beam_time'] = sprintf("%02d hr %02d min", $hours, $minutes );

        $total_data_taking_percent = 0.;
        $runs_duration_sec = $instruments[$instr]['runs_duration_sec'];
        $total_runs_duration_sec += $runs_duration_sec;
        if( $total_hutch_beam_time_sec > 0.) {
            $total_data_taking_percent =  100. * $runs_duration_sec / $total_hutch_beam_time_sec;
            if( $total_data_taking_percent > 100. ) $total_data_taking_percent = 100.;
        }
        $hours   = floor(  $runs_duration_sec / 3600. );
        $minutes = floor(( $runs_duration_sec % 3600. ) / 60. );
        $instruments[$instr]['total_data_taking'] = sprintf("%02d hr %02d min", $hours, $minutes );
        $instruments[$instr]['total_data_taking_percent'] = sprintf("%5.1f", $total_data_taking_percent);
    }
    $total_data_taking_percent = 0.0;
    if( $total_beam_time_sec > 0.) {
        $total_data_taking_percent =  100. * $total_runs_duration_sec / $total_beam_time_sec;
        if( $total_data_taking_percent > 100. ) $total_data_taking_percent = 100.;
    }
    $hours   = floor(  $total_beam_time_sec / 3600. );
    $minutes = floor(( $total_beam_time_sec % 3600. ) / 60. );
    $total_beam_time = sprintf("%02d hr %02d min", $hours, $minutes );

    $hours   = floor(  $total_beam_destinations_sec / 3600. );
    $minutes = floor(( $total_beam_destinations_sec % 3600. ) / 60. );
    $total_beam_destinations = sprintf("%02d hr %02d min", $hours, $minutes );

    $hours   = floor(  $total_runs_duration_sec / 3600. );
    $minutes = floor(( $total_runs_duration_sec % 3600. ) / 60. );
    $total_data_taking = sprintf("%02d hr %02d min", $hours, $minutes );

    LogBook::instance()->commit();
    RegDB::instance()->commit();
    ExpTimeMon::instance()->commit();

    report_success( array(
        'shift'                       => $start->toStringDay(),
        'start'                       => $start->toStringShort(),
        'stop'                        => $stop->toStringShort(),
        'start_sec'                   => $start_sec,
        'stop_sec'                    => $stop_sec,
        'instrument_names'            => ExpTimeMon::instrument_names(),
        'beam_destination_names'      => ExpTimeMon::beam_destinations(),
        'beam_destinations'           => $beam_destinations,
        'total_beam_destinations'     => $total_beam_destinations,
        'total_beam_destinations_sec' => $total_beam_destinations_sec,
        'lcls_status'                 => $lcls_status,
        'total_beam_time_sec'         => $total_beam_time_sec,
        'total_beam_time'             => $total_beam_time,
        'total_data_taking'           => $total_data_taking,
        'total_data_taking_percent'   => sprintf("%5.1f", $total_data_taking_percent),
        'instruments'                 => $instruments,
        'systems'                     => $systems
    ));

} catch( Exception $e ) { report_error( '<pre>'.print_r($e, true).'</pre>'); }
  
?>
