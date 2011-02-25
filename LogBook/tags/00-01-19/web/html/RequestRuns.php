<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request for displaying runs of an experiment.
 */
$exper_id = null;
$shift_id = null;

if( isset( $_GET['id'] )) {
    $exper_id = trim( $_GET['id'] );
    if( $exper_id == '' )
        die( "experiment identifier can't be empty" );

} else if( isset( $_GET['shift_id'] )) {
    $shift_id = trim( $_GET['shift_id'] );
    if( $shift_id == '' )
        die( "shift identifier can't be empty" );

} else {
    die( "no definitive scope to search for runs" );
}
$last_run_requested = isset( $_GET['last'] );

$begin_str = '';
if( isset( $_GET['begin'] ))
    $begin_str = trim( $_GET['begin'] );

$end_str = '';
if( isset( $_GET['end'] ))
    $end_str = trim( $_GET['end'] );

/* Translate timestamps which may also contain shortcuts
 */
function translate_time( $experiment, $str ) {
    $str_trimmed = trim( $str );
    if( $str_trimmed == '' ) return null;
    switch( $str_trimmed[0] ) {
        case 'b':
        case 'B': return $experiment->begin_time();
        case 'e':
        case 'E': return $experiment->end_time();
        case 'm':
        case 'M': return LusiTime::minus_month();
        case 'w':
        case 'W': return LusiTime::minus_week();
        case 'd':
        case 'D': return LusiTime::minus_day();
        case 'y':
        case 'Y': return LusiTime::yesterday();
        case 't':
        case 'T': return LusiTime::today();
        case 'h':
        case 'H': return LusiTime::minus_hour();
    }
    $result = LusiTime::parse( $str_trimmed );
    if( is_null( $result )) $result = LusiTime::from64( $str_trimmed );
    return $result;
}

function run2json( $run ) {

    $begin_time_url =
        "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">".
        $run->begin_time()->toStringShort().
        '</a>';
    $end_time_status =
        is_null( $run->end_time()) ?
        '<b><em style="color:red;">on-going</em></b>' :
        $run->end_time()->toStringShort();
    $shift_begin_time_url =
        "<a href=\"javascript:select_shift({$run->shift()->id()})\" class=\"lb_link\">".
        $run->shift()->begin_time()->toStringShort().
        '</a>';

    return json_encode(
        array (
            "id"  => $run->id(),
            "shift_id"  => $run->shift()->id(),
            "num"  => $run->num(),
            "begin_time" => $begin_time_url,
            "end_time" => $end_time_status,
            "shift_begin_time" => $shift_begin_time_url
        )
    );
}

/*
 * Return JSON objects with a list of experiments.
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    if( !is_null( $exper_id )) {
        $experiment = $logbook->find_experiment_by_id( $exper_id )
            or die( "no such experiment" );

        // Timestamps are translated here because of possible shoftcuts which
        // may reffer to the experiment's validity limits.
        //
        $begin = null;
        if( $begin_str != '' ) {
            $begin = translate_time( $experiment, $begin_str );
            if( is_null( $begin ))
                report_error( "begin time has invalid format" );
        }
        $end = null;
        if( $end_str != '' ) {
            $end = translate_time( $experiment, $end_str );
            if( is_null( $end ))
                report_error( "end time has invalid format" );
        }
        if( !is_null( $begin ) && !is_null( $end ) && !$begin->less( $end ))
            report_error( "invalid interval - begin time isn't strictly less than the end one" );

        if(( !is_null( $begin ) || !is_null( $end )) && $last_run_requested )
            report_error( "conflicting options - last run can't be requested along with begin or end times" );

        if( $last_run_requested ) {
            $runs = array( );
            $last_run = $experiment->find_last_run();
            if( !is_null( $last_run ))
                array_push( $runs, $last_run );
        } else {
            $runs = $experiment->runs_in_interval( $begin, $end );
        }

    } else if( !is_null( $shift_id )) {
        $shift = $logbook->find_shift_by_id( $shift_id )
            or die( "no such shift" );
            $runs = $shift->runs();
        $experiment = $shift->parent();
    } else {
        die( "internal implementation error" );
    }
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Proceed to the operation
    //
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $runs as $r ) {
      if( $first ) {
          $first = false;
          echo "\n".run2json( $r );
      } else {
          echo ",\n".run2json( $r );
      }
    }
    print <<< HERE
 ] } }
HERE;

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
