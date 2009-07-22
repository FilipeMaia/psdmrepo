<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying parameters of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );

$begin_str = '';
if( isset( $_GET['begin'] ))
    $begin_str = trim( $_GET['begin'] );

$end_str = '';
if( isset( $_GET['end'] ))
    $end_str = trim( $_GET['end'] );

$last_shift_requested = isset( $_GET['last'] );

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

function shift2json( $shift ) {

    $begin_time_url =
        "<a href=\"javascript:select_shift(".$shift->id().")\">".
        $shift->begin_time()->toStringShort().
        '</a>';
    $end_time_status =
        is_null( $shift->end_time()) ?
        '<b><em style="color:red;">on-going</em></b>' :
        $shift->end_time()->toStringShort();

    return json_encode(
        array (
            "begin_time" => $begin_time_url,
            "end_time"  => $end_time_status,
            "leader"  => $shift->leader(),
            "num_runs"  => $shift->num_runs()
        )
    );
}

/*
 * Return JSON objects with a list of experiments.
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
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

    if(( !is_null( $begin ) || !is_null( $end )) && $last_shift_requested )
        report_error( "conflicting options - last shift can't be requested along with begin or end times" );

    if( $last_shift_requested ) {
        $shifts = array( );
        $last_shift = $experiment->find_last_shift();
        if( !is_null( $last_shift ))
            array_push( $shifts, $last_shift );
    } else {
        $shifts = $experiment->shifts_in_interval( $begin, $end );
    }

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $shifts as $s ) {
      if( $first ) {
          $first = false;
          echo "\n".shift2json( $s );
      } else {
          echo ",\n".shift2json( $s );
      }
    }
    print <<< HERE
 ] } }
HERE;

    $logbook->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
