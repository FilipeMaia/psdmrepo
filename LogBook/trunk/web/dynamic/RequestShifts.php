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

$last_shift_requested = isset( $_GET['last'] );

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

    if( $last_shift_requested ) {
        $shifts = array( );
        $last_shift = $experiment->find_last_shift();
        if( !is_null( $last_shift ))
            array_push( $shifts, $last_shift );
    } else {
        $shifts = $experiment->shifts();
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
