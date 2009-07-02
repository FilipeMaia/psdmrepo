<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying runs of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );

$last_run_requested = isset( $_GET['last'] );

function run2json( $run ) {

    $begin_time_url =
        "<a href=\"javascript:select_run(".$run->id().")\">".
        $run->begin_time()->toStringShort().
        '</a>';
    $end_time_status =
        is_null( $run->end_time()) ?
        '<b><em style="color:red;">Taking Data</em></b>' :
        $run->end_time()->toStringShort();
    $shift_begin_time_url =
        "<a href=\"javascript:select_shift(".$run->shift()->id().")\">".
        $run->shift()->begin_time()->toStringShort().
        '</a>';

    return json_encode(
        array (
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

    $experiment = $logbook->find_experiment_by_id( $id )
        or die( "no such experiment" );

    if( $last_run_requested ) {
        $runs = array( );
        $last_run = $experiment->find_last_run();
        if( !is_null( $last_run ))
            array_push( $runs, $last_run );
    } else {
        $runs = $experiment->runs();
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

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
