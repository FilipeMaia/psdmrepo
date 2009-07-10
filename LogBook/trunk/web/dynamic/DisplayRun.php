<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying a status of a run.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "run identifier can't be empty" );
} else
    die( "no valid run identifier" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $run = $logbook->find_run_by_id( $id )
        or die( "no such run" );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $status = $run->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $run_status = '<b><i><em style="color:gray">Ended</em></i></b>';
    } else {
        $run_status = '<b><i><em style="color:red">on-going</em></i></b>';
    }
    $end_time_str = is_null( $run->end_time()) ? '' : $run->end_time()->toStringShort();

    $con = new RegDBHtml( 0, 0, 875, 50 );
    $con->label    (   0,   0, 'Status:' )
        ->value    (  50,   0, $run_status )
        ->label    ( 125,   0, 'Begin Time:' )
        ->value    ( 210,   0, $run->begin_time()->toStringShort())
        ->button   ( 365,   0, 'prev_run_button', '<b>&lt; See Previous Run</b>' )
        ->button   ( 515,   0, 'next_run_button', '<b>See Next Run &gt;</b>' )
        ->label    ( 125,  25, 'End Time:'   )
        ->value    ( 210,  25, $end_time_str );

    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>