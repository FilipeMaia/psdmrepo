<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying a status of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );


/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $status = $experiment->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $experiment_status = '<b><i><em style="color:gray">completed</em></i></b>';
    } else if( $status < 0 ) {
        $experiment_status = '<b><i><em style="color:green">in preparation</em></i></b>';
    } else {
        $experiment_status = '<b><i><em style="color:red">on-going</em></i></b>';
    }

    $con = new RegDBHtml( 0, 0, 925, 140 );
    echo $con
        ->label    (   0,   0, 'Status:' )
        ->value    (  50,   0, $experiment_status )
        ->label    ( 150,   0, 'Begin Time:' )
        ->value    ( 235,   0, $experiment->begin_time()->toStringShort())
        ->label    ( 150,  25, 'End Time:' )
        ->value    ( 235,  25, $experiment->end_time()->toStringShort())
        ->label    (   0,  60, 'Last Shift' )
        ->container(   0,  85, 'shifts_table' )
        ->label    ( 450,  60, 'Last Run' )
        ->container( 450,  85, 'runs_table' )
        ->button   ( 450,   0, 'detail_button', '<b>Experiment Registration Info &gt;</b>' )
        ->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>