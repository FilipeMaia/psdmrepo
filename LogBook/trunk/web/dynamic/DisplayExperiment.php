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
        $experiment_status = '<b><i><em style="color:gray">Completed</em></i></b>';
    } else if( $status < 0 ) {
        $experiment_status = '<b><i><em style="color:green">In Preparation</em></i></b>';
    } else {
        $experiment_status = '<b><i><em style="color:red">Taking Data</em></i></b>';
    }

    $con = new RegDBHtml( 0, 0, 875, 140 );
    echo $con
        ->label    (   0,   0, 'Status:' )
        ->value    (  50,   0, $experiment_status )
        ->label    ( 175,   0, 'Begin Time:' )
        ->value    ( 260,   0, $experiment->begin_time()->toStringShort())
        ->label    ( 175,  25, 'End Time:'   )
        ->value    ( 260,  25, $experiment->end_time()->toStringShort())
        ->label    (   0,  60, 'Last Shift' )
        ->container(   0,  85, 'shifts_table' )
        ->label    ( 410,  60, 'Last Run' )
        ->container( 410,  85, 'runs_table' )
        ->button   ( 410,   0, 'detail_button', 'See Registration Info >' )
        ->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>