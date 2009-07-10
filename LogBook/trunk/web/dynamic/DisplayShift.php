<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying a status of a shift.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "shift identifier can't be empty" );
} else
    die( "no valid shift identifier" );


/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $shift = $logbook->find_shift_by_id( $id )
        or die( "no such shift" );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $status = $shift->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $shift_status = '<b><i><em style="color:gray">Ended</em></i></b>';
    } else {
        $shift_status = '<b><i><em style="color:red">on-going</em></i></b>';
    }

    $con = new RegDBHtml( 0, 0, 875, 105 );
    $con->label    (   0,   0, 'Status:' )
        ->value    (  50,   0, $shift_status )
        ->label    ( 125,   0, 'Begin Time:' )
        ->value    ( 210,   0, $shift->begin_time()->toStringShort())
        ->button   ( 365,   0, 'prev_shift_button', '<b>&lt; See Previous Shift</b>' )
        ->button   ( 515,   0, 'next_shift_button', '<b>See Next Shift &gt;</b>' )
        ->label    ( 125,  25, 'End Time:'   );
    if( is_null( $shift->end_time())) { $con
        ->button   ( 210,  20, 'close_shift_button', '<b>Close This Shift</b>' );
    } else { $con
        ->value    ( 210,  25, $shift->end_time()->toStringShort());
    }
    $con->label    ( 125,  50, 'Leader:' )
        ->value    ( 210,  50, $shift->leader())
        ->label    (   0,  80, 'Runs' );

    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>