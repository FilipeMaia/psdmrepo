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

    $status = $shift->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $shift_status = '<b><em style="color:gray">Ended</em></b>';
    } else {
        $shift_status = '<b><em style="color:red">Open</em></b>';
    }

    $prev_shift = $shift->parent()->find_prev_shift_for( $shift );
    if( is_null( $prev_shift )) $prev_shift_url = "&lt; Prev Shift";
    else
        $prev_shift_url = "<a href=\"javascript:select_shift({$prev_shift->id()})\">&lt; Prev Shift</a>";

    $next_shift = $shift->parent()->find_next_shift_for( $shift );
    if( is_null( $next_shift )) $next_shift_url = "Next Shift &gt;";
    else
        $next_shift_url = "<a href=\"javascript:select_shift({$next_shift->id()})\">Next Shift &gt;</a>";

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 550, 80 );
    $con->label    (   0,   0, 'Status:' )
        ->value    (  50,   0, $shift_status )
        ->label    ( 125,   0, 'Leader:' )
        ->value    ( 210,   0, $shift->leader())
        ->label    ( 125,  20, 'Begin Time:' )
        ->value    ( 210,  20, $shift->begin_time()->toStringShort())
        ->label    ( 365,   0, $prev_shift_url, false )
        ->label    ( 445,   0, $next_shift_url, false )
        ->label    ( 125,  40, 'End Time:'   );
    if( is_null( $shift->end_time())) { $con
        ->value    ( 210,  40, $shift_status )
        ->button   ( 445,  35, 'close_shift_button', 'Close...' );
    } else { $con
        ->value    ( 210,  40, $shift->end_time()->toStringShort());
    }
    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>