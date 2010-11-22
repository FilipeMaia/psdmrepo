<?php

require_once( 'LogBook/LogBook.inc.php' );
require_once( 'LusiTime/LusiTime.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDBHtml;
use RegDB\RegDBException;

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

    $experiment = $run->parent();
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
    $status = $run->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $run_status = '<b><em style="color:gray">Ended</em></b>';
    } else {
        $run_status = '<b><em style="color:red">Open</em></b>';
    }
    $end_time_str = is_null( $run->end_time()) ? $run_status : $run->end_time()->toStringShort();

    $prev_run = $run->parent()->find_prev_run_for( $run );
    if( is_null( $prev_run )) $prev_run_url = "&lt; Prev Run";
    else
        $prev_run_url =
            "<a href=\"javascript:select_run({$prev_run->shift()->id()},{$prev_run->id()})\" class=\"lb_link\">&lt; Prev Run</a>";

    $next_run = $run->parent()->find_next_run_for( $run );
    if( is_null( $next_run )) $next_run_url = "Next Run &gt;";
    else
        $next_run_url =
            "<a href=\"javascript:select_run({$next_run->shift()->id()},{$next_run->id()})\" class=\"lb_link\">Next Run &gt;</a>";

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 550, 80 );
    $con->label    (   0,   0, 'Status:' )
        ->value    (  50,   0, $run_status )
        ->label    ( 125,   0, 'Number:' )
        ->value    ( 210,   0, $run->num())
        ->label    ( 125,  20, 'Begin Time:' )
        ->value    ( 210,  20, $run->begin_time()->toStringShort())
        ->label    ( 365,   0, $prev_run_url, false )
        ->label    ( 445,   0, $next_run_url, false )
        ->label    ( 125,  40, 'End Time:'   )
        ->value    ( 210,  40, $end_time_str );

    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>