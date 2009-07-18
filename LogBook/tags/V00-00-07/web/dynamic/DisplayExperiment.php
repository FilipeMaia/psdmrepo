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

    $shift = $experiment->find_last_shift();
    $run = $experiment->find_last_run();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $reginstration_url =
        "<a href=\"javascript:window.open('/tests/RegDB/dynamic/index.php?".
        "action=view_experiment&id={$experiment->id()}&name={$experiment->name()}')\">Experiment Registration Info &gt;</a>";

    $status = $experiment->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $experiment_status = '<b><i><em style="color:gray">CLOSED</em></i></b>';
    } else if( $status < 0 ) {
        $experiment_status = '<b><i><em style="color:green">PREPARATION</em></i></b>';
    } else {
        $experiment_status = '<b><em style="color:red">Active</em></b>';
    }

    $all_shifts_url = "( <a href=\"javascript:list_shifts()\">see all &gt;</a> )";
    $shift_url = 'Last Shift &gt;';
    $shift_leader = '';
    $shift_begin_time = '';
    $shift_end_time = '';
    if( !is_null( $shift )) {
        $shift_url = "<a href=\"javascript:select_shift({$shift->id()})\">Last Shift &gt;</a>";
        $shift_leader = $shift->leader();
        $shift_begin_time = $shift->begin_time()->toStringShort();
        $shift_end_time = is_null( $shift->end_time()) ? '<b><em style="color:red">Open</em></b>' : $shift->end_time()->toStringShort();
    }
    $all_runs_url = "( <a href=\"javascript:list_runs()\">see all &gt;</a> )";
    $run_url = 'Last Run &gt;';
    $run_number = '';
    $run_begin_time = '';
    $run_end_time = '';
    if( !is_null( $run )) {
        $run_number = $run->num();
        $run_url = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\">Last Run &gt;</a>";
        $run_begin_time = $run->begin_time()->toStringShort();
        $run_end_time = is_null( $run->end_time()) ? '<b><em style="color:red">Open</em></b>' : $run->end_time()->toStringShort();
    }


    $con = new RegDBHtml( 0, 0, 695, 140 );
    echo $con
        ->label    ( 380,   0, $reginstration_url, false  )
        ->label    (   0,   0, 'Status:'     )->value(  50,   0, $experiment_status )
        ->label    ( 150,   0, 'Begin Time:' )->value( 235,   0, $experiment->begin_time()->toStringShort())
        ->label    ( 150,  20, 'End Time:'   )->value( 235,  20, $experiment->end_time()->toStringShort())

        ->label    (  50,  60, $shift_url, false )
        ->label    ( 150,  60, 'Leader:'     )->value( 235,  60, $shift_leader )
        ->label    ( 150,  80, 'Begin Time:' )->value( 235,  80, $shift_begin_time )
        ->label    (  50,  80, $all_shifts_url, false )
        ->label    ( 150, 100, 'End Time:'   )->value( 235, 100, $shift_end_time )

        ->label    ( 380,  60, $run_url, false  )
        ->label    ( 480,  60, 'Number:'     )->value( 565,  60, $run_number )
        ->label    ( 480,  80, 'Begin Time:' )->value( 565,  80, $run_begin_time )
        ->label    ( 380,  80, $all_runs_url, false )
        ->label    ( 480, 100, 'End Time:'   )->value( 565, 100, $run_end_time )

        ->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>