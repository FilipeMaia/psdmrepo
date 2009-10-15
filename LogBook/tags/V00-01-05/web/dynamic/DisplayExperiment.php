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

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Proceed to the operation
    //
    $shift = $experiment->find_last_shift();
    $run = $experiment->find_last_run();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $status = $experiment->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $experiment_status = '<b><i><em style="color:gray">CLOSED</em></i></b>';
    } else if( $status < 0 ) {
        $experiment_status = '<b><i><em style="color:green">PREPARATION</em></i></b>';
    } else {
        $experiment_status = '<b><em style="color:red">Active</em></b>';
    }

    $all_shifts_url = "<a href=\"javascript:list_shifts()\" class=\"lb_link\">See List of all shifts</a>";
    $shift_url = 'See last shift';
    if( !is_null( $shift )) {
        $shift_url = "<a href=\"javascript:select_shift({$shift->id()})\" class=\"lb_link\">See last shift</a>";
    }
    $all_runs_url = "<a href=\"javascript:list_runs()\" class=\"lb_link\">List of all runs</a>";
    $run_url = 'See last run';
    if( !is_null( $run )) {
        $run_url = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">See last run</a>";
    }


    $con = new RegDBHtml( 0, 0, 900, 250 );
    echo $con
        ->label       ( 400,   0, 'Description')
        //->textbox   ( 400,  20, $experiment->description(), 500, 140 )
        ->container_1 ( 400,  20, '<pre style="padding:4px; font-size:14px;">'.$experiment->description().'</pre>', 500, 140, false, '#efefef' )

        ->label       (   0,   0, 'Status:'     )->value(  50,   0, $experiment_status )
        ->label       ( 150,   0, 'Begin Time:' )->value( 250,   0, $experiment->begin_time()->toStringShort())
        ->label       ( 150,  20, 'End Time:'   )->value( 250,  20, $experiment->end_time()->toStringShort())

        ->label       ( 150,  60, 'Leader:'     )->value( 250,  60, $experiment->leader_account())
        ->label       ( 150,  80, 'POSIX Group:')->value( 250,  80, $experiment->POSIX_gid())

        ->label       ( 150, 120, $shift_url, false )
        ->label       ( 150, 140, $all_shifts_url, false )
        ->label       ( 150, 180, $run_url, false  )
        ->label       ( 150, 200, $all_runs_url, false )

        ->label       ( 400, 180, 'Contact Info')
        ->container_1 ( 400, 200, '<pre style="padding:4px; font-size:14px;">'.$experiment->contact_info().'</pre>', 500, 50, false, '#efefef' )

        ->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>