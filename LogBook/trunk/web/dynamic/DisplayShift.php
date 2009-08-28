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
    $end_time_str = is_null( $shift->end_time()) ? $shift_status : $shift->end_time()->toStringShort();

    $prev_shift_url = "&lt; Prev shift";
    $prev_shift = $shift->parent()->find_prev_shift_for( $shift );
    if( !is_null( $prev_shift ))
        $prev_shift_url = "<a href=\"javascript:select_shift({$prev_shift->id()})\" class=\"lb_link\">{$prev_shift_url}</a>";

    $next_shift_url = "Next shift &gt;";
    $next_shift = $shift->parent()->find_next_shift_for( $shift );
    if( !is_null( $next_shift ))
        $next_shift_url = "<a href=\"javascript:select_shift({$next_shift->id()})\" class=\"lb_link\">{$next_shift_url}</a>";

    $begin_shift_url = "";
    $close_shift_url = "";

    // Disabled this check because we may want to have shifts beyond the data taking
    // period of an experiment.
    //
    // if( $status <= 0 && $shift->parent()->in_interval( LusiTime::now()) == 0 ) {
    if( $status <= 0 ) {
        $begin_shift_url = "<a href=\"javascript:begin_new_shift()\" class=\"lb_link\">Begin new shift</a>";
        $close_shift_url = "<a href=\"javascript:close_shift({$shift->id()})\" class=\"lb_link\">Close current shift</a>";
    }

    $crew = '';
    foreach( $shift->crew() as $m )
        $crew .= ' '.$m;


    // See if the 'Goals' record is found among messages
    //
    $experiment = $shift->parent();
    $entries = $experiment->search (
        $shift->id(),   // $shift_id=
        null,           // $run_id=
        '',             // $text2search
        false,          // $search_in_messages=
        true,           // $search_in_tags=
        false,          // $search_in_values=
        false,          // $posted_at_experiment=
        true,           // $posted_at_shifts=
        false,          // $posted_at_runs=
        null,           // $begin=
        null,           // $end=
        'SHIFT_GOALS',  // $tag=
        '',             // $author=
        null            // $since=
    );
    $goals = '';
    foreach( $entries as $e )
        $goals .= $e->content();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 710 ,200 );
    $con->label    (   0,   0, 'Status:'     )->value(  50,   0, $shift_status )
        ->label    ( 125,   0, 'Begin Time:' )->value( 210,   0, $shift->begin_time()->toStringShort())
        ->label    ( 355,   0, $prev_shift_url,  false )
        ->label    ( 440,   0, $next_shift_url,  false );
    if( LogBookAuth::isAuthenticated())
    $con->label    ( 540,   0, $begin_shift_url, false )
        ->label    ( 540,  20, $close_shift_url, false );
    $con->label    ( 125,  20, 'End Time:'   )->value( 210,  20, $end_time_str )
        ->label    ( 125,  60, 'Leader:'     )->value( 210,  60, $shift->leader())
        ->label    ( 125,  80, 'Shift Crew:' )->value( 210,  80, $crew )
        ->label    ( 125, 120, 'Goals')
        ->container_1 ( 125, 140, '<pre style="padding:4px; font-size:14px;">'.$goals.'</pre>', 535, 80, false, '#efefef' );

    echo $con->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>