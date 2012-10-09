<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request for creating a new shift.
 */
if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else {
    die( "no valid experiment identifier" );
}
if( isset( $_POST['leader'] )) {
    $leader = trim( $_POST['leader'] );
    if( $leader == '' )
        die( "shift leader can't be empty" );
} else {
    die( "no valid shift leader" );
}
if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}
if( isset( $_POST['author'] )) {
    $author = trim( $_POST['author'] );
    if( $author == '' )
        die( "shift request author can't be empty" );
} else {
    die( "no valid shift request author" );
}
if( isset( $_POST['goals'] )) {
    $goals = trim( $_POST['goals'] );
} else {
    die( "no valid parameter for shift goals" );
}

$crew = array();
if( isset( $_POST['max_crew_size'] )) {
    $max_crew_size = (int)trim( $_POST['max_crew_size'] );
    for( $i=0; $i<$max_crew_size; $i++) {
        $key = 'member'.$i;
        if( isset( $_POST[$key] )) {
            $member = trim( $_POST[$key] );
            if( $member != '' ) array_push( $crew, $member );
        }
    }
}

try {

    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $id ) or die("failed to find the experiment" );
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canManageShifts( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to manage shifts of the experiment',
            '../index.php?action=select_experiment'.
                '&instr_id='.$instrument->id().
                '&instr_name='.$instrument->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name()));
        exit;
    }

    // Proceed to the operation
    //
    $begin_time = LusiTime::now();
    $shift = $experiment->create_shift( $leader, $crew, $begin_time );

    $entry = $experiment->create_entry (
        $author, 'TEXT', $goals, $shift_id=$shift->id());
    $entry->add_tag( 'SHIFT_GOALS', '' );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'select_experiment_and_shift' )
            header( 'Location: ../index.php?action=select_experiment_and_shift'.
                '&instr_id='.$instrument->id().
                '&instr_name='.$instrument->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$shift->id());
        else
            header( 'Location: ../'.$actionSuccess );
    }
    LogBook::instance()->commit();

} catch( LogBookException  $e ) { print $e->toHtml(); }
  catch( LusiTimeException $e ) { print $e->toHtml(); }

?>
