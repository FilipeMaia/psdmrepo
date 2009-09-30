<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for creating a new shift
 * in the database.
 */
if( !LogBookAuth::isAuthenticated()) return;

if( isset( $_POST['leader'] )) {
    $leader = trim( $_POST['leader'] );
    if( $leader == '' )
        die( "shift leader's name can't be empty" );
} else
    die( "no valid shift leader account" );

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = $_POST['experiment_name'];
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['begin_time'] )) {
    $begin_time = LusiTime::parse( trim( $_POST['begin_time'] ));
    if( is_null( $begin_time ))
        die( "begin time has invalid format" );
} else
    die( "no begin time for shift" );

if( isset( $_POST['end_time'] )) {
    $str = trim( $_POST['end_time'] );
    if( $str == '' ) $end_time=null;
    else {
        $end_time = LusiTime::parse( $str );
        if( is_null( $end_time ))
            die( "end time has invalid format" );
    }
} else
    die( "no end time for shift" );

// Read optional names of crew members submitted with the request
//
define( MAX_MEMBERS, 3 );   // max number of members to attach

$crew = array();
for( $i = 0; $i < MAX_MEMBERS; $i++ ) {

    $key  = 'crew_member'.$i;
    if( isset( $_POST[$key] )) {

        $member = trim( $_POST[$key] );
        if( $member != '' )
            array_push( $crew, $member );
    }
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("failed to find the experiment" );

    $shift = $experiment->create_shift( $leader, $crew, $begin_time, $end_time );
    /* Redirect to another page to see all experiments
     */
    header( 'Location: ListShifts.php' ) ;

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}