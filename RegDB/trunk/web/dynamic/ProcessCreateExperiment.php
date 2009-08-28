<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for creating a new experiment
 * in the database.
 */
if( !RegDBAuth::isAuthenticated()) return;

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['instrument_name'] )) {
    $instrument_name = trim( $_POST['instrument_name'] );
    if( $instrument_name == '' )
        die( "instrument name can't be empty" );
} else
    die( "no valid instrument name" );

if( isset( $_POST['begin_time'])) {
    $begin_time = LusiTime::parse( trim( $_POST['begin_time'] ));
    if( is_null( $begin_time ))
        die("begin time has invalid format");
} else
    die( "no begin time for experiment" );

if( isset( $_POST['end_time'])) {
    $end_time = LusiTime::parse( trim( $_POST['end_time'] ));
    if( is_null( $end_time ))
        die("end time has invalid format");
} else
    die( "no end time for experiment" );

if( isset( $_POST['group'] )) {
    $group = trim( $_POST['group'] );
    if( $group == '' )
        die( "group name can't be empty" );
} else
    die( "no valid group name" );

if( isset( $_POST['leader'] )) {
    $leader = trim( $_POST['leader'] );
    if( $leader == '' )
        die( "leader account name can't be empty" );
} else
    die( "no valid leader account" );

if( isset( $_POST['contact'] )) {
    $contact = trim( $_POST['contact'] );
    if( $contact == '' )
        die( "contact information field can't be empty" );
} else
    die( "no valid contact information" );

if( isset( $_POST['description'] )) {
    $description = trim( $_POST['description'] );
    if( $description == '' )
        die( "experiment description field can't be empty" );
} else
    die( "no valid experiment description" );

if( isset( $_POST['params'] )) {
    $str = stripslashes( trim( $_POST['params'] ));
    if( $str == 'null' ) $params = null;
    else {
        $params = json_decode( $str );
        if( is_null( $params ))
            die( "failed to translate JSON object with a list of parameters" );
    }
} else
    die( "no valid experiment parameters collection" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $experiment = $regdb->register_experiment (
        $experiment_name, $instrument_name, $description,
        LusiTime::now(), $begin_time, $end_time,
        $group, $leader, $contact );

    /* Add parameters if any
     */
    if( !is_null( $params ))
        foreach( $params as $p )
            $param = $experiment->add_param( $p[0], $p[1], $p[2] )
                or die( "failed to add experiment parameter: {$pa}");

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: index.php' );
        else if( $actionSuccess == 'list_experiments' )
            header( 'Location: index.php?action=list_experiments' );
        else if( $actionSuccess == 'view_experiment' )
            header( 'Location: index.php?action=view_experiment&id='.$experiment->id().'&name='.$experiment->name());
        else if( $actionSuccess == 'edit_experiment' )
            header( 'Location: index.php?action=edit_experiment&id='.$experiment->id().'&name='.$experiment->name());
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>