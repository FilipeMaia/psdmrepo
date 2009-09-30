<?php

require_once('AuthDB/AuthDB.inc.php');

/*
 * This script will process a request for creating a new role player
 * in the database.
 */
// TODO: This needs to be changed with the real test
//
//if( !RegDBAuth::isAuthenticated()) return;

//print_r( $_POST );
//echo "<br>";

if( isset( $_POST['application_name'] )) {
    $application_name = trim( $_POST['application_name'] );
    if( $application_name == '' )
        die( "application name can't be empty" );
} else {
    die( "no valid application name" );
}

if( isset( $_POST['role_name'] )) {
    $role_name = trim( $_POST['role_name'] );
    if( $role_name == '' )
        die( "role name can't be empty" );
} else {
    die( "no valid role name" );
}

if( isset( $_POST['instrument_name'] )) {
    $instrument_name = trim( $_POST['instrument_name'] );
} else {
    die( "no valid instrument name" );
}
if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
} else {
    die( "no valid experiment name" );
}

if(( $instrument_name == '' && $experiment_name != '' ) ||
   ( $instrument_name != '' && $experiment_name == '' ))
    die( "inconsistent values of the instrument and experimenmt name parameters" );

if( isset( $_POST['user'] )) {
    $user = trim( $_POST['user'] );
} else {
    die( "no valid user name" );
}

if( isset( $_POST['group'] )) {
    $group = trim( $_POST['group'] );
} else {
    die( "no valid group name" );
}

if( $user != '' && $group != '' ) {
    die( "user name and group name are mutually exclusive parameters" );
} else if( $user == '' && $group == ''  ) {
    die( "user name or group name must be provided" );
}

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

//print_r( $privileges );
//echo "<br>";

/* Proceed with the operation
 */
try {
    $authdb = new AuthDB();
    $authdb->begin();

    $regdb = new RegDB();
    $regdb->begin();

    $exper_id = null;
    if( $instrument_name != '' && $experiment_name != '' ) {
        $experiment = $regdb->find_experiment( $instrument_name, $experiment_name )
            or die( "no such experiment: {$experiment_name}" );
        $exper_id = $experiment->id();
    }
    $role = $authdb->find_role( $application_name, $role_name )
        or die( "no such role: {$role_name} for application: {$application_name}" );

    $player = $user;
    if( $player == '' ) $player = 'gid:'.$group;

    $authdb->createRolePlayer( $application_name, $role_name, $exper_id, $player );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: index.php' );
        else if( $actionSuccess == 'list_role_players' )
            header( 'Location: index.php?action=list_role_players' );
        else
            ;
    }
    $authdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>