<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use RegDB\RegDB;

/*
 * This script will process a request for creating a new role player
 * in the database.
 */
function report_error($msg) {
    print $msg;
    exit;
}

if( isset( $_POST['application_name'] )) {
    $application_name = trim( $_POST['application_name'] );
    if( $application_name == '' )
        report_error( "application name can't be empty" );
} else {
    report_error( "no valid application name" );
}

if( isset( $_POST['role_name'] )) {
    $role_name = trim( $_POST['role_name'] );
    if( $role_name == '' )
        report_error( "role name can't be empty" );
} else {
    report_error( "no valid role name" );
}

if( isset( $_POST['instrument_name'] )) {
    $instrument_name = trim( $_POST['instrument_name'] );
} else {
    report_error( "no valid instrument name" );
}
if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
} else {
    report_error( "no valid experiment name" );
}

if(( $instrument_name == '' && $experiment_name != '' ) ||
   ( $instrument_name != '' && $experiment_name == '' ))
    report_error( "inconsistent values of the instrument and experimenmt name parameters" );

if( isset( $_POST['user'] )) {
    $user = trim( $_POST['user'] );
} else {
    report_error( "no valid user name" );
}

if( isset( $_POST['group'] )) {
    $group = trim( $_POST['group'] );
} else {
    report_error( "no valid group name" );
}

if( $user != '' && $group != '' ) {
    report_error( "user name and group name are mutually exclusive parameters" );
} else if( $user == '' && $group == ''  ) {
    report_error( "user name or group name must be provided" );
}

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

//print_r( $privileges );
//echo "<br>";

/* Proceed with the operation
 */
try {
    AuthDB::instance()->begin();
    RegDB::instance()->begin();

    $exper_id = null;
    if( $instrument_name != '' && $experiment_name != '' ) {
        $experiment = RegDB::instance()->find_experiment( $instrument_name, $experiment_name )
            or report_error( "no such experiment: {$experiment_name}" );
        $exper_id = $experiment->id();
    }
    $role = AuthDB::instance()->find_role( $application_name, $role_name )
        or report_error( "no such role: {$role_name} for application: {$application_name}" );

    $player = $user;
    if( $player == '' ) $player = 'gid:'.$group;

    AuthDB::instance()->createRolePlayer( $application_name, $role_name, $exper_id, $player );
    AuthDB::instance()->commit();

    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home'             ) header('Location: ../index.php');
        elseif ($actionSuccess == 'list_role_players') header('Location: ../index.php?action=list_role_players');
    }

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>