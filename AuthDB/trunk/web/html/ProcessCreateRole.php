<?php

require_once( 'AuthDB/AuthDB.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDBAuth;

/*
 * This script will process a request for creating a new role
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
} else
    die( "no valid application name" );

if( isset( $_POST['role_name'] )) {
    $role_name = trim( $_POST['role_name'] );
    if( $role_name == '' )
        die( "role name can't be empty" );
} else
    die( "no valid role name" );

if( isset( $_POST['privileges'] )) {
    $str = stripslashes( trim( $_POST['privileges'] ));
    if( $str == 'null' ) $privileges = null;
    else {
        $privileges = json_decode( $str );
        if( is_null( $privileges ))
            die( "failed to translate JSON object with a list of privileges" );
    }
} else
    die( "no valid role privileges collection" );

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

    $authdb->createRole( $application_name, $role_name, $privileges );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: index.php' );
        else if( $actionSuccess == 'list_roles' )
            header( 'Location: index.php?action=list_roles' );
        else
            ;
    }
    $authdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>