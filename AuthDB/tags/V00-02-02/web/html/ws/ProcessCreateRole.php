<?php

require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;

/*
 * This script will process a request for creating a new role
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
} else
    report_error( "no valid application name" );

if( isset( $_POST['role_name'] )) {
    $role_name = trim( $_POST['role_name'] );
    if( $role_name == '' )
        report_error( "role name can't be empty" );
} else
    report_error( "no valid role name" );

if( isset( $_POST['privileges'] )) {
    $str = stripslashes( trim( $_POST['privileges'] ));
    if( $str == 'null' ) $privileges = null;
    else {
        $privileges = json_decode( $str );
        if( is_null( $privileges ))
            report_error( "failed to translate JSON object with a list of privileges" );
    }
} else
    report_error( "no valid role privileges collection" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    AuthDB::instance()->begin();
    AuthDB::instance()->createRole( $application_name, $role_name, $privileges );
    AuthDB::instance()->commit();

    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home'      ) header('Location: ../index.php');
        elseif ($actionSuccess == 'list_roles') header('Location: ../index.php?action=list_roles');
    }

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>