<?php

require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;

/*
 * This script will process a request for deleting an existing application
 * from the database.
 */
function report_error($msg) {
    print $msg;
    exit;
}
if( isset( $_POST['name'] )) {
    $name = trim( $_POST['name'] );
    if( $name == '' )
        report_error( "application name can't be empty" );
} else
    report_error( "no valid application name prtovided" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    AuthDB::instance()->begin();
    AuthDB::instance()->deleteApplication( $name );
    AuthDB::instance()->commit();

    if( isset( $actionSuccess )) {
        if    ( $actionSuccess == 'home' )       header( 'Location: ../index.php' );
        elseif( $actionSuccess == 'list_roles' ) header( 'Location: ../index.php?action=list_roles' );
    }

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>