<?php

require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;

/*
 * This script will process a request for deleting an existing role
 * from the database.
 */
function report_error($msg) {
    print $msg;
    exit;
}
if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        report_error( "role identifier can't be empty" );
} else
    report_error( "no valid role identifier prtovided" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    AuthDB::instance()->begin();
    AuthDB::instance()->deleteRole( $id );
    AuthDB::instance()->commit();

    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home'      ) header('Location: ../index.php');
        elseif ($actionSuccess == 'list_roles') header('Location: ../index.php?action=list_roles');
    }

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>