<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDBAuth;

/*
 * This script will process a request for deleting an existing role
 * from the database.
 */
// TODO: This needs to be changed with the real test
//
//if( !RegDBAuth::isAuthenticated()) return;

if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "role identifier can't be empty" );
} else
    die( "no valid role identifier prtovided" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $authdb = new AuthDB();
    $authdb->begin();

    $authdb->deleteRole( $id );

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