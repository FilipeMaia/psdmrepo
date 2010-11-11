<?php

require_once( 'AuthDB/AuthDB.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDBAuth;

/*
 * This script will process a request for deleting an existing application
 * from the database.
 */
// TODO: This needs to be changed with the real test
//
//if( !RegDBAuth::isAuthenticated()) return;

if( isset( $_POST['name'] )) {
    $name = trim( $_POST['name'] );
    if( $name == '' )
        die( "application name can't be empty" );
} else
    die( "no valid application name prtovided" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $authdb = new AuthDB();
    $authdb->begin();

    $authdb->deleteApplication( $name );

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