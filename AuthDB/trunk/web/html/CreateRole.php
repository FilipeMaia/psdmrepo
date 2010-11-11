<?php

require_once( 'AuthDB/AuthDB.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDBHtml;

/*
 * This script will lay out a form for creating a new role.
 */

// TODO: This needs to be changed with the real test
//
//if( !RegDBAuth::isAuthenticated()) return;

/* Proceed with the operation
 */
try {
    $authdb = new AuthDB();
    $authdb->begin();

    $applications = $authdb->applications();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 700, 125 );
    echo $con
        ->label         (   0,  25, 'Application:' )
        ->value_input   ( 100,  25, 'application_name' )
        ->label         ( 275,  25, 'or' )
        ->select_input  ( 305,  22, 'application_name_select', $applications )
        ->label         (   0,  75, 'Role:' )
        ->value_input   ( 100,  75, 'role_name' )
        ->html();

    $authdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>