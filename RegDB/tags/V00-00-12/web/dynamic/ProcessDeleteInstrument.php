<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for deleting an instrument
 * from the database.
 */
if( !RegDBAuth::isAuthenticated()) return;

if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "instrument identifier can't be empty" );
} else
    die( "no valid instrument identifier" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $regdb->delete_instrument_by_id ( $id );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: index.php' );
        else if( $actionSuccess == 'list_instruments' )
            header( 'Location: index.php?action=list_instruments' );
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>