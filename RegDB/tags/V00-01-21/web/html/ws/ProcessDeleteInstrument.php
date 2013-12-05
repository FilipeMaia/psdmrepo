<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for deleting an instrument
 * from the database.
 */
if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

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
    RegDB::instance()->begin();
    RegDB::instance()->delete_instrument_by_id ( $id );

    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home')             header('Location: ../index.php');
        elseif ($actionSuccess == 'list_instruments') header('Location: ../index.php?action=list_instruments');
    }
    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>