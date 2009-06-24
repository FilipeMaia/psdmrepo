<?php

require_once('RegDB.inc.php');

/*
 * This script will process a request for editing an instrument
 * in the database.
 */
if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "instrument identifier can't be empty" );
} else
    die( "no valid instrument identifier" );

if( isset( $_POST['description'] )) {
    $description = trim( $_POST['description'] );
    if( $description == '' )
        die( "instrument description field can't be empty" );
} else
    die( "no valid instrument description" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $instrument = $regdb->find_instrument_by_id ( $id )
        or die( "no such instrument" );

    $instrument->set_description( $description );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: RegDB_v1.php' );
        else if( $actionSuccess == 'list_instruments' )
            header( 'Location: RegDB_v1.php?action=list_instruments' );
        else if( $actionSuccess == 'view_instrument' )
            header( 'Location: RegDB_v1.php?action=view_instrument&id='.$instrument->id().'&name='.$instrument->name());
        else if( $actionSuccess == 'edit_instrument' )
            header( 'Location: RegDB_v1.php?action=edit_instrument&id='.$instrument->id().'&name='.$instrument->name());
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>