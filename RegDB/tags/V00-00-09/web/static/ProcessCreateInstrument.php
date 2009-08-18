<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for creating a new instrument
 * in the database.
 */
if( isset( $_POST['instrument_name'] )) {
    $instrument_name = trim( $_POST['instrument_name'] );
    if( $instrument_name == '' )
        die( "instrument name can't be empty" );
} else
    die( "no valid instrument name" );

if( isset( $_POST['description'] )) {
    $description = trim( $_POST['description'] );
    if( $description == '' )
        die( "instrument description field can't be empty" );
} else
    die( "no valid instrument description" );

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $instrument = $regdb->register_instrument (
        $instrument_name, $description );

    header( 'Location: DisplayInstrument.php?id='.$instrument->id());

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>