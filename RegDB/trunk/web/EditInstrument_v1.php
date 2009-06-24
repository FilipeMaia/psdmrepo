<?php

require_once('RegDB.inc.php');
require_once('RegDBHtml.php');

/*
 * This script will process a request for modifying parameters of an instrument.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "instrument identifier can't be empty" );
} else
    die( "no valid instrument identifier" );


/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $instrument = $regdb->find_instrument_by_id( $id )
        or die( "no such instrument" );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 700, 150 );
    echo $con
        ->label         ( 300,   0, 'Description')
        ->label         (   0,  25, 'Instrument: ' )
        ->value         ( 100,  25, $instrument->name())
        ->textarea_input( 300,  25, 'description', 500, 125, $instrument->description())
        ->html();

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>