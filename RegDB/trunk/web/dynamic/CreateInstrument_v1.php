<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will lay out a form for creating a new instrument.
 */

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 700, 150 );
    echo $con
        ->label         ( 300,   0, 'Description')
        ->label         (   0,  25, 'Instrument: ' )
        ->value_input   ( 100,  25, 'instrument_name' )
        ->textarea_input( 300,  25, 'description', 500, 125 )
        ->html();

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>