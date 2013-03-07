<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying parameters of an instrument.
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
    RegDB::instance()->begin();

    $instrument = RegDB::instance()->find_instrument_by_id( $id )
        or die( "no such instrument" );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 600, 150 );
    echo $con
        ->label   ( 200,   0, 'Description')
        ->label   (   0,  25, 'Instrument: ' )
        ->value   ( 100,  25, $instrument->name())
        ->textarea( 200,  25, $instrument->description(), 500, 125 )
        ->html();

    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>