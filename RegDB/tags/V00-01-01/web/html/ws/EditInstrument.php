<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for modifying parameters of an instrument.
 */
if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

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

    $con = new RegDBHtml( 0, 0, 700, 150 );
    echo $con
        ->label         ( 200,   0, 'Description')
        ->label         (   0,  25, 'Instrument: ' )
        ->value         ( 100,  25, $instrument->name())
        ->textarea_input( 200,  25, 'description', 500, 125, $instrument->description())
        ->html();

    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>