<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;
use RegDB\RegDBException;
/*
 * This script will lay out a form for creating a new instrument.
 */
if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

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

    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>