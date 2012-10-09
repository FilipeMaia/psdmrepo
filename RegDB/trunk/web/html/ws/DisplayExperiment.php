<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying parameters of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );


/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();
    $instrument_url =
        "<a href=\"javascript:view_instrument(".$instrument->id().",'".$instrument->name()."')\">".$instrument->name().'</a>';

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 800, 250 );
    echo $con
        ->label   ( 300,   0, 'Description')
        ->label   (   0,  25, 'Experiment: ' )
        ->value   ( 100,  25, $experiment->name().'&nbsp;&nbsp;[ ID='.$experiment->id().' ]' )
        ->textarea( 300,  25, $experiment->description(), 500, 125 )
        ->label   (   0,  50, 'Instrument: ' )
        ->value   ( 100,  50, $instrument_url )
        ->label   (   0, 100, 'Begin Time: ' )
        ->value   ( 100, 100, $experiment->begin_time()->toStringShort())
        ->label   (   0, 125, 'End Time: '   )
        ->value   ( 100, 125, $experiment->end_time()->toStringShort())
        ->label   ( 300, 175, 'Contact Info')
        ->label   (   0, 200, 'POSIX Group: ')
        ->value   ( 100, 200, $experiment->POSIX_gid())
        ->textarea( 300, 200, $experiment->contact_info(), 500, 50)
        ->label   (   0, 225, 'Leader: '     )
        ->value   ( 100, 225, $experiment->leader_account())
        ->html();

    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>