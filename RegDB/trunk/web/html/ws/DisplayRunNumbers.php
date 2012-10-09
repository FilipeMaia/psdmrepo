<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying run numbers of an experiment.
 */
if( isset( $_GET['exper_id'] )) {
    $exper_id = trim( $_GET['exper_id'] );
    if( $exper_id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );


/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id( $exper_id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();
    $last_run   = $experiment->last_run();

    $experiment_url =
        "<a href=\"javascript:view_experiment(".$experiment->id().",'".$experiment->name()."')\">".$experiment->name().'</a>';
    $instrument_url =
        "<a href=\"javascript:view_instrument(".$instrument->id().",'".$instrument->name()."')\">".$instrument->name().'</a>';

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 425, 125 );
    echo $con
        ->label   (   0,   0, 'Experiment: ' )
        ->value   ( 125,   0, $experiment_url )
        ->label   (   0,  25, 'Instrument: ' )
        ->value   ( 125,  25, $instrument_url )
        ->label   (   0,  75, 'Total # of runs: ' )
        ->value   ( 125,  75, count( $experiment->runs()))
        ->label   (   0, 100, 'Last run #: ' )
        ->value   ( 125, 100, is_null( $last_run ) ? 'n/a' : $last_run->num())
        ->label   ( 175, 100, 'Requested: '   )
        ->value   ( 275, 100, is_null( $last_run ) ? 'n/a' : $last_run->request_time()->toStringShort())
        ->html();

    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>