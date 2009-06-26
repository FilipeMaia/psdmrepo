<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for modifying parameters of an experiment.
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
    $regdb = new RegDB();
    $regdb->begin();

    $experiment = $regdb->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();
    $group      = $experiment->POSIX_gid();

    $instrument = $instrument->name();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 700, 250 );
    echo $con
        ->label         ( 300,   0, 'Description')
        ->label         (   0,  25, 'Experiment: ' )
        ->value         ( 100,  25, $experiment->name())
        ->textarea_input( 300,  25, 'description', 500, 125, $experiment->description())
        ->label         (   0,  50, 'Instrument: ' )
        ->value         ( 100,  50, $instrument )
        ->label         (   0, 100, 'Begin Time: ' )
        ->value_input   ( 100, 100, 'begin_time', $experiment->begin_time()->toStringShort())
        ->label         (   0, 125, 'End Time: '   )
        ->value_input   ( 100, 125, 'end_time', $experiment->end_time()->toStringShort())
        ->label         ( 300, 175, 'Contact Info')
        ->label         (   0, 200, 'POSIX Group: ')
        ->value         ( 100, 200, $group )
        ->textarea_input( 300, 200, 'contact', 500, 50, $experiment->contact_info())
        ->label         (   0, 225, 'Leader: '     )
        ->value         ( 100, 225, $experiment->leader_account())
        ->html();

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>