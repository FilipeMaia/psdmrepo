<?php

require_once( 'regdb/regdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;

use FileMgr\FileMgrIrodsWs;

/*
 * This script will generate a module with input elements for the filter form
 * in a context of the specified experiment.
 */
header( 'Content-type: text/html' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print $msg;
    exit;
}

if( isset( $_GET['exper_id'] )) {
    $exper_id = trim( $_GET['exper_id'] );
    if( $exper_id == '' ) {
        $exper_id( "experiment identifier can't be empty" );
    }
} else {
    die( "no valid experiment identifier" );
}

/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    $experiment =  RegDB::instance()->find_experiment_by_id( $exper_id );
    if (is_null($experiment)) report_error( "no such experiment" );

    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !RegDBAuth::instance()->canRead())
        report_error(
            RegDBAuth::reporErrorHtml(
                'You are not authorized to access any information from the Experiment Registration Database',
                'index.php'));

    $range = FileMgrIrodsWs::max_run_range( $instrument->name(), $experiment->name(), array( 'xtc', 'hdf5' ));
    $range_of_runs = $range['min'].'-'.$range['max'];
    $range_help = '1,3,5,10-20,200';

    $con = new RegDBHtml( 0, 0, 350, 140 );
    echo $con
        ->label         (   0,   0, 'R u n s' )
        ->radio_input   (   0,  25, 'runs', 'all',   true  )->label      ( 25, 20, 'all', false )
        ->radio_input   (   0,  45, 'runs', 'range', false )->value_input( 25, 40, 'runs_range', $range_of_runs, $range_help, 8 )

        ->label         ( 130,   0, 'A r c h i v e d' )
        ->radio_input   ( 130,  25, 'archived', 'yes_or_no', true  )->label( 150,  20, 'yes or no', false )
        ->radio_input   ( 130,  45, 'archived', 'yes',       false )->label( 150,  45, 'yes',       false )
        ->radio_input   ( 130,  65, 'archived', 'no',        false )->label( 150,  65, 'no',        false )

        ->label         ( 250,   0, 'D i s k' )
        ->radio_input   ( 250,  25, 'local', 'yes_or_no', true  )->label( 270,  20, 'yes or no', false )
        ->radio_input   ( 250,  45, 'local', 'yes',       false )->label( 270,  40, 'yes',       false )
        ->radio_input   ( 250,  65, 'local', 'no',        false )->label( 270,  60, 'no',        false )

        ->label         (   0,  90, 'T y p e s' )
        ->checkbox_input(   0, 115, 'xtc',   'xtc',   true )->label(  20, 110, 'XTC',   false )
        ->checkbox_input(   0, 135, 'hdf5',  'hdf5',  true )->label(  20, 130, 'HDF5',  false )

        ->button        ( 130, 120, 'reset_filter_button',  'Reset', 'reset filter to its initial state' )
        ->button        ( 205, 120, 'submit_filter_button', 'Apply', 'update the file list using this filter' )
        ->button        ( 280, 120, 'import_list_button',   'Import', 'get an import list for selected files for which a disk copy is available' )
        
        ->html();

     RegDB::instance()->commit();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>