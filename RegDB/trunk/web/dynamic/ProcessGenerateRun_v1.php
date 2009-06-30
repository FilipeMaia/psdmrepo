<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for generating the next run number
 * for an experiment. The run record will get permanently stored in
 * the database.
 */
if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();

    $experiment = $regdb->find_experiment_by_id ( $id )
        or die( 'no such experiment' );

    $run = $experiment->generate_run()
        or die( 'failed to generate the number' );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: RegDB.php' );
        else if( $actionSuccess == 'view_run_numbers' )
            header( 'Location: RegDB.php?action=view_run_numbers&id='.$experiment->id().'&name='.$experiment->name());
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>