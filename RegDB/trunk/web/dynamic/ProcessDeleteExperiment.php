<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will process a request for deleting an experiment
 * from the database.
 */
if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

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

    $regdb->delete_experiment_by_id ( $id );

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'home' )
            header( 'Location: index.php' );
        else if( $actionSuccess == 'list_experiments' )
            header( 'Location: index.php?action=list_experiments' );
        else
            ;
    }
    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>