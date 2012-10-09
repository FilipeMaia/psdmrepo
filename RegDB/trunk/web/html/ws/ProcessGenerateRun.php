<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

/*
 * This script will process a request for generating the next run number
 * for an experiment. The run record will get permanently stored in
 * the database.
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
    RegDB::instance()->begin();
    $experiment = RegDB::instance()->find_experiment_by_id ( $id ) or die( 'no such experiment' );
    $run = $experiment->generate_run() or die( 'failed to generate the number' );

    if( isset( $actionSuccess )) {
        if     ($actionSuccess == 'home')             header('Location: ../index.php' );
        elseif ($actionSuccess == 'view_run_numbers') header('Location: ../index.php?action=view_run_numbers&id='.$experiment->id().'&name='.$experiment->name());
    }
    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>