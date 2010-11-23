<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

/*
 * This script will process a request for deleting the specified free-form entry.
 */
if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "free-form entry identifier can't be empty" );
} else {
    die( "no valid free-form entry identifier" );
}
if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
    if( isset( $_POST['shift_id'] )) {
       $shift_id = trim( $_POST['shift_id'] );
    }
    if( isset( $_POST['run_id'] )) {
       $run_id = trim( $_POST['run_id'] );
    }
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $entry = $logbook->find_entry_by_id( $id )
        or die( "no such free-form entry" );

    $experiment = $entry->parent();
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canDeleteMessages( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to delete messages of the experiment',
            'index.php?action=select_experiment'.
                '&instr_id='.$instrument->id().
                '&instr_name='.$instrument->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name()));
        exit;
    }

    // Proceed to the operation
    //
    $logbook->delete_entry_by_id( $id );

    // Return back to the caller
    //
    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'select_experiment' ) {
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name());
        } else if( $actionSuccess == 'select_experiment_and_shift' ) {
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$shift_id);
        } else if( $actionSuccess == 'select_experiment_and_run' ) {
            $run = $experiment->find_run_by_id( $run_id )
                or die( "no such run" );
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$shift_id,
                '&run_id='.$run_id);
        } else {
            ;
        }
    }
    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>
