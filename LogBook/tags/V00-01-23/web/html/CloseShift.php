<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request for closing an on-going shift.
 */
if( isset( $_POST['id'] )) {
    $shift_id = trim( $_POST['id'] );
    if( $shift_id == '' )
        die( "shift identifier can't be empty" );
} else
    die( "no valid shift identifier" );

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $shift = $logbook->find_shift_by_id( $shift_id )
        or die( "no such shift" );

    $experiment = $shift->parent();
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canManageShifts( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to manage shifts of the experiment',
            'index.php?action=select_experiment_and_shift'.
                '&instr_id='.$instrument->id().
                '&instr_name='.$instrument->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$shift->id()));
        exit;
    }

    // Proceed to the operation
    //
    $shift->close( LusiTime::now());

    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'select_experiment_and_shift' )
            header( 'Location: index.php?action=select_experiment_and_shift'.
                '&instr_id='.$instrument->id().
                '&instr_name='.$instrument->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$shift->id());
        else
            ;
    }
    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
