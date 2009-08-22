<?php

require_once('LogBook/LogBook.inc.php');

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

    $shift->close( LusiTime::now());

    $experiment = $shift->parent();
    $instrument = $experiment->instrument();

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

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
