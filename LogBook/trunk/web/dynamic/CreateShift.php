<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for creating a new shift.
 */
if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else {
    die( "no valid experiment identifier" );
}
if( isset( $_POST['leader'] )) {
    $leader = trim( $_POST['leader'] );
    if( $leader == '' )
        die( "shift leader can't be empty" );
} else {
    die( "no valid shift leader" );
}
if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or die("failed to find the experiment" );

    $crew = array();
    $begin_time = LusiTime::now();
    $shift = $experiment->create_shift( $leader, $crew, $begin_time );

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
