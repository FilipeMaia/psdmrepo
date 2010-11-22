<?php

require_once( 'LogBook/LogBook.inc.php' );
require_once( 'LusiTime/LusiTime.inc.php' );

use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request for updating subscription entries
 * of the currently logged user.
 */
if( isset( $_POST['exper_id'] )) {
    $exper_id = trim( $_POST['exper_id'] );
    if( $exper_id == '' )
        die( "experiment identifier can't be empty" );
} else {
    die( "no valid experiment identifier" );
}

if( isset( $_POST['subscriptions'] )) {
    $str = stripslashes( trim( $_POST['subscriptions'] ));
    if( $str == 'null' ) $addresses = array();
    else {
        $addresses = json_decode( $str );
        if( is_null( $addresses ))
            die( "failed to translate JSON object with a list of subscribers' addresses" );
    }
} else {
    die( "no valid subscriptions to update" );
}

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $exper_id )
        or die( "no such experiment" );

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Find the current subscriptions and compare them with the new list.
    // Make appropriate changes to the database.
    //
    // NOTE: We assume that automatic notifications (of either "greetings"
    // or "goodby" type will be sent to recipients whose status will change.
    //
    $subscriptions = $experiment->subscriptions( LogBookAuth::instance()->authName());

    // Step 1: finding and subscribing all recipients which aren't in
    // the current list.
    //
    foreach( $addresses as $address ) {
        $found = false;
        foreach( $subscriptions as $s ) {
        	if( $s->address() == $address ) {
        		$found = true;
        		break;
        	}
        }
        if( !$found )
            $experiment->subscribe(
                LogBookAuth::instance()->authName(),
                $address,
                LusiTime::now(),
                $_SERVER['REMOTE_ADDR']
            );
    }

    // Step 2: finding and unsubscribing all recipients which aren't in
    // the new list.
    //
    foreach( $subscriptions as $s ) {
    	if( !in_array( $s->address(), $addresses ))
    	    $experiment->unsubscribe( $s->id());
    }

    // Return back to the caller
    //
    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'select_experiment' ) {
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name());
        } else {
            ;
        }
    }
    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
