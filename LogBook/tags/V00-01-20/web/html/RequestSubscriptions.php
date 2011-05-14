<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

/*
 * This script will process a request for displaying subscriptions made by specified
 * user account in a context of an experiment. If no user account parameters
 * is presented then all subscriptions will be displayed.
 */
if( isset( $_GET['exper_id'] )) {
    $exper_id = trim( $_GET['exper_id'] );
    if( $exper_id == '' )
        die( "experiment identifier can't be empty" );
} else {
    die( "no valid experiment identifier" );
}

$subscribed_by = null;
if( isset( $_GET['subscribed_by'] )) {
    $subscribed_by = trim( $_GET['subscribed_by'] );
    if( $subscribed_by == '' )
        die( "user account can't be empty" );
}

function subscription2json( $subscription ) {

    return json_encode(
        array (
            "delete"          => false,
            "subscriber"      => $subscription->subscriber(),
            "address"         => $subscription->address(),
            "subscribed_time" => $subscription->subscribed_time()->toStringShort(),
            "subscribed_host" => $subscription->subscribed_host()
        )
    );
}

/*
 * Return JSON objects with a list of subscriptions.
 */
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

    // Proceed to the operation
    //
    $subscriptions = $experiment->subscriptions( $subscribed_by );

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $subscriptions as $s ) {
      if( $first ) {
          $first = false;
          echo "\n".subscription2json( $s );
      } else {
          echo ",\n".subscription2json( $s );
      }
    }
    print <<< HERE
 ] } }
HERE;

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>
