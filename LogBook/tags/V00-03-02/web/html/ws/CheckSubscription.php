<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;

use LusiTime\LusiTime;

/* This script will check or manage e-mail notification subscription of the
 * current in a context of the specified experimnent. If successful the script
 * will return a JSON object with a flag indicating the current state of the subscription.
 * Otherwise another JSON object with an explanation of the problem will be returned.
 * 
 * Parameters of the script (all mandatory):
 *
 * exper_id  - a numeric identifieer of the experiment
 * operation - requested operation (see details below)
 * id        - an optional identifier of a subscription entry to be unsubscribed
 *
 * NOTES:
 * - if no subscriber id is presented then the current logged user will be
 *   assumed based on their account's UID.
 * - the subscriber id is allowed not to exist in case if the subscriber
 *   has already been unsubscribed. This just to deal with synchronization
 *   issues.
 */
header( 'Content-type: application/json' );
header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

/**
 * This function is used to report errors back to the script caller applications
 *
 * @param $msg - a message to be reported
 */

function report_error( $msg ) {
    echo '{"Status":"error","Message":'.json_encode( $msg ).'}';
    exit;
}

/* Translate & analyze input parameters
 */
if( !isset( $_GET['exper_id'] )) report_error( 'no experiment identifier parameter found' );
$exper_id = (int)trim( $_GET[ 'exper_id' ] );
if( $exper_id <= 0 ) report_error( 'invalid experiment identifier' );

$operations = array(
	'CHECK' => 0,
	'SUBSCRIBE' => 1,
	'UNSUBSCRIBE' => 2
);
if( !isset( $_GET['operation'] )) report_error( 'no operation code found' );
$operation = strtoupper( trim( $_GET[ 'operation' ] ));
if( !array_key_exists( $operation, $operations )) report_error( "invalid operation code: '{$operation}'" );

if( isset($_GET['id'])) {
	$id = trim($_GET['id']);
	if( $id == '' ) report_error( 'subscribed identifier can not be empty' );
}

/**
 * Return a JSON document with the current state of the subscription.
 *
 * @param $is_subscribed - True if subscribed
 */	
function return_result( $is_subscribed, $all_subscriptions ) {
    echo
        '{"Status":"success","Subscribed":'.json_encode( $is_subscribed ? 1 : 0 ).
        ', "AllSubscriptions": '.json_encode($all_subscriptions).'}';
}

try {
    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $exper_id );
    if( is_null( $experiment )) report_error( 'no such experiment exists' );

    if( !LogBookAuth::instance()->canRead( $experiment->id())) report_error( 'You are not authorized for this operation' );

    $subscription = null;
    $uid          = LogBookAuth::instance()->authName();
    $address      = $uid.'@slac.stanford.edu';

    if( isset( $id )) {

    	// Assume the specified identifier (may not already exist if it was unsubscribed
    	// earlier).
    	//
    	$subscription = $experiment->find_subscriber_by_id( $id );
    	if( !is_null($subscription)) {
	    	$uid     = $subscription->subscriber();
    		$address = $subscription->address();
    	}
    } else {

    	// Search for the current logged user's subscriptions
    	//
	   	foreach( $experiment->subscriptions( $uid ) as $s ) {
   			if(( $s->subscriber() == $uid ) && ( $s->address() == $address )) {
   				$subscription = $s;
   				break;
   			}
   		}
    }
	switch( $operation ) {
	case 'CHECK':
		break;
	case 'SUBSCRIBE':
		if( is_null( $subscription ))
			$subscription = $experiment->subscribe( $uid, $address, LusiTime::now(), $_SERVER['REMOTE_ADDR'] );
		break;
	case 'UNSUBSCRIBE':
		if( !is_null( $subscription )) {
			$experiment->unsubscribe( $subscription->id());
			$subscription = null;
		}
		break;
	}

	$all_subscriptions = array();
	foreach( $experiment->subscriptions() as $s ) {
		array_push(
			$all_subscriptions,
			array(
				'id' => $s->id(),
				'subscriber' => $s->subscriber(),
				'address' => $s->address(),
				'subscribed_time' => $s->subscribed_time()->toStringShort(),
				'subscribed_host' => $s->subscribed_host()
			)
		);
	}
	LogBook::instance()->commit();

	return_result( !is_null( $subscription ), $all_subscriptions );

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
