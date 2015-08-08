<?php

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

require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id  =            $SVC->required_int('exper_id') ;
    $operation = strtoupper($SVC->required_str('operation')) ;
    $id        =            $SVC->optional_int('id', null) ;

    if (!in_array($operation, array('CHECK', 'SUBSCRIBE', 'UNSUBSCRIBE') ))
        $SVC->abort("invalid operation code: '{$operation}'") ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort('no such experiment exists') ;

    if (!$SVC->logbookauth()->canRead($experiment->id()))
        $SVC->abort('Not authorized for this operation') ;

    $subscription = null ;
    $uid          = $SVC->logbookauth()->authName() ;
    $address      = $uid.'@slac.stanford.edu' ;

    if ($id) {

        // Assume the specified identifier (may not already exist if it was unsubscribed
        // earlier).
        //
        $subscription = $experiment->find_subscriber_by_id($id) ;
        if ($subscription) {
            $uid     = $subscription->subscriber() ;
            $address = $subscription->address() ;
        }
    } else {

        // Search for the current logged user's subscriptions
        //
        foreach ($experiment->subscriptions($uid) as $s) {
            if (($s->subscriber() == $uid) && ($s->address() == $address)) {
                $subscription = $s ;
                break ;
            }
        }
    }
    switch ($operation) {
    case 'CHECK' :
        break;
    case 'SUBSCRIBE' :
        if( !$subscription)
            $subscription = $experiment->subscribe($uid, $address, LusiTime::now(), $_SERVER['REMOTE_ADDR']) ;
        break ;
    case 'UNSUBSCRIBE' :
        if ($subscription) {
            $experiment->unsubscribe($subscription->id()) ;
            $subscription = null ;
        }
        break ;
    }

    $all_subscriptions = array() ;
    foreach ($experiment->subscriptions() as $s)
        array_push (
            $all_subscriptions ,
            array (
                'id'              => $s->id() ,
                'subscriber'      => $s->subscriber() ,
                'address'         => $s->address() ,
                'subscribed_time' => $s->subscribed_time()->toStringShort() ,
                'subscribed_host' => $s->subscribed_host())) ;

    $SVC->finish (
        array (
            "Subscribed"       => $subscription ? 1 : 0 ,
            "AllSubscriptions" => $all_subscriptions
        )
    ) ;
}) ;

?>
