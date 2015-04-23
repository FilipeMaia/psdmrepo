<?php

/*
 * Update current user's subscription for the file migration delays
 * 
 * For complete documentation see JIRA ticket:
 * https://jira.slac.stanford.edu/browse/PSDH-35
 *
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {
    
    $uid    = $SVC->required_str  ('uid') ;
    $events = $SVC->required_json ('events') ;

    /* -------------------------------------------------------------
     *   If the user is not interested in a single event then kick
     *   him/her out from subscribers.
     * ------------------------------------------------------------
     */

    $unsubscribe = true ;
    foreach ($events as $e) {
        if ($e->subscribed) {
            $unsubscribe = false ;
            break ;
        }
    }
    if ($unsubscribe) {

        // Don't bother if the user is no longer subscribed.

        if ($SVC->sysmon()->find_fm_delay_subscriber($uid)) {
            $SVC->sysmon()->remove_fm_delay_subscriber ($uid) ;
        }
    } else {

        // Allow a choice from any existing instrument as well as
        // an empty string for all instruments.

        $instr = $SVC->required_enum ('instr' ,
                                      array_merge (
                                          $SVC->regdb()->instrument_names() ,
                                          array('')) ,
                                      array('ignore_case' => true, 'convert' => 'toupper')) ;

        $last_sec  = $SVC->required_int ('last_sec') ;
        $delay_sec = $SVC->required_int ('delay_sec') ;

        // Depending on what already exists add a brand new subscription
        // or update an existing one.
        
        if ($SVC->sysmon()->find_fm_delay_subscriber($uid)) {
            $SVC->sysmon()->update_fm_delay_subscriber (
                $uid ,
                $SVC->authdb()->authName() ,
                $instr ,
                $last_sec ,
                $delay_sec ,
                $events
            ) ;
        } else {
            $SVC->sysmon()->add_fm_delay_subscriber (
                $uid ,
                $SVC->authdb()->authName() ,
                $instr ,
                $last_sec ,
                $delay_sec ,
                $events
            ) ;
        }
    }
    
    /* ------------------------------------------------------------------
     *   Send e-mail notification on the transitionin the subscriptions
     *   status to a person affected by the operation.
     * ------------------------------------------------------------------
     */
    $url = 'https://pswww.slac.stanford.edu/' ;
    $now = LusiTime::now() ;

    $transition = $unsubscribe ?
"The message was sent by the automated notification system because  this e-mail\n" .
"has been just unregistered from recieving alerts on the file migration delays.\n"
        :
"The message was sent by the automated notification system because  this e-mail\n" .
"has been just registered from recieving alerts on the file migration delays.\n" ;

    $msg        = <<<HERE

                             ** ATTENTION **

{$transition}

The change has been requested by:

  '{$SVC->authdb()->authName()}' @ {$SVC->authdb()->authRemoteAddr()} [ {$now->toStringShort()} ]

You can  manage  your  criteria  or unsubscribe from recieving notifications
by  using "Data Migration Monitor/Notifier"  found  in the "Data Management"
section of: {$url}

HERE;
    $SVC->configdb()->do_notify (
        "{$uid}@slac.stanford.edu" ,
        $unsubscribe ? '*** UNSUBSCRIBED ***' : '*** SUBSCRIBED ***' ,
        $msg ,
        'LCLS Data Migration Monitor') ;

    /* -------------------------------------------------
     *   Return the current state of all subscriptions
     * -------------------------------------------------
     */
    if (PHP_VERSION_ID < 50400) {
        /*
         * JSON-ready object serialization control is provided
         * through a special interface JsonSerializable as
         * of PHP 5.4. The method jsonSerialize will return \stdClass
         * object with members ready for the JSON serialization.
         * Until that we have to call this method explicitly.
         */
        $users = array() ;
        foreach ($SVC->sysmon()->fm_delay_subscribers() as $s)
            array_push($users, $s->jsonSerialize()) ;

        return array ('users' => $users) ;
    }
    return array (
        'users' => $SVC->sysmon()->fm_delay_subscribers()) ;}) ;

?>
