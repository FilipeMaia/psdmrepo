<?php

/**
 * This service will save an e-mail notification configuration and return an
 * updated object.
 * 
 * Parameters:
 * 
 *   <recipient> <uid> <event_name> <enabled>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $recipient = $SVC->required_str ('recipient') ;
    $policy    = $SVC->optional_str ('policy', null) ;
    if (!is_null($policy)) {
        $SVC->irep()->update_notification_schedule($recipient, $policy) ;
    } else {
        $uid        = $SVC->required_str ('uid') ;
        $event_name = $SVC->required_str ('event_name') ;
        $enabled    = $SVC->required_bool('enabled') ;

        $event_type = $SVC->irep()->find_notify_event_type($recipient, $event_name) ;
        if (is_null($event_type)) $SVC->abort('unknown notification event') ;

        $notification = $SVC->irep()->find_notification($uid, $event_type->id()) ;
        if (is_null($notification))
            $notification = $SVC->irep()->add_notification($uid, $event_type->id(), $enabled) ;

        $SVC->irep()->update_notification($notification->id(), $enabled) ;
    }
    $SVC->finish (\Irep\IrepUtils::notifications2array($SVC->irep())) ;
}) ;


?>