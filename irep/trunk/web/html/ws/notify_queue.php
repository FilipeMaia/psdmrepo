<?php

/**
 * This service will return notification lists.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $action = $SVC->required_str('action') ;
    $id     = $SVC->optional_int('id', null) ;

    if (is_null($id)) {
        foreach ($SVC->irep()->notify_queue() as $entry) {
            $id = $entry->id() ;
            switch ($action) {
                case 'submit' : $SVC->irep()->submit_notification_event($id) ; break ;
                case 'delete' : $SVC->irep()->delete_notification_event($id) ; break ;
                default :      $SVC->abort("operation is not implemented for action: {$action} and id: {$id}") ;
            }
        }
    } else {
        switch ($action) {
            case 'submit' : $SVC->irep()->submit_notification_event($id) ; break ;
            case 'delete' : $SVC->irep()->delete_notification_event($id) ; break ;
            default :       $SVC->abort("operation is not implemented for action: {$action} and id: {$id}") ;
        }
    }
    $SVC->finish(\Irep\IrepUtils::notifications2array($SVC->irep())) ;
}) ;
  
?>
