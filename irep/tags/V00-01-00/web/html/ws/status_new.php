<?php

/**
 * This service will create a new status or a substatus and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   { <status> | <status> <status2> ]
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $status_name  = $SVC->required_str('status') ;
    $status2_name = $SVC->optional_str('status2', null) ;

    if (is_null($status2_name)) {
        $status = $SVC->irep()->find_status_by_name($status_name) ;
        if (!is_null($status)) $SVC->abort("the status {$status->name()} already exists") ;
        $status = $SVC->irep()->add_status($status_name) ;
        $status->add_status2('') ;
    } else {
        $status = $SVC->irep()->find_status_by_name($status_name) ;
        if (is_null($status)) $SVC->abort("the status not found: {$status_name}") ;
        $status2 = $status->find_status2_by_name($status2_name) ;
        if (!is_null($status2)) $SVC->abort("the sub-status {$status->name()}::{$status2_name} already exists") ;
        $status->add_status2($status2_name) ;

    }

    $SVC->finish(\Irep\IrepUtils::statuses2array($SVC->irep())) ;
}) ;

?>
