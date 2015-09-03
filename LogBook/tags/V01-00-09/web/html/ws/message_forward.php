<?php

/*
 * Forward a  message to e-mail recipients
 *
 * PARAMETERS:
 *
 *  <id>          - message identifier
 *  <recipients>  - JSON array with e-mail addresses
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $id           = $SVC->required_int ('id') ;
    $recipients   = $SVC->required_json('recipients') ;

    $entry = $SVC->safe_assign ($SVC->logbook()->find_entry_by_id($id) ,
                                "no e-Log entry found for ID: {$id}") ;

    $SVC->assert ($SVC->logbookauth()->canRead($entry->experiment()->id()) ,
                  "not authorized to read e-Log of experiment:  {$entry->experiment()->name()}") ;

    $entry->experiment()->forward($entry, $recipients, $SVC->authdb()->authName()) ;
}) ;

?>
