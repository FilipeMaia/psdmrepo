<?php

/*
 * Delete or un-delete an existing message
 * 
 * PARAMETERS:
 * 
 *   <id> [<operation>]
 */

require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id        = $SVC->required_int ('id') ;
    $operation = $SVC->required_enum('operation', array('delete', 'undelete')) ;

    $entry = $SVC->safe_assign ($SVC->logbook()->find_entry_by_id($id) ,
                                "no message found for id={$id}") ;

    $experiment = $entry->parent() ;

    $SVC->assert ($SVC->logbookauth()->canPostNewMessages($experiment->id()) ,
                  "not authorized to (un-)delete messages of experiment id={$experiment->id()}") ;

    switch ($operation) {

        case 'delete' :

            require_once 'lusitime/lusitime.inc.php' ;

            $deleted_time = LusiTime\LusiTime::now() ;
            $deleted_by   = $SVC->logbookauth()->authName() ;

            $experiment->delete_entry($id, $deleted_time, $deleted_by) ;

            return array (
                "deleted_time" => $deleted_time->toStringShort(),
                "deleted_by"   => $deleted_by) ;

        case 'undelete' :
            $experiment->undelete_entry($id) ;
            break ;
    }
}) ;
?>
