<?php

/*
 * This script will process a request for changing a scope of teh specified
 * free-form entry.
 *
 * A JSON object with the result of the operation will be returned.
 * If the operation was successfull then the reply will also contain
 * an additional information about a new scope of the message.
 */

require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id      = $SVC->required_int('id') ;
    $scope   = $SVC->required_str('scope') ;
    $run_num = $SVC->optional_int('run_num', null) ;

    $entry = $SVC->logbook()->find_entry_by_id($id) ;
    if (!$entry) $SVC->abort("no message entry for id={$id}") ;

    if (!$SVC->logbookauth()->canEditMessages($entry->parent()->id()))
        $SVC->abort('not authorized to modify messages for the experiment') ;

    switch ($scope) {
        case 'run' :
            if ($entry->parent_entry_id()) $SVC->abort('operation is not permitted on child messages') ;
            if ($entry->shift_id())        $SVC->abort('operation is not permitted on messages associated with shifts') ;
            if (!$run_num) {
                $entry = $SVC->logbook()->dettach_entry_from_run($entry) ;
            } else {
                $run = $entry->parent()->find_run_by_num($run_num) ;
                if (!$run) $SVC->abort('no such run found in the experiemnt') ;
                $entry = $SVC->logbook()->attach_entry_to_run($entry, $run) ;
            }
            $SVC->finish(array (
                'run_id'  => $entry->run_id() ? $entry->run_id()     : 0 ,
                'run_num' => $entry->run_id() ? $entry->run()->num() : 0
            )) ;
    }
    $SVC->finish() ;
}) ;

?>
