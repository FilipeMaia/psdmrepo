<?php

/*
 * This script will perform the search for a single free-form entry (or entries)
 *
 * PARAMETERS:
 * 
 *   { id=<id> | exper_id=<exper_id> run_num=<run_num> } ['show_in_vicinity={0|1}']
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

use LogBook\LogBookUtils ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id = $SVC->optional_int('id', null) ;

    $exper_id = null ;
    $run_num  = null ;
    
    if (!$id) {
        $exper_id = $SVC->required_int('exper_id') ;
        $run_num =  $SVC->required_int('run_num') ;
    }

    $show_in_vicinity = $SVC->optional_bool('show_in_vicinity', false) ;

    $results = null ;

    if ($id) {

        $entry = $SVC->safe_assign ($SVC->logbook()->find_entry_by_id($id) ,
                                    'no such message entry') ;

        $SVC->assert ($SVC->logbookauth()->canRead($entry->parent()->id()) ,
                      'not authorized to read messages for the experiment') ;

        $results = $show_in_vicinity ?
            LogBookUtils::search_around_message ($entry->id(), function ($msg) { $SVC->abort($msg) ; }) :
            array (LogBookUtils::entry2array($entry)) ;
 
    } elseif ($exper_id) {

        $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                         "no experiment found for id={$exper_id}") ;

        $run = $SVC->safe_assign ($experiment->find_run_by_num($run_num) ,
                                  "no run {$run_num} found within experiment of id={$experiment->id()}") ;

        $SVC->assert ($SVC->logbookauth()->canRead($experiment->id()) ,
                      'not authorized to read messages for the experiment') ;

        $results = $show_in_vicinity ?
            LogBookUtils::search_around_run ($run->id(), function ($msg) { $SVC->abort($msg) ; }) :
            array (LogBookUtils::run2array($run, 'run')) ;

    } else {
        $SVC->abort('internal error: unsupported mode') ;
    }
    return array (
        'ResultSet' => array (
            'Result' => $results
        )
    ) ;
}) ;
?>