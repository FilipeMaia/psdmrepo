<?php

/*
 * This script will perform the search for messages in a scope of the specified run.
 * The output will include:
 * 
 * - messages exlicitly associated with that run
 * - messages not-associated with any run but posted since the begin
 *   time of the run through the begin time of the next one (or through the end
 *   if no next run exists).
 * 
 * The output will exclude messages associated with other runs.
 *
 * PARAMETERS:
 * 
 *   <exper_id> <run> [ <inject_deleted_messages> ]
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
use LogBook\LogBookUtils ;


/**
 * The functon will produce a sorted list of timestamps based on keys of
 * the input dictionary. If two consequitive begin/end run records are found
 * for the same run then the records will be collapsed into a single record 'run'
 * with the begin run timestamp.
 * 
 * NOTE: The contents of input array will be modified for collapsed runs
 *       by replacing types for 'begin_run' / 'end_run' with just 'run'.
 */
function sort_by_time (&$entries_by_timestamps) {

    $all_timestamps = array_keys($entries_by_timestamps) ;
    sort($all_timestamps) ;

    // First check if we need to collapse here anything.

    $timestamps     = array() ;
    $prev_begin_run = null ;

    foreach ($all_timestamps as $t) {
        foreach ($entries_by_timestamps[$t] as $pair) {
            $entry = $pair['object'] ;
            switch ($pair['type']) {
                case 'entry' :
                    $prev_begin_run = null ;
                    array_push($timestamps, $t) ;
                    break ;
                case 'begin_run' :
                    $prev_begin_run = $t ;
                    array_push($timestamps, $t) ;
                    break ;
                case 'end_run' :
                    if (is_null( $prev_begin_run)) {
                        array_push($timestamps, $t) ;
                    } else {
                        foreach (array_keys($entries_by_timestamps[$prev_begin_run]) as $k) {
                            if ($entries_by_timestamps[$prev_begin_run][$k]['type'] == 'begin_run') {
                                $entries_by_timestamps[$prev_begin_run][$k]['type'] = 'run' ;
                                $prev_begin_run = null ;
                                break ;
                            }
                        }
                    }
                    break ;
                }
        }
    }

    // Remove duplicates (if any). They may show up if an element of
    // $entries_by_timestamps will have more than one entry.

    $timestamps = array_unique($timestamps) ;

    return $timestamps ;
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;
    $run_num  = $SVC->required_int('run') ;

    $inject_deleted_messages = $SVC->optional_flag('inject_deleted_messages') ;

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     'no such experiment') ;

    $SVC->assert ($SVC->logbookauth()->canRead($experiment->id()) ,
                  "not authorized to read from experiment_id={$experiment->id()}") ;

    $run = $SVC->safe_assign ($experiment->find_run_by_num($run_num) ,
                              "run {$run_num} can't be found") ;

    $next_run = $experiment->find_run_by_num($run_num + 1) ;

    $entries_by_timestamps = array() ;

    // Inject entries for the run itself

    {
        $t = $run->begin_time()->to64() ;
        if (!array_key_exists($t, $entries_by_timestamps)) $entries_by_timestamps[$t] = array () ;
        array_push (
            $entries_by_timestamps[$t] ,
            array (
                'type'   => 'begin_run' ,
                'object' => $run)) ;

        if ($run->end_time()) {
            $t = $run->end_time()->to64() ;
            if (!array_key_exists($t, $entries_by_timestamps)) $entries_by_timestamps[$t] = array () ;
            array_push (
                $entries_by_timestamps[$t] ,
                array (
                    'type'   => 'end_run' ,
                    'object' => $run)) ;
        }
    }

    // Messages posted in the validity range associated with the run
    // except those associated with other runs.

    foreach ($experiment->search (
            null ,  // $shift_id
            null ,  // $run_id
            '' ,    // $text2search
            true ,  // $search_in_messages,
            false , // $search_in_tags,
            false , // $search_in_values,
            true ,  // $posted_at_experiment,
            false , // $posted_at_shifts,
            false , // $posted_at_runs,
            $run->begin_time() ,
            $next_run ? $next_run->begin_time() : null ,
            null ,  // $tag,
            null ,  // $author,
            null ,  // $since
            null ,  // $limit
            $inject_deleted_messages) as $e) {

        // Skip messages from other runs

        if ($e->run_id() != $run->id) continue ;

        // Put others into the array

        $t = $e->insert_time()->to64() ;
        if (!array_key_exists($t, $entries_by_timestamps)) $entries_by_timestamps[$t] = array () ;
        array_push (
            $entries_by_timestamps[$t] ,
            array (
                'type'   => 'entry' ,
                'object' => $e)) ;
    }

    // Merge in the run-specific entries which are explicitly associated with
    // the run, no matter when those messages were posted. Make sure there won't
    // be any duplicates. The duplicates are identified by timestamps.

    foreach ($experiment->entries_of_run($run->id(), $inject_deleted_messages) as $e) {
        $skip = false ;
        $t = $e->insert_time()->to64() ;
        if (array_key_exists($t, $entries_by_timestamps)) {
            foreach ($entries_by_timestamps[$t] as $pair)
                if (($pair['type'] == 'entry') && ($pair['object']->id() == $e->id())) {
                    $skip = true ;
                    break ;
                }
        } else {
            $entries_by_timestamps[$t] = array () ;
        }
        if ($skip) continue ;
        array_push (
            $entries_by_timestamps[$t] ,
            array (
                'type'   => 'entry' ,
                'object' => $e)) ;
    }

    // Now produce the desired output

    $result = array() ;

    foreach (sort_by_time($entries_by_timestamps) as $t)
        foreach ($entries_by_timestamps[$t] as $pair)
            array_push (
                $result ,
                $pair['type'] == 'entry' ?
                    LogBookUtils::entry2array($pair['object'],                $posted_at_instrument, $inject_deleted_messages) :
                    LogBookUtils::run2array  ($pair['object'], $pair['type'], $posted_at_instrument)) ;

    return array (
        'ResultSet' => array (
            'Result' => $result
        )
    ) ;
}) ;

?>

