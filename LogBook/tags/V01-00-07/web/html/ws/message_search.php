<?php

/*
 * This script will perform the search for a single free-form entry (or entries)
 *
 * PARAMETERS:
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LogBook\LogBookUtils ;

use LusiTime\LusiTime ;

/**
 * Translate timestamps which may also contain shortcuts
 */
function translate_time ($experiment, $str) {
    $str_trimmed = trim($str) ;
    if ($str_trimmed == '') return null ;
    switch ($str_trimmed[0]) {
        case 'b':
        case 'B': return $experiment->begin_time() ;
        case 'e':
        case 'E': return $experiment->end_time() ;
        case 'm':
        case 'M': return LusiTime::minus_month() ;
        case 'w':
        case 'W': return LusiTime::minus_week() ;
        case 'd':
        case 'D': return LusiTime::minus_day() ;
        case 'y':
        case 'Y': return LusiTime::yesterday() ;
        case 't':
        case 'T': return LusiTime::today() ;
        case 'h':
        case 'H': return LusiTime::minus_hour() ;
    }
    $result = LusiTime::parse($str_trimmed) ;
    if (!$result) $result = LusiTime::from64($str_trimmed) ;
    return $result ;
}

/**
 * The functon will produce a sorted list of timestamps based on keys of
 * the input dictionary. If two consequitive begin/end run records are found
 * for the same run then the records will be collapsed into a single record 'run'
 * with the begin run timestamp. The list may also be truncated if the limit has
 * been requested. In that case excessive entries will be removed from the _HEAD_
 * of the input array.
 * 
 * NOTE: The contents of input array will be modified for collapsed runs
 *       by replacing types for 'begin_run' / 'end_run' with just 'run'.
 */
function sort_and_truncate_from_head ($SVC, &$entries_by_timestamps, $limit) {

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

    // Do need to truncate. Apply different limiting techniques depending
    // on a value of the parameter.

    if (!$limit) return $timestamps ;

    $result = array() ;

    $limit_num = null ;
    $unit      = null ;
    if (2 == sscanf($limit, "%d%s", $limit_num, &$unit)) {

        $nsec_ago = 1000000000 * $limit_num ;
        switch ($unit) {
            case 's': break ;
            case 'm': $nsec_ago *=            60 ; break ;
            case 'h': $nsec_ago *=          3600 ; break ;
            case 'd': $nsec_ago *=     24 * 3600 ; break ;
            case 'w': $nsec_ago *= 7 * 24 * 3600 ; break ;
            default :
                $SVC->abort('illegal format of the limit parameter') ;
        }
        $now_nsec = LusiTime::now()->to64() ;
        foreach ($timestamps as $t) {
            if ($t >= ($now_nsec - $nsec_ago)) array_push($result, $t) ;
        }

    } else {

        $limit_num = (int)$limit ;

        /* Return the input array if no limit specified or if the array is smaller
         * than the limit.
         */
        if (count($timestamps) <= $limit_num) return $timestamps ;

        $idx = 0 ;
        $first2copy_idx =  count($timestamps) - $limit_num ;

        foreach ($timestamps as $t) {
            if ($idx >= $first2copy_idx) array_push($result, $t) ;
            $idx = $idx + 1 ;
        }
    }
    return $result ;
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {


    // -------------------------------------------------------------------------
    // -------------------- PARSE PARAMETERS OF THE SCRIPT ---------------------
    // -------------------------------------------------------------------------

    $exper_id = $SVC->required_int('id') ;

    $shift_id = $SVC->optional_int('shift_id', null) ;
    $run_id   = $SVC->optional_int('run_id',   null) ;

    $SVC->assert (!($shift_id && $run_id) ,
                  "'shift_id' and 'run_id' can not be used witin the same request") ;

    $text2search = $SVC->optional_str('text2search', '') ;

    $search_in_messages = $SVC->optional_bool('search_in_messages', false) ;
    $search_in_tags     = $SVC->optional_bool('search_in_tags',     false) ;
    $search_in_values   = $SVC->optional_bool('search_in_values',   false) ;

    $SVC->assert ($search_in_messages || $search_in_tags || $search_in_values ,
                  "('search_in_messages', 'search_in_tags', 'search_in_values') not in the request" ) ;

    $posted_at_instrument = $SVC->optional_bool('posted_at_instrument', false) ;
    $posted_at_experiment = $SVC->optional_bool('posted_at_experiment', false) ;
    $posted_at_shifts     = $SVC->optional_bool('posted_at_shifts',     false) ;
    $posted_at_runs       = $SVC->optional_bool('posted_at_runs',       false) ;

    $SVC->assert ($posted_at_experiment || $posted_at_shifts || $posted_at_runs ,
                  "('posted_at_experiment', 'posted_at_shifts', 'posted_at_runs') not in the request" ) ;


    $begin_str = $SVC->optional_str('begin',  '') ;
    $end_str   = $SVC->optional_str('end',    '') ;
    $tag       = $SVC->optional_str('tag',    '') ;
    $author    = $SVC->optional_str('author', '') ;

    $range_of_runs = $SVC->optional_str ('range_of_runs', '') ;
    $inject_runs   = $SVC->optional_bool('inject_runs',   false) ;

    if ($range_of_runs) {

        $SVC->assert (!($begin_str || $end_str) ,
                      "begin/end time limits can't be used together with the range of runs") ;

        $SVC->assert (!($shift_id || $run_id) ,
                      "shidt_id or run_id can't be used together with the range of runs") ;

        $inject_runs = true ;       // Force it because a request was made to search around
                                    // a specific run or a range of those.

        $posted_at_instrument = false ; // the range of runs isn't compatible with
                                        // the broader search accross instruments
    }

    $inject_deleted_messages = $SVC->optional_flag('inject_deleted_messages') ;

    // This is a special modifier which (if present) is used to return an updated list
    // of messages since (strictly newer than) the specified time.
    // 
    // NOTES:
    // - this parameter will only be respected if it strictly falls into
    //   the [begin,end) interval of the request!
    // - unlike outher time related parameters of the service this one is expected
    //   to be a full precision 64-bit numeric representation of time.

    $since_str = $SVC->optional_str('since', '') ;

    // This is a special modifier which (if present) is used to return a list
    // of messages before (strictly older than) the specified time.
    // 
    // NOTES:
    // - this parameter will effectively override the [,end) time
    // - this parameter will only be respected if it strictly falls into
    //   the [begin,end) interval of the request!
    // - unlike outher time related parameters of the service this one is expected
    //   to be a full precision 64-bit numeric representation of time.

    $before_str = $SVC->optional_str('before', '') ;

    // This is a special modifier which (if present) is used to return a shortened list
    // of messages.

    $limit = $SVC->optional_str('limit', null) ;
    if ($limit == 'all') $limit = null ;




    // -------------------------------------------------------------------------
    // ---------------------- EXECUTE THE REQUEST ------------------------------
    // -------------------------------------------------------------------------

    // Make adjustments relative to the primary experiment of the search.

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     'no such experiment') ;

    // Recalculate 'begin' or 'end' time limits if a range of runs was requested
    // and find entries which were explicitly associated with runs.
    //
    // The new limits will include all messages posted:
    // 
    //   - since the start of the first.
    //
    //   - before the beginning of the next to the last run. Of no such next run
    //     is found then all messages till the end of the logbook will be selected.

    $run_specific_entries = array() ;

    if ($range_of_runs) {

        // Parse the run numbers first. If the parse succeeds and no last run
        // is provided then assume the second run as the last one.

        if (strpos($range_of_runs, '-') === false) {
            $r1 = $range_of_runs ;
            $r2 = '' ;
        } else {
            list($r1,$r2) = explode('-', $range_of_runs, 2) ;
        }
        $r1 = $SVC->safe_assign (trim($r1) ,"syntax error in the range of runs") ;
        $r2 = trim($r2) ;

        $first_run_num = null ;
        $SVC->assert (sscanf($r1, "%d", $first_run_num) == 1 && $first_run_num > 0 ,
                      'syntax error in the first run number of the range') ;

        $last_run_num = $first_run_num ;
        if ($r2)
            $SVC->assert (sscanf($r2, "%d", $last_run_num) == 1 && $last_run_num > 0 && $last_run_num >= $first_run_num ,
                          'syntax error in the last run number of the range') ;

        $first_run = $SVC->safe_assign ($experiment->find_run_by_num($first_run_num) ,
                                        "run {$first_run_num} can't be found") ;
                                        
        $last_run  = $SVC->safe_assign ($experiment->find_run_by_num($last_run_num) ,
                                        "run {$last_run_num} can't be found") ;

        $begin_str = $first_run->begin_time()->toStringShort() ;

        $next_run = $experiment->find_run_by_num($last_run_num + 1) ;
        if ($next_run) {
            $end_str = $next_run->begin_time()->toStringShort();
        } else {
            $end_str = '' ;
        }

        // Find messages which are explicitly associated with runs, no matter
        // when those messages were posted. The messages (if any) will be automatically
        // mixed into the output result.

        for ($num = $first_run_num; $num <= $last_run_num; ++$num) {
            $run = $experiment->find_run_by_num($num) ;
            if (!$run) continue ;

            foreach ($experiment->entries_of_run($run->id(), $inject_deleted_messages) as $e)
                array_push($run_specific_entries, $e) ;
        }
    }

    // Timestamps are translated here because of possible shoftcuts which
    // may refer to the experiment's validity limits.

    $begin = $begin_str ?
        $SVC->safe_assign (translate_time($experiment, $begin_str) ,
                           'begin time has invalid format') :
        null ;

    $end = $end_str ?
        $SVC->safe_assign (translate_time($experiment, $end_str) ,
                           'end time has invalid format') :
        null ;

    if ($before_str)
        $end = $SVC->safe_assign (LusiTime::from64($before_str) ,
                                  'before time has invalid format') ;

    $SVC->assert (!($begin && $end && !$begin->less($end)) ,
                  "invalid interval - begin time isn't strictly less than the end one") ;
            
    // For explicitly specified shifts and runs force the search limits not
    // to exceed their intervals (if the one is specified).

    $begin4runs = $begin ;
    $end4runs   = $end ;
    if ($shift_id) {
        $shift = $SVC->safe_assign ($experiment->find_shift_by_id($shift_id) ,
                                    "no shift with shift_id=".$shift_id." found") ;

        $begin4runs = $shift->begin_time() ;
        $end4runs   = $shift->end_time() ;
    }
    if ($run_id) {
        $run = $SVC->safe_assign ($experiment->find_run_by_id($run_id) ,
                                  "no run with run_id=".$run_id." found") ;

        $begin4runs = $run->begin_time() ;
        $end4runs   = $run->end_time() ;
    }
    $since = $since_str ?
        $SVC->safe_assign (LusiTime::from64($since_str) ,
                           "illegal value of parameter 'since'") :
        null ;

    // Read just the 'begin' parameter for runs if 'since' is present.
    // Completelly ignore 'since' if it doesn't fall into an interval of
    // the requst.
    
    if ($since) {
        $since4runs = $since ;
        if ($begin4runs && $since->less($begin4runs)) {
            $since4runs = null ;
        }
        if ($end4runs && $since->greaterOrEqual($end4runs)) {
            $since4runs = null ;
        }
        if ($since4runs) $begin4runs = $since4runs ;
    }

    // Mix entries and run records in the right order. Results will be merged
    // into this dictionary before returning to the client.

    $entries_by_timestamps = array () ;

    // Scan all relevant experiments. Normally it would be just one. However, if
    // the instrument is selected then all experiments of the given instrument will
    // be taken into consideration.

    $experiments = $posted_at_instrument ?
        $SVC->logbook()->experiments_for_instrument($experiment->instrument()->name()) :
        $experiments = array ($experiment);

    foreach ($experiments as $e) {

        // Filter experiments based on user privileges

        if (!$SVC->logbookauth()->canRead($e->id())) {

            // Silently skip this experiemnt if browsing accross the whole instrument.
            // The only exception would be the main experient from which we started
            // things.

            if ($posted_at_instrument && ($e->id() != $exper_id)) continue ;

            $SVC->abort('not authorized to read messages for the experiment') ;
        }
 
        // Get the info for entries and (if requested) for runs.
        // 
        // NOTE: If the full text search is involved then the search will
        //       propagate down to children subtrees as well. However, the resulting
        //       list of entries will only contain the top-level ("thread") messages.
        //       To ensure so we're going to pre-scan the result of the query to identify
        //       children and finding their top-level parents. The parents will be put into
        //       the result array. Also note that we're not bothering about having duplicate
        //       entries in the array becase this will be sorted out on the next step.

        $entries = array() ;
        foreach (
            $e->search (
                $e->id() == $exper_id ? $shift_id : null ,  // the parameter makes sense for the main experiment only
                $e->id() == $exper_id ? $run_id   : null ,  // ditto
                $text2search ,
                $search_in_messages ,
                $search_in_tags ,
                $search_in_values ,
                $posted_at_experiment ,
                $posted_at_shifts ,
                $posted_at_runs ,
                $begin ,
                $end ,
                $tag ,
                $author ,
                $since ,
                null , /* $limit */
                $inject_deleted_messages ,
                $search_in_messages && $text2search     // include children into the search for
                                                        // the full-text search in message bodies.
            ) as $entry) {

                $parent = $entry->parent_entry() ;
                if (!$parent) {
                    array_push($entries, $entry) ;
                } else {
                    while (true) {
                        $parent_of_parent = $parent->parent_entry() ;
                        if (!$parent_of_parent) break ;
                        $parent = $parent_of_parent ;
                    }
                    array_push($entries, $parent) ;
                }
        }

        $runs = !$inject_runs ? array () : $e->runs_in_interval($begin4runs, $end4runs /*, $limit*/) ;

        // Merge both results into the dictionary for further processing.

        foreach ($entries as $e) {
            $t = $e->insert_time()->to64() ;
            if (!array_key_exists($t, $entries_by_timestamps)) $entries_by_timestamps[$t] = array () ;
            array_push (
                $entries_by_timestamps[$t] ,
                array (
                    'type'   => 'entry' ,
                    'object' => $e));
        }
        foreach ($runs as $r) {

            // The following fix helps to avoid duplicating "begin_run" entries because
            // the way we are getting runs (see before) would yeld runs in the interval:
            //
            //   [begin4runs,end4runs)

            if (!$begin4runs || $begin4runs->less($r->begin_time())) {
                $t = $r->begin_time()->to64() ;
                if (!array_key_exists($t, $entries_by_timestamps)) $entries_by_timestamps[$t] = array () ;
                array_push (
                    $entries_by_timestamps[$t] ,
                    array (
                        'type'   => 'begin_run' ,
                        'object' => $r)) ;
            }

            if ($r->end_time()) {
                $t = $r->end_time()->to64() ;
                if (!array_key_exists($t, $entries_by_timestamps)) $entries_by_timestamps[$t] = array () ;
                array_push (
                    $entries_by_timestamps[$t] ,
                    array (
                        'type'   => 'end_run' ,
                        'object' => $r)) ;
            }
        }
    }

    // Merge in the run-specific entries. Make sure there won't be any duplicates.
    // The duplicates can be easily identified becouase they would be within
    // the same timestamps.

    foreach ($run_specific_entries as $e) {
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

    foreach (sort_and_truncate_from_head($SVC, $entries_by_timestamps, $limit) as $t)
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

