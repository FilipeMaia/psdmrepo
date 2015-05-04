<?php

/* This script will process a request for retreiving all runs of the experiments.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

use LogBook\LogBookUtils ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id      = $SVC->required_int('exper_id') ;
    $range_of_runs = $SVC->optional_str('range_of_runs', '') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("experiment for id={$exper_id}") ;

    if (!$SVC->logbookauth()->canRead($experiment->id())) $SVC->abort('not authorized for the operation') ;

    $first_run_num = null ;
    $last_run_num  = $first_run_num ;

    if ('' != $range_of_runs) {

        /* Parse the run numbers first. If the parse succeeds and no last run
         * is provided then assume the second run as the last one.
         */
        if (strpos($range_of_runs, '-') === false) {
            $r1 = $range_of_runs ;
            $r2 = '' ;
        } else {
            list($r1,$r2) = explode('-', $range_of_runs, 2) ;
        }
        $r1 = trim($r1) ;
        $r2 = trim($r2) ;
        if ('' == $r1) $SVC->abort('syntax error in the range of runs') ;

        $first_run_num = null ;
        if ((1 != sscanf($r1, "%d", $first_run_num)) or ($first_run_num <= 0))
            $SVC->abort('syntax error in the first run number of the range') ;

        $last_run_num = $first_run_num ;
        if ('' != $r2)
            if ((1 != sscanf($r2, "%d", $last_run_num)) or ($last_run_num <= 0))
                $SVC->abort('syntax error in the last run number of the range') ;

        if ($last_run_num < $first_run_num)
            $SVC->abort('last run in the range can\'t be less than the first one') ;

        $first_run = $experiment->find_run_by_num($first_run_num) ;
        if (is_null($first_run)) $SVC->abort("run {$first_run_num} can't be found") ;

        $last_run = $experiment->find_run_by_num($last_run_num) ;
        if (is_null($last_run)) $SVC->abort("run {$last_run_num} can't be found") ;
    }

    $max_total_seconds = 1 ;
    $runs = array() ;
    foreach ($experiment->runs() as $r) {

        // -- skip runs which are not allowed by the filter (if any provided)

        if (!is_null($first_run_num) &&
            (($r->num() < $first_run_num) || ($r->num() > $last_run_num))) continue ;

        $total_seconds = is_null($r->end_time()) ? 0 : $r->end_time()->sec - $r->begin_time()->sec ;
        if ($total_seconds > $max_total_seconds ) $max_total_seconds = $total_seconds ;
        $durat  = '' ;
        $durat1 = '' ;
        if ($total_seconds) {
            $seconds_left = $total_seconds ;

            $day          = floor($seconds_left / (24 * 3600)) ;
            $seconds_left = $seconds_left % (24 * 3600) ;

            $hour         = floor($seconds_left / 3600) ;
            $seconds_left = $seconds_left % 3600 ;

            $min          = floor($seconds_left / 60) ;
            $seconds_left = $seconds_left % 60 ;

            $sec          = $seconds_left;

            $durat = sprintf("%02d:%02d.%02d", $hour, $min, $sec) ;
            $durat1 = LogBookUtils::format_seconds_1($total_seconds) ;
        }
        array_push (
            $runs ,
            array (
                'id'     => $r->id() ,
                'num'    => $r->num() ,
                'day'    => $r->begin_time()->toStringDay() ,
                'ymd'    => $r->begin_time()->toStringDay() ,
                'hms'    => $r->begin_time()->toStringHMS() ,
                'ival'   => $r->begin_time()->toStringHMS().(is_null($r->end_time()) ? ' - <span style="color:red; font-weight:bold;">on-going</span>' : ' - '.$r->end_time()->toStringHMS()) ,
                'durat'  => $durat ,
                'durat1' => $durat1 ,
                'sec'    => $total_seconds)) ;
    }

    $SVC->finish (array (
        'Runs'       => array_reverse($runs) ,  // the newest run shoudl go first
        'MaxSeconds' => $max_total_seconds
    )) ;
}) ;

?>
