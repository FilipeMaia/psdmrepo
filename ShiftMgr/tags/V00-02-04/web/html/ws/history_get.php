<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Return the history in teh specified scope or range.
 * 
 * Depenidng on the input parameters, the service has two modes
 * of operation:
 * 
 * 1. Exact search based on the specified shift, area or time allocation
 *    identifier
 * 
 *    <shift_id> | <area_id> | <time_id>
 * 
 * 2. Range search
 * 
 *    [<instr_name>] <range> [<begin>] [<end>] [<since>]
 */

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {

    // First, try to see if the exact search requested

    $shift_id = $SVC->optional_int ('shift_id', null) ;
    if ($shift_id) {
        $shift = $SVC->shiftmgr()->find_shift_by_id($shift_id) ;
        if (!$shift) $SVC->abort("No shift found for id {$shift_id}") ;
        $SVC->finish (
            array (
                'history' => \ShiftMgr\Utils::history2array (
                    $shift->history())));
    }
    $area_id = $SVC->optional_int ('area_id', null) ;
    if ($area_id) {
        $area = $SVC->shiftmgr()->find_shift_area_by_id($area_id) ;
        if (!$area) $SVC->abort("No area found for id {$area_id}") ;
        $SVC->finish (
            array (
                'history' => \ShiftMgr\Utils::history2array (
                    $area->history())));
    }
    $time_id = $SVC->optional_int ('time_id', null) ;
    if ($time_id) {
        $time_allocation = $SVC->shiftmgr()->find_shift_time_by_id($time_id) ;
        if (!$time_allocation) $SVC->abort("No time allocation found for id {$time_id}") ;
        $SVC->finish (
            array (
                'history' => \ShiftMgr\Utils::history2array (
                    $time_allocation->history())));
    }

    // Now try searching by ranges

    $instr_name = $SVC->optional_str ('instr_name', null) ;
    $range      = $SVC->required_str ('range') ;
    $begin_time = $SVC->optional_time('begin', LusiTime::parse('2013-07-01 00:00:00')) ;
    $end_time   = $SVC->optional_time('end', null) ;
    $since_time = $SVC->optional_time('since', null) ;

    switch ($range) {
        case 'week' :
            $begin_time = LusiTime::minus_week() ;
            $end_time   = null ;
            break ;
        case 'month' :
            $begin_time = LusiTime::minus_month() ;
            $end_time   = null ;
            break ;
        case 'range' :
            if (!is_null($end_time) && !$begin_time->less($end_time))
                $SVC->abort('the begin time of the time interval to search for the history events must be strictly less than the end time') ;
            break ;
        default :
            $SVC->abort('unknown range type requested from the shifts service') ;
    }
    $SVC->finish (
        array (
            'history' => \ShiftMgr\Utils::history2array (
                $SVC->shiftmgr()->history (
                    $instr_name ,
                    $begin_time ,
                    $end_time ,
                    $since_time))));
});

?>
