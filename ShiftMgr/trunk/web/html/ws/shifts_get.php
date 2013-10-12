<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use ShiftMgr\ShiftMgr ;
use ShiftMgr\Utils ;

/**
 * The Web Service to find shifts based on the specified criteria:
 * 
 * 1. Exact search
 * 
 *    <shift_id>
 * 
 * 2. Range search:
 * 
 *    <range> [<begin>] [<end>] [<stopper>] [<door>] [<lcls>] [<daq>] [<instruments>] [<types>] [<group_by_day>]
 *
 * Note that if the <group_by_day> flag is found among the parameters
 * of the request then the service will return a different object groupping
 * shifts by days (when they start).
 */
\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {

    // ------------------------------
    //   Try the exact search first
    // ------------------------------

    $shift_id = $SVC->optional_int('shift_id', null) ;
    if ($shift_id) {
        $shift = $SVC->shiftmgr()->find_shift_by_id($shift_id) ;
        if (!$shift) $SVC->abort("no shift found for ID {$shift_id}") ;
        $SVC->finish(array('shifts' => Utils::shifts2array(array($shift)))) ;
    }
    
    // ----------------------------
    //   Now try the range search
    // ----------------------------

    $group_by_day = $SVC->optional_flag('group_by_day') ;

    $shifts = Utils::query_shifts($SVC) ;

    if ($group_by_day) {

        // Shifts groupped by day across all requsted instruments (if any)

        $days2shifts = array() ;
        foreach ($shifts as $shift) {

            $begin_day = $shift->begin_time()->toStringDay() ;

            if (!array_key_exists($begin_day, $days2shifts)) {

                $area_problems = array() ;
                foreach (ShiftMgr::$area_names as $name)
                    $area_problems[$name] = 0 ;

                $days2shifts[$begin_day] = array (
                    'begin'  => array (
                        'day' => $begin_day ) ,
                    'num'    => 0 ,
                    'area_problems' => $area_problems ,
                    'shifts' => array()
                ) ;
            }

            $days2shifts[$begin_day]['num'] = $days2shifts[$begin_day]['num'] + 1 ;
            foreach ($shift->areas() as $area)
                if ($area->problems())
                    $days2shifts[$begin_day]['area_problems'][$area->name()] += 1 ;
            array_push (
                $days2shifts[$begin_day]['shifts'] ,
                Utils::shift2array($shift)
            ) ;
        }
        $days_keys = array_keys($days2shifts) ;
        sort($days_keys) ;
        rsort($days_keys) ;

        $days = array() ;
        foreach ($days_keys as $key)
            array_push ($days, $days2shifts[$key]) ;

        $SVC->finish(array('days' => Utils::days2array($days))) ;

    } else {
        
        // Shifts sorted by the begin time

        $SVC->finish(array('shifts' => Utils::shifts2array($shifts))) ;
    }
});

?>
