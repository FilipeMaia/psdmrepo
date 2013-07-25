<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {

    $instr_name = $SVC->required_str ('instr_name') ;
    $range      = $SVC->required_str ('range') ;
    $all        = $SVC->optional_bool('all', true) ;
    $begin_time = $SVC->optional_time('begin', LusiTime::parse('2013-07-01 00:00:00')) ;
    $end_time   = $SVC->optional_time('end',   null) ;

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
                $SVC->abort('the begin time of the time interval to search for shifts must be strictly less than the end time') ;
            break ;
        default :
            $SVC->abort('unknown range type requested from the shifts service') ;
    }

    $SVC->shiftmgr()->precreate_shifts_if_needed($instr_name, $begin_time, $end_time) ;

    $all_shifts = $SVC->shiftmgr()->shifts($instr_name, $begin_time, $end_time) ;
    $shifts = array() ;
    if ($all)
        $shifts = $all_shifts ;
    else
        foreach ($all_shifts as $shift)
            if ($shift->stopper() || $shift->door())
                array_push ($shifts, $shift) ;

    $SVC->finish(array('shifts' => \ShiftMgr\Utils::shifts2array($shifts)));
});

?>
