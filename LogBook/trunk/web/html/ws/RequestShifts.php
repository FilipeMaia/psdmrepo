<?php

/**
 * Return a list of shifts for teh experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;


/* Translate timestamps which may also contain shortcuts
 */
function translate_time ($experiment, $str) {
    $str = strtoupper(trim($str)) ;
    if ($str === '') return null ;
    switch ($str[0]) {
        case 'B' : return $experiment->begin_time() ;
        case 'E' : return $experiment->end_time() ;
        case 'M' : return LusiTime::minus_month() ;
        case 'W' : return LusiTime::minus_week() ;
        case 'D' : return LusiTime::minus_day() ;
        case 'Y' : return LusiTime::yesterday() ;
        case 'T' : return LusiTime::today() ;
        case 'H' : return LusiTime::minus_hour() ;
    }
    $result = LusiTime::parse($str) ;
    if ($result) return $result ;
    return LusiTime::from64($str) ;
}

function shift2array ($shift) {
    $begin_time_url =
        "<a href=\"javascript:select_shift(".$shift->id().")\" class=\"lb_link\">" .
        $shift->begin_time()->toStringShort() .
        '</a>' ;
    $end_time_status =
        is_null( $shift->end_time()) ?
        '<b><em style="color:red;">on-going</em></b>' :
        $shift->end_time()->toStringShort() ;

    return array (
        "id"         => $shift->id() ,
        "begin_time" => $begin_time_url ,
        "end_time"   => $end_time_status ,
        "leader"     => $shift->leader() ,
        "num_runs"   => $shift->num_runs()
    ) ;
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id             = $SVC->required_int ('id') ;
    $begin_str            = $SVC->optional_str ('begin', '') ;
    $end_str              = $SVC->optional_str ('end',   '') ;
    $last_shift_requested = $SVC->optional_flag('last',  false) ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $instrument = $experiment->instrument();

    if (!$SVC->logbookauth()->canRead ($experiment->id()))
        $SVC->abort('You are not authorized to access any information about the experiment') ;

    // Timestamps are translated here because of possible shoftcuts which
    // may refer to the experiment's validity limits.
    //
    $begin = null ;
    if ($begin_str !== '') {
        $begin = translate_time($experiment, $begin_str) ;
        if (!$begin) $SVC->abort('begin time has invalid format') ;
    }
    $end = null ;
    if ($end_str !== '') {
        $end = translate_time($experiment, $end_str) ;
        if (!$end ) $SVC->abort('end time has invalid format')  ;
    }
    if ($begin && $end && !$begin->less($end))
        $SVC->abort("invalid interval - begin time isn't strictly less than the end one") ;

    if (($begin || $end) && $last_shift_requested)
        $SVC->abort("conflicting options - last shift can't be requested along with begin or end times") ;

    $shifts = array() ;
    if ($last_shift_requested) {
        $last_shift = $experiment->find_last_shift() ;
        if ($last_shift)
            array_push($shifts, shift2array($last_shift)) ;
    } else {
        foreach ($experiment->shifts_in_interval($begin, $end) as $shift)
            array_push($shifts, shift2array($shift)) ;
    }
    $SVC->finish (
        array (
            'ResultSet' => array (
                'Result' => $shifts
            )
        )
   ) ;
}) ;

?>
