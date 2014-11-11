<?php

/**
 * This service will return the statistics for the data migraton rates in
 * the specified context and an interval of time.
 * 
 * Parameters:
 * 
 *   <instr_name> [<direction>] [<begin_time>] [<end_time>] [<max_entries>]
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;
    if (!$SVC->regdb()->find_instrument_by_name($instr_name))
        $SVC->abort("no such instrument '{$instr_name}' exists") ;

    $KNOWN_DIRECTIONS = array('DSS2FFB', 'FFB2ANA', 'OTHER') ;

    $directions = $KNOWN_DIRECTIONS ;
    $direction  = $SVC->optional_str('direction', null) ;
    if (!is_null($direction)) {
        if (!in_array($direction, $KNOWN_DIRECTIONS))
            $SVC->abort("no such instrument '{$instr_name}' exists") ;
        $directions = array($direction) ;
    }
    $begin_time_sec = $SVC->optional_int('begin_time', 0) ;
    $end_time_sec   = $SVC->optional_int('end_time', \LusiTime\LusiTime::now()->sec) ;
    if ($begin_time_sec > $end_time_sec) {
        $SVC->abort('the end_time must be less or equal to the begin_time') ;
    }
    $max_entries = $SVC->optional_int('max_entries', 25) ;
    if ($max_entries < 1) {
        $SVC->abort('the max_ebtry parameter must be greater than 0') ;
    }
    $instr_name_lc = strtolower($instr_name) ;

    $SIMULATED_RATES = array (
        array(  0, 134,   45, 304,   9) ,
        array( 20, 100,    5, 355, 189) ,
        array( 40, 120,    0,   0, 100) ,
        array( 80, 100,   45, 104,  80) ,
        array(120, 178,    5, 204,  22) ,
        array(160, 190,   45, 304,   0) ,
        array(180, 140,   45,  04,   0) ,
        array(100, 120,   55,  30,   0) ,
        array( 10, 150,   45,  57,   2) ,
        array(  0, 130,   45, 100,  45)
    ) ;
    $SIMULATED_SHIFT_PER_DIRECTION = array (
        'DSS2FFB' => 0 ,
        'FFB2ANA' => 3 ,
        'OTHER'   => 7
    ) ;
    $SIMULATED_HOSTS_PER_DIRECTION = array (
        'DSS2FFB' => array (
            "daq-{$instr_name_lc}-dss01" ,
            "daq-{$instr_name_lc}-dss02" ,
            "daq-{$instr_name_lc}-dss03" ,
            "daq-{$instr_name_lc}-dss04" ,
            "daq-{$instr_name_lc}-dss05") ,
        
        'FFB2ANA' => array (
            "psana{$instr_name_lc}ffb01" ,
            "psana{$instr_name_lc}ffb02" ,
            "psana{$instr_name_lc}ffb03" ,
            "psana{$instr_name_lc}ffb04" ,
            "psana{$instr_name_lc}ffb05") ,
        'OTHER'   => array (
            "{$instr_name_lc}-export01" ,
            "{$instr_name_lc}-export02" ,
            "{$instr_name_lc}-export03" ,
            "{$instr_name_lc}-export04" ,
            "{$instr_name_lc}-export05" )
    ) ;

    $begin_time_sec = $begin_time_sec ? $begin_time_sec : $end_time_sec - $max_entries ;
    $xfer = array (
        'begin_time_sec' => $begin_time_sec ,
        'end_time_sec'   => $end_time_sec ,
        'request_time'   => $end_time_sec ,
        'directions'     => array()
    ) ;
    foreach ($directions as $direction) {
        $stats = array() ;
        for ($ts = $end_time_sec; $ts > $begin_time_sec; $ts-- ) {
            $time = new \LusiTime\LusiTime($ts, 0) ;
            array_push($stats, array (
                'time'      => array (
                    'day'   => $time->toStringDay() ,
                    'hms'   => $time->toStringHMS()) ,
                'timestamp' => $end_time_sec ,
                'rates'     => $SIMULATED_RATES[($ts + $SIMULATED_SHIFT_PER_DIRECTION[$direction]) % 10]
            )) ;
        }
        $xfer['directions'][$direction] = array (
            'in_hosts'       => $SIMULATED_HOSTS_PER_DIRECTION[$direction] ,
            'stats'          => $stats
        ) ;
    }
    return array('xfer' => $xfer) ;
}) ;
  
?>
