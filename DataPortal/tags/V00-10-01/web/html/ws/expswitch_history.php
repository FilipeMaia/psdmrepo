<?php

/*
 * Return the history records in a scope of the specified instrument
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;

    $history = array() ;
    foreach ($SVC->regdb()->experiment_switches($instr_name) as $switch) {
        $exper_id = intval($switch['exper_id']) ;
        $exper = $SVC->logbook()->find_experiment_by_id($exper_id) ;
        if (!$exper)
            $SVC->abort("the active experiment with id={$exper_id} is no longer registered") ;

        $requestor_uid = $switch['requestor_uid'] ;
        $requestor_account = $SVC->regdb()->find_user_account($requestor_uid) ;
        if (!$requestor_account)
            $SVC->abort("user account not found for: {$requestor_uid}") ;

        $switch_time = LusiTime::from64(intval($switch['switch_time'])) ;

        array_push($history, array (
            'exper_id'        => $exper->id() ,
            'exper_name'      => $exper->name() ,
            'station'         => intval($switch['station']) ,
            'switch_time'     => array (
                'sec_nsec' => $switch_time->to64() ,
                'time'     => $switch_time->toStringShort() ,
                'ymd'      => $switch_time->toStringDay() ,
                'hms'      => $switch_time->toStringHMS()
            ) ,
            'requestor_uid'   => $requestor_uid ,
            'requestor_gecos' => $requestor_account['gecos'])) ;
    }
    $SVC->finish (array (
        'history' => $history
    )) ;
}) ;
?>
