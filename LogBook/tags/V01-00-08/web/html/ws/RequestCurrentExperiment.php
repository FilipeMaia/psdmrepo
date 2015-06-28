<?php

/*
 * Return JSON objects with the name of the current experiment.
 */

require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    
    $instr_name = $SVC->required_str('instr') ;
    $station    = $SVC->optional_int('station', 0) ;

    $switch = $SVC->safe_assign ($SVC->regdb()->last_experiment_switch($instr_name, $station) ,
                                 "no current experiment for instrument: {$instr_name}:{$station}") ;
    $exper_id = intval($switch['exper_id']) ;

    $experiment = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                     "failed to find experiment for id={$exper_id}") ;

    return array (
        "ResultSet" => array (
            "Result" => $experiment->is_facility() ?
                array (
                    "location"           => $experiment->instrument()->name() ,
                    "name"               => $experiment->name() ,
                    "id"                 => $experiment->id() ,
                    "registration_time"  => $experiment->registration_time()->toStringShort() ,
                    "description"        => $experiment->description()) :
                array (
                    "instrument"         => $experiment->instrument()->name() ,
                    "name"               => $experiment->name() ,
                    "id"                 => $experiment->id() ,
                    "begin_time"         => $experiment->begin_time()->toStringShort() ,
                    "end_time"           => $experiment->end_time()->toStringShort() ,
                    "registration_time"  => $experiment->registration_time()->toStringShort() ,
                    "description"        => $experiment->description()))) ;
}) ;

?>
