<?php

/*
 * Return JSON objects with a list of experiments.
 */

require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name  = $SVC->optional_str ('instr', null) ;
    $is_location = $SVC->optional_flag('is_location', false) ;

    $all_experiments = $instr_name ?
        $SVC->regdb()->experiments_for_instrument($instr_name) :
        $SVC->regdb()->experiments() ;

    // Leave only those experiments the logged user is authorizated to see

    $experiments2array = array () ;
    foreach ($all_experiments as $e)
        if ($SVC->logbookauth()->canRead($e->id())) {
            if ($is_location xor $e->is_facility()) continue ;
            array_push($experiments2array , $e->is_facility() ?
                array (
                    "location"           => $e->instrument()->name() ,
                    "name"               => $e->name() ,
                    "id"                 => $e->id() ,
                    "registration_time"  => $e->registration_time()->toStringShort() ,
                    "description"        => $e->description()) :
                array (
                    "instrument"        => $e->instrument()->name(),
                    "name"              => $e->name(),
                    "id"                => $e->id(),
                    "begin_time"        => $e->begin_time()->toStringShort(),
                    "end_time"          => $e->end_time()->toStringShort(),
                    "registration_time" => $e->registration_time()->toStringShort(),
                    "description"       => $e->description())) ;
        }

    return array (
        "ResultSet" => array (
            "Result" => $experiments2array)) ;

}) ;
?>
