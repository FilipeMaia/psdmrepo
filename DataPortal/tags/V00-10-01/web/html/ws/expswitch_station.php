<?php

/*
 * Return the information about an active experiment at the specified instrument
 * and a station
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;
    $station    = $SVC->required_int('station') ;

    $SVC->finish (array (
        'current' => DataPortal\SwitchUtils::current($SVC, $instr_name, $station)
    )) ;
}) ;
?>
