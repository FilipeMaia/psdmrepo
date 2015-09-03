<?php

/**
 * Return the list of EPICS sections for an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

/**
 * Return an array with definitions of EPICS PV sections and PVs
 * 
 * @see function LogBook\LogBookUtils::get_epics_sections()
 */
DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $SVC->finish (LogBook\LogBookUtils::get_epics_sections($experiment)) ;
}) ;

?>
