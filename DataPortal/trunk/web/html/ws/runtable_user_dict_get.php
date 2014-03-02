<?php

/**
 * Return a dictionary of column definitions for user tables
 * available for an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id  = $SVC->required_int('exper_id') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $types = array(
        'Editable' => array() ,
        'Run Info' => array(
            array('name' => 'Run Duration', 'descr' => 'Run Duration') ,
            array('name' => 'Run Title'   , 'descr' => 'Run Title'   )
        ) ,
        'Calibrations'  => array(
            array('name' => 'dark',     'descr' => 'dark') ,
            array('name' => 'flat',     'descr' => 'flat') ,
            array('name' => 'geometry', 'descr' => 'geometry')
        ) ,
        'DAQ Detectors' => array()
    ) ;

    $daq_detectors = LogBook\LogBookUtils::get_daq_detectors($experiment) ;
    foreach ($daq_detectors['detector_names'] as $d) {
        array_push(
            $types['DAQ Detectors'] ,
            array('name' => $d , 'descr' => $d)
        ) ;
    }

    $epics_sections = LogBook\LogBookUtils::get_epics_sections($experiment) ;
    foreach ($epics_sections['section_names'] as $section_name) {
        $section = $epics_sections['sections'][$section_name] ;
        $types['EPICS:'.$section['title']] = $section['parameters'] ;
    }

    $SVC->finish (array('types' => $types)) ;
}) ;

?>
