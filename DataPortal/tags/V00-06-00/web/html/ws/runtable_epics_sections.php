<?php

/**
 * Return the list of EPICS sections for an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

/**
 * Return an array with section descriptors:
 * 
 *   [ { 'name' : <section_name>,
 *       'title': <section_title>
 *     },
 *     ...
 *   ]
 * 
 * Note that the information is organized into an array to preserve
 * the order in which the sections should be used in the Web UI.
 */
DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $instr_name = $experiment->instrument()->name() ;

    $section_names = array() ;  // ordered list of names
    $sections = array() ;

    // Predefined sections first

    $in_use = array() ;
    
    foreach (array('HEADER', $instr_name) as $area) {
        foreach (LogBook\LogBookUtils::$sections[$area] as $section) {
            $parameters = array() ;
            foreach ($section['PARAMS'] as $p) {
                $p_name = $p['name'] ;
                $in_use[$p_name] = True ;
                array_push($parameters, $p_name) ;
            }
            $s_name = $section['SECTION'] ;
            array_push($section_names, $s_name) ;
            $sections[$s_name] = array (
                'title'      => $section['TITLE'] ,
                'parameters' => $parameters) ;
        }
    }
    
    // The last section is for any other parameters

    $parameters = array() ;
    foreach ($experiment->run_params() as $p) {
        $p_name = $p->name() ;
        if (!array_key_exists($p_name, $in_use)) {
            $in_use[$p_name] = True ;
            array_push($parameters, $p_name) ;
        }
    }
    $s_name = 'FOOTER' ;
    array_push($section_names, $s_name) ;
    $sections[$s_name] = array (
        'title'      => 'Additional Parameters' ,
        'parameters' => $parameters) ;

    $SVC->finish (array(
        'section_names' => $section_names ,
        'sections'      => $sections
    )) ;
}) ;

?>
