<?php

/**
 * Return the information about EPICS variables configured for runs of the experiment.
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
 * Note that teh information is organized into an array to preserve
 * the order in which the sections should be used in the Web UI.
 * 
 * @param string $instr_name
 * @return array
 */
function sections ($instr_name, $parameters) {

    $result = array() ;

    $in_use = array() ;     // parameters found in one of the sections below

    // Package parameters from predefined sections first

    foreach (array('HEADER', $instr_name) as $area) {
        foreach (LogBook\LogBookUtils::$sections[$area] as $section) {
            $parameter_names = array() ;
            foreach ($section['PARAMS'] as $param) {
                $name = $param['name'] ;
                if (array_key_exists($name, $in_use)) continue ;
                array_push($parameter_names, $name) ;
                $in_use[$name] = 1 ;
            }
            array_push($result, array (
                'name'       => $section['SECTION'] ,
                'title'      => $section['TITLE'] ,
                'parameters' => $parameter_names)) ;
        }
    }
    
    // The remaining parameters will go into the last section

    $parameter_names = array() ;
    foreach ($parameters as $param) {
        $name = $param->name() ;
        if (array_key_exists($name, $in_use)) continue ;
        array_push($parameter_names, $name) ;
    }
    array_push($result, array (
        'name'  => 'FOOTER' ,
        'title' => 'Additional Parameters',
        'parameters' => $parameter_names)) ;

    return $result ;
}

/**
 * Return a combined dictionary of predefined and database parameters
 * organized like a nested dictonary:
 * 
 *   { <param> : { 'section': <sect>,
 *                 'descr'  : <descr>'
 *               }
 *   }
 * 
 * @param string $instr_name
 * @param array $parameters
 * @return array
 */
function parameter2section_description ($instr_name, $parameters) {

    $result = array() ;

    // First, go through the list of predefined (hardwired) parameters
    // to figure our their associations with sections.

    foreach (array('HEADER', $instr_name) as $area)
        foreach (LogBook\LogBookUtils::$sections[$area] as $section)
            foreach ($section['PARAMS'] as $param)
                $result[$param['name']] = array (
                    'section' => $section['SECTION'] ,
                    'descr'   => $param['descr']) ;


    // Then go through the list of parameters found in the database to
    // pick up the remaining ones not found witin any predefined sections.
    // Those would go to 'FOOTER'.

    foreach ($parameters as $param) {
        $name = $param->name() ;
        if (!array_key_exists($name, $result)) {
            $descr = $param->description() ;
            if ($descr === '') $descr = $name ;
            $result[$name] = array (
                'section' => 'FOOTER' ,
                'descr'   => $descr) ;
        }
    }
    return $result ;
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id       = $SVC->required_int('exper_id') ;
    $from_runnum    = $SVC->optional_int('from_run', 0) ;
    $through_runnum = $SVC->optional_int('through_run', 0) ;

    if (($from_runnum && $through_runnum) && ($from_runnum > $through_runnum))
        $SVC->abort("illegal range of runs: make sure the second run is equal or greater then the first one") ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $instr_name = $experiment->instrument()->name() ;

    $runs = array() ;
    foreach ($experiment->runs() as $run) {

        $runnum = $run->num() ;
        if ($from_runnum    && ($runnum < $from_runnum))    continue ;
        if ($through_runnum && ($runnum > $through_runnum)) continue ;

        $run_params = array() ;
        foreach ($run->values() as $param_value)
            $run_params[$param_value->name()] = $param_value->value() ;

        $runs[$runnum] = $run_params ;
    }
    $parameters = $experiment->run_params('') ;

    $SVC->finish (array(
        'sections'   => \sections($instr_name, $parameters) ,
        'parameters' => \parameter2section_description($instr_name, $parameters) ,
        'runs'       => $runs
    )) ;
}) ;

?>
