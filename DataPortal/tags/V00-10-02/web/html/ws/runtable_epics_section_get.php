<?php

/**
 * Return values of parameters found in the specified EPICS section
 * for a range of runs of an  experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id       = $SVC->required_int('exper_id') ;
    $from_runnum    = $SVC->optional_int('from_run', 0) ;
    $through_runnum = $SVC->optional_int('through_run', 0) ;
    $section        = $SVC->optional_str('section', '') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $instr_name = $experiment->instrument()->name() ;

    if ($from_runnum < 0) {
        $last_run = $experiment->find_last_run() ;
        if ($last_run) {
            $runnum = $last_run->num() ;
            $from_runnum = abs($from_runnum) <= $runnum ? $runnum - abs($from_runnum) : 0 ;
        }
    }
    
    if (($from_runnum && $through_runnum) && ($from_runnum > $through_runnum))
        $SVC->abort("illegal range of runs: make sure the second run is equal or greater then the first one") ;

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

    $SVC->finish (array(
        'runs' => $runs
    )) ;
}) ;

?>
