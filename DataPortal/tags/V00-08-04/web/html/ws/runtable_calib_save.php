<?php

/**
 * Update calibration status of the specified run of the experiment.
 * Return the updated record of that run.
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {
    $exper_id = $SVC->required_int('exper_id') ;
    $runnum   = $SVC->required_int('run') ;
    $dark     = $SVC->required_int('dark') ;
    $flat     = $SVC->required_int('flat') ;
    $geom     = $SVC->required_int('geom') ;
    $comment  = $SVC->optional_str('comment', '') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $run = $experiment->find_run_by_num($runnum) ;
    if (!$run) $SVC->abort("no run {$runnum} found within experiment with id={$xper_id}") ;

    $run->set_attr_val_INT ('Calibrations', 'dark',    $dark) ;
    $run->set_attr_val_INT ('Calibrations', 'flat',    $flat) ;
    $run->set_attr_val_INT ('Calibrations', 'geom',    $geom) ;
    $run->set_attr_val_TEXT('Calibrations', 'comment', $comment) ;

    $runs = array() ;
    $runs[$runnum] = array(
        'dark'    => $dark ? '1' : '' ,
        'flat'    => $flat ? '1' : '',
        'geom'    => $geom ? '1' : '',
        'comment' => $comment
    ) ;
    $SVC->finish (array('runs' => $runs)) ;
}) ;

?>
