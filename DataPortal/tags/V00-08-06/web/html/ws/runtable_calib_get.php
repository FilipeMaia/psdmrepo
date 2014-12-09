<?php

/**
 * Return calibration information about all known runs of teh experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id       = $SVC->required_int('exper_id') ;
    $from_runnum    = $SVC->optional_int('from_run', 0) ;
    $through_runnum = $SVC->optional_int('through_run', 0) ;

    if (($from_runnum && $through_runnum) && ($from_runnum > $through_runnum))
        $SVC->abort("illegal range of runs: make sure the second run is equal or greater then the first one") ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $runs = array() ;
    foreach ($experiment->runs() as $run) {

        $runnum = $run->num() ;
        if ($from_runnum    && ($runnum < $from_runnum))    continue ;
        if ($through_runnum && ($runnum > $through_runnum)) continue ;

        $dark = 0 ;
        $flat = 0 ;
        $geom = 0 ;
        $comment = '' ;

        foreach ($run->attributes('Calibrations') as $attr) {
            switch ($attr->name()) {
                case 'dark'    : $dark    = $attr->val() ; break ;
                case 'flat'    : $flat    = $attr->val() ; break ;
                case 'geom'    : $geom    = $attr->val() ; break ;
                case 'comment' : $comment = $attr->val() ; break ;
            }
        }
        $runs[$runnum] = array (
            'dark'    => $dark ? '1' : '',
            'flat'    => $flat ? '1' : '',
            'geom'    => $geom ? '1' : '',
            'comment' => $comment
        ) ;
    }
    
    $SVC->finish (array('runs' => $runs)) ;
}) ;

?>
