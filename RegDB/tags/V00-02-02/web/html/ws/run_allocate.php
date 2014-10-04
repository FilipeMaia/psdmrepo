<?php

/*
 * Allocate the next run number for an experiment.
 * 
 * The run record will get permanently stored in the database. The service will return
 * the number in a JSON object
 * 
 * AUTHORIZATION: being able to post messages for the experiment's e-log,
 * or 
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;

    $experiment = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $is_authorized = $SVC->logbookauth()->canPostNewMessages($experiment->id()) || $experiment->operator_uid() == $SVC->authdb()->authName() ;
    $SVC->assert ($is_authorized ,
                  "not authorized to allocate run numbers for experiment id={$exper_id}") ;

    $run = $SVC->safe_assign ($experiment->generate_run() ,
                              "failed to allocate run for experiment id={$exper_id}") ;

    return array (
        'exper_id' => $experiment->id() ,
        'runnum'   => $run->num()) ;
}) ;

?>
