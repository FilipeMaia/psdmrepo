<?php

/*
 * Start the specified run for an experiment.
 * 
 * The run record will get permanently stored in the LogBook database. The service will return
 * the information about the run in a JSON object
 * 
 * AUTHORIZATION: being able to post messages for the experiment's e-log,
 * or 
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $num      = $SVC->required_int ('num') ;
    $type     = $SVC->required_str ('type') ;

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $is_authorized = $SVC->logbookauth()->canPostNewMessages($experiment->id()) ||
                     $experiment->regdb_experiment()->operator_uid() == $SVC->authdb()->authName() ;
    $SVC->assert ($is_authorized ,
                  "not authorized to begin/end runs for experiment id={$exper_id}") ;

    $run = $SVC->safe_assign ($experiment->create_run($num, $type, LusiTime\LusiTime::now()) ,
                              "failed to start run {$num} at experiment id={$exper_id}") ;

    return $run->to_array() ;
}) ;
?>
