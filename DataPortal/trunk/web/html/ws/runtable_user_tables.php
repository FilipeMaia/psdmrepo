<?php

/**
 * Return defnitions of user tables for an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $table_data = array() ;

    foreach ($experiment->run_tables() as $table) {
        $table_data[$table->id()] = $table->as_data() ;
    }
    $run = $experiment->find_last_run() ;
    $SVC->finish (array(
        'table_data'  => $table_data ,
        'last_runnum' => $run ? $run->num() : 0
    )) ;
}) ;

?>
