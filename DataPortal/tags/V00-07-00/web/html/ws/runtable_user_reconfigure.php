<?php

/**
 * Update a configuration of the specified user table of an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $id       = $SVC->required_int ('id') ;
    $name     = $SVC->required_str ('name') ;
    $descr    = $SVC->required_str ('descr') ;
    $coldef   = $SVC->required_JSON('coldef') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $table = $experiment->find_run_table_by_id($id) ;
    if (!$table) $SVC->abort("no run table found for id={$id}") ;
    
    $table = $table->reconfigure (
        $name ,
        $descr ,
        $SVC->authdb()->authName() ,
        $coldef) ;

    $SVC->finish (array(
        'table_data' => $table->as_data()
    )) ;
}) ;

?>
