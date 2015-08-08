<?php

/**
 * Return a dictionary of column definitions for user tables
 * available for an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $name     = $SVC->required_str ('name') ;
    $descr    = $SVC->required_str ('descr') ;
    $coldef   = $SVC->required_JSON('coldef') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $table = $experiment->create_run_table (
        $name ,
        $descr ,
        $SVC->authdb()->authName() ,
        $coldef) ;

    $SVC->finish (array(
        'table_data' => $table->as_data()
    )) ;
}) ;

?>
