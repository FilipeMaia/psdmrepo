<?php

/**
 * Delete specified user table from the database
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;
    $table_id = $SVC->required_int('table_id') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $table = $experiment->find_run_table_by_id($table_id) ;
    if (!$table) $SVC->abort("no run table found for id: {$table_id}") ;

    $experiment->delete_run_table_by_id($table_id) ;

    $SVC->finish() ;
}) ;

?>
