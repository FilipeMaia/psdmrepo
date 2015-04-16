<?php

/**
 * Update cells at specified user table
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $table_id = $SVC->required_int ('table_id') ;
    $cells    = $SVC->required_JSON('cells') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $table = $experiment->find_run_table_by_id($table_id) ;
    if (!$table) $SVC->abort("no run table found for id: {$table_id}") ;

    foreach ($cells as $cell)
        $table->update($cell->run_id, $cell->coldef_id, $cell->value, $SVC->authdb()->authName()) ;

    $table = $experiment->find_run_table_by_id($table->id()) ;

    $SVC->finish(array(
        'info' => array (
            'modified_time' => $table->modified_time()->toStringShort() ,
            'modified_uid'  => $table->modified_uid()
        )
    )) ;
}) ;

?>
