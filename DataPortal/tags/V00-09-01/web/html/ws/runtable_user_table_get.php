<?php

/**
 * Return values of parameters found in the specified user table
 * for a range of runs of an  experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;


DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id       = $SVC->required_int('exper_id') ;
    $table_id       = $SVC->required_int('table_id') ;
    $from_runnum    = $SVC->optional_int('from_run', 0) ;
    $through_runnum = $SVC->optional_int('through_run', 0) ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$xper_id}") ;

    $last_run = $experiment->find_last_run() ;

    if ($from_runnum < 0) {
        if ($last_run) {
            $runnum = $last_run->num() ;
            $from_runnum = abs($from_runnum) <= $runnum ? $runnum - abs($from_runnum) : 0 ;
        }
    }
    
    if (($from_runnum && $through_runnum) && ($from_runnum > $through_runnum))
        $SVC->abort("illegal range of runs: make sure the second run is equal or greater then the first one") ;

    $runs = array() ;
    $run2id = array() ;

    $table = $experiment->find_run_table_by_id($table_id) ;
    if (!$table) $SVC->abort("no run table found for id: {$table_id}") ;

    $columns_cache = $table->columns() ;    // for the sake of optimization

    foreach ($experiment->runs() as $run) {

        $runnum = $run->num() ;
        if ($from_runnum    && ($runnum < $from_runnum))    continue ;
        if ($through_runnum && ($runnum > $through_runnum)) continue ;

        $runs  [$runnum] = $table->row($run, $columns_cache) ;
        $run2id[$runnum] = $run->id() ;
    }

    $SVC->finish (array(
        'info'   => array (
            'modified_time' => $table->modified_time()->toStringShort() ,
            'modified_uid'  => $table->modified_uid()
        ) ,
        'runs'     => $runs ,
        'run2id'   => $run2id ,
        'last_run' => $last_run ? $last_run->num() : 0
    )) ;
}) ;

?>
