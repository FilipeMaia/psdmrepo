<?php

/**
 * Return the information about DAQ counters for runs of the experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id       = $SVC->required_int('exper_id') ;
    $from_runnum    = $SVC->optional_int('from_run', 0) ;
    $through_runnum = $SVC->optional_int('through_run', 0) ;

    if (($from_runnum && $through_runnum) && ($from_runnum > $through_runnum))
        $SVC->abort("illegal range of runs: last run must be equal or greater then the first one") ;

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $SVC->finish (LogBook\LogBookUtils::get_daq_counters($experiment, $from_runnum, $through_runnum)) ;
}) ;

?>
