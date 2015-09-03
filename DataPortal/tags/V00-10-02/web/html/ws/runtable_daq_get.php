<?php

/**
 * Return the information about DAQ detectors of the experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $section  = $SVC->required_enum('section' ,
                                    array('detectors', 'totals') ,
                                    array('ignore_case' => true, 'convert' => 'tolower')) ;

    $from_runnum    = $SVC->optional_int('from_run',    0) ;
    $through_runnum = $SVC->optional_int('through_run', 0) ;

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;
    if ($from_runnum < 0) {
        $last_run = $experiment->find_last_run() ;
        if ($last_run) {
            $runnum = $last_run->num() ;
            $from_runnum = abs($from_runnum) <= $runnum ? $runnum - abs($from_runnum) : 0 ;
        }
    }
    $SVC->assert (!(($from_runnum && $through_runnum) &&
                    ($from_runnum >  $through_runnum)) ,
                  "illegal range of runs: {$from_runnum}-{$through_runnum}") ;
    
    switch ($section) {
        case 'detectors' : 
            return LogBook\LogBookUtils::get_daq_detectors_new (
                $experiment ,
                $from_runnum ,
                $through_runnum) ;
        case 'totals' : 
            return LogBook\LogBookUtils::get_daq_detector_totals (
                $experiment ,
                $from_runnum ,
                $through_runnum) ;
    }
    $SVC->abort('implementation error') ;
}) ;

?>
