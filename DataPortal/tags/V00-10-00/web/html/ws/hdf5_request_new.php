<?php

/**
 * This script will submit a new translation request.
 *
 * PARAMETERS:
 * 
 *   <exper_id> - experiment id
 *   <runnum>   - run number
 *   <service>  - the service name
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $runnum   = $SVC->required_int ('runnum') ;

    $service = $SVC->required_enum('service' ,
                                   array('STANDARD', 'MONITORING') ,
                                   array('ignore_case' => true, 'convert' => 'toupper')) ;

    $exper = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                "no such experiment exists for id={$exper_id}") ;

    $run = $SVC->safe_assign ($exper->find_run_by_num($runnum) ,
                              "no run {$runnum} exists experiment id={$exper_id}" );    
    return array (
        'request' => $SVC->ifacectrlws($service)->create_request (
            $exper->instrument()->name() ,
            $exper->name() ,
            $runnum)
    ) ;
}) ;

?>
