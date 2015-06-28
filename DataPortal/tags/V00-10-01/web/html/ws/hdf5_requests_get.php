<?php
/**
 * Return HDF5 translation requests for the experiment.
 * 
 * MANDATORY PARAMETERS:
 *
 *   <exper_id>
 *
 * OPTIONAL PARAMETERS OF THE FILTER:
 *
 *   <service>      - the service name
 *   <runs>        - a range of runs
 *   <status>      - a desired status for translation requests
 *   <show_files>  - a flag indicating if files should be shown as well for each run
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id      = $SVC->required_int ('exper_id') ;
    $service       = $SVC->optional_enum('service' ,
                                         array('STANDARD', 'MONITORING') ,
                                         'STANDARD' ,
                                         array('ignore_case' => true, 'convert' => 'toupper')) ;

    $range_of_runs = $SVC->optional_str ('runs', null) ;
    $status        = $SVC->optional_enum('status' ,
                                         array('FINISHED', 'FAILED', 'TRANSLATING', 'QUEUED', 'NOT-TRANSLATED') ,
                                         null ,
                                         array('ignore_case' => true, 'convert' => 'toupper')) ;

    return array (
        "requests" =>
            DataPortal\Translator1::get_requests (
                $SVC->ifacectrlws($service) ,
                $exper_id ,
                $range_of_runs ,
                $status
            )
    ) ;
}) ;
  
?>