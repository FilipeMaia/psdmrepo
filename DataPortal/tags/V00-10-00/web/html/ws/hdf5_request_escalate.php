<?php

/**
 * This script will escalate the priority of a pending translation request.
 *
 * PARAMETERS:
 * 
 *   <exper_id> - an identifier of the experiment
 *   <id>       - an identifier of the request
 *
 * OPTIONAL PARAMETERS OF THE FILTER:
 *
 *   <service>  - the service name
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $id       = $SVC->required_int ('id') ;

    $service = $SVC->optional_enum('service' ,
                                   array('STANDARD', 'MONITORING') ,
                                   'STANDARD' ,
                                   array('ignore_case' => true, 'convert' => 'toupper')) ;

    $exper = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                "no such experiment exists for id={$exper_id}") ;

    /* Find the highest priority of the pending requests (if any).
     */
    $priority = 0 ;
    foreach ($SVC->ifacectrlws($service)->experiment_requests (
        $exper->instrument()->name() ,
        $exper->name()) as $req) {

    	if (($req->status == 'Initial_Entry') || ($req->status == 'Waiting_Translation')) {
            if ($req->priority > $priority) { $priority = $req->priority ; }
        }
    }
    $priority++ ;

    $request = $SVC->ifacectrlws($service)->set_request_priority($id, $priority) ;
    
    return array (
        'priority' => $request->priority
    ) ;
}) ;

?>
