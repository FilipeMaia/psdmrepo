<?php


/**
 * Print a log file for the specified translation request
 * 
 * PARAMETERS:
 * 
 *   <id> <service>
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\Service::run_handler ('GET', function ($SVC) {

    $id      = $SVC->required_int ('id') ;
    $service = $SVC->required_enum('service' ,
                                   array('STANDARD', 'MONITORING') ,
                                   array('ignore_case' => true, 'convert' => 'toupper')) ;

    $req = $SVC->safe_assign ($SVC->ifacectrlws($service)->request_by_id($id) ,
                              "no request found for service='{$service}'::id={$id}") ;

    header('Content-type: text/html') ;
    header("Cache-Control: no-cache, must-revalidate") ; // HTTP/1.1
    header("Expires: Sat, 26 Jul 1997 05:00:00 GMT") ;   // Date in the past

    echo $SVC->ifacectrlws($service)->log($req->log_url) ;
}) ;

?>
