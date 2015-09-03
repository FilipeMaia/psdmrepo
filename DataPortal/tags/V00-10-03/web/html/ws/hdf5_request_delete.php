<?php

/**
 * This script will delete a pending translation request.
 *
 * PARAMETERS:
 * 
 *   <id>       - an identifier of the request
 *
 * OPTIONAL PARAMETERS OF THE FILTER:
 *
 *   <service>  - the service name
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id = $SVC->required_int ('id') ;

    $service = $SVC->optional_enum('service' ,
                                   array('STANDARD', 'MONITORING') ,
                                   'STANDARD' ,
                                   array('ignore_case' => true, 'convert' => 'toupper')) ;

    $SVC->ifacectrlws($service)->delete_request($id) ;
}) ;

?>
