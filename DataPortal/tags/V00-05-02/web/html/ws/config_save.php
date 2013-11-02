<?php

/**
 * This service will save a value of the specified application configuration parameter
 * in the database and return it as a JSON object.
 * 
 * Parameters:
 * 
 *   <application> <scope> <parameter> <value>
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $application = $SVC->required_str('application') ;
    $scope       = $SVC->required_str('scope') ;
    $parameter   = $SVC->required_str('parameter') ;
    $value       = $SVC->required_str('value') ;

    $obj = $SVC->configdb()->save_application_parameter($application, $scope, $parameter, $value) ;

    $SVC->finish (
        is_null($obj) ?
            array ('found' => 0) :
            array ('found' => 1, 'value' => $obj)) ;
}) ;

?>
