<?php

/**
 * This service will locate the specified application configuration parameter
 * in the database and return it as a JSON object.
 * 
 * Parameters:
 * 
 *   <application> <scope> <parameter>
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $application = $SVC->required_str('application') ;
    $scope       = $SVC->required_str('scope') ;
    $parameter   = $SVC->required_str('parameter') ;

    $obj = $SVC->configdb()->find_application_parameter($application, $scope, $parameter) ;

    $SVC->finish (
        is_null($obj) ?
            array ('found' => 0) :
            array ('found' => 1, 'value' => $obj)) ;
}) ;

?>
