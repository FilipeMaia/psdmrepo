<?php

/**
 * This service will return history events in the specified scope.
 * 
 * Parameters:
 * 
 *   <equipment_id>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $equipment_id = $SVC->required_int('equipment_id') ;

    $equipment = $SVC->irep()->find_equipment_by_id($equipment_id) ;
    if (is_null($equipment)) $SVC->abort("no equipment found for ID: {$equipment_id}") ;

    $SVC->finish(array (
        'scope' => 'equipment' ,
        'equipment_id' => $equipment_id ,
        'history' => \Irep\IrepUtils::equipment_history2array($equipment))) ;
}) ;

?>
