<?php

/**
 * This service will update properties of an equipment and return a database entry information
 * for the updated item.
 * 
 * Parameters:
 * 
 *   <equipment_id> <status> <status2> <serial> <pc> <location_id> <custodian> <description> <comment>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $SVC->irep()->can_edit_inventory() or
        $SVC->abort('your account not authorized for the operation') ;

    $equipment_id = $SVC->required_int('equipment_id') ;
    $status       = $SVC->required_str('status') ;
    $status2      = $SVC->required_str('status2') ;
    $serial       = $SVC->required_str('serial') ;
    $pc           = $SVC->required_str('pc') ;
    $location_id  = $SVC->required_int('location_id') ;
    $custodian    = $SVC->required_str('custodian') ;
    $description  = $SVC->required_str('description') ;
    $comment      = $SVC->required_str('comment') ;

    $equipment = $SVC->irep()->find_equipment_by_id($equipment_id) ;
    if (is_null($equipment)) $SVC->abort("no equipment found for id: {$equipment_id}") ;

    $location = $SVC->irep()->find_location_by_id($location_id) ;
    if (is_null($location)) $SVC->abort("no location found for id: {$location_id}") ;

    $num_updates = 0 ;
    $properties2update = array () ;
    if ($equipment->status     () != $status          ) { $num_updates++; $properties2update['status']      = $status ; }
    if ($equipment->status2    () != $status2         ) { $num_updates++; $properties2update['status2']     = $status2 ; }
    if ($equipment->serial     () != $serial          ) { $num_updates++; $properties2update['serial']      = $serial ; }
    if ($equipment->description() != $description     ) { $num_updates++; $properties2update['description'] = $description ; }
    if ($equipment->pc         () != $pc              ) { $num_updates++; $properties2update['pc']          = $pc ; }
    if ($equipment->location   () != $location->name()) { $num_updates++; $properties2update['location']    = $location->name() ; }
    if ($equipment->custodian  () != $custodian       ) { $num_updates++; $properties2update['custodian']   = $custodian ; }
    if ($num_updates)
        $equipment = $SVC->irep()->update_equipment (
            $equipment_id ,
            $properties2update ,
            $comment
        ) ;

    $SVC->finish(\Irep\IrepUtils::equipment2array(array($equipment))) ;
}) ;

?>
