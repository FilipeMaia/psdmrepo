<?php

/**
 * This service will update properties of an equipment and return a database entry information
 * for the updated item.
 * 
 * Parameters:
 * 
 *   <equipment_id> <status> <status2> <serial> <pc> <location_id> <room_id> <rack> <elevation>
 *   <custodian> <description> <comment> <tags2add> <tags2remove>
 *
 * Where:
 * 
 *   <tags2add> is a JSON array with names of tags to add
 *   <tags2remove> is a JSON array with numeric identifiers of tags to delete
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
    $location_id  = $SVC->optional_int('location_id', 0) ;
    $room_id      = $SVC->optional_int('room_id', 0) ;
    $rack         = $SVC->optional_str('rack', '') ;
    $elevation    = $SVC->optional_str('elevation', '') ;
    $custodian    = $SVC->required_str('custodian') ;
    $description  = $SVC->required_str('description') ;
    $comment      = $SVC->required_str('comment') ;

    $tags2add           = $SVC->required_JSON('tags2add') ;
    $tags2remove        = $SVC->required_JSON('tags2remove') ;
    $attachments2remove = $SVC->required_JSON('attachments2remove') ;

    $equipment = $SVC->irep()->find_equipment_by_id($equipment_id) ;
    if (is_null($equipment)) $SVC->abort("no equipment found for id: {$equipment_id}") ;

    $location_name = '' ;
    if ($location_id) {
        $location = $SVC->irep()->find_location_by_id($location_id) ;
        if (is_null($location)) $SVC->abort("no location found for id: {$location_id}") ;
        $location_name =  $location->name() ;
    }
    $room_name = '' ;
    if ($room_id) {
        $room = $SVC->irep()->find_room_by_id($room_id) ;
        if (is_null($room)) $SVC->abort("no room found for id: {$room_id}") ;
        $room_name =  $room->name() ;
    }

    $num_updates = 0 ;
    $properties2update = array () ;
    if ($equipment->status     () != $status       ) { $num_updates++; $properties2update['status']      = $status ; }
    if ($equipment->status2    () != $status2      ) { $num_updates++; $properties2update['status2']     = $status2 ; }
    if ($equipment->serial     () != $serial       ) { $num_updates++; $properties2update['serial']      = $serial ; }
    if ($equipment->description() != $description  ) { $num_updates++; $properties2update['description'] = $description ; }
    if ($equipment->pc         () != $pc           ) { $num_updates++; $properties2update['pc']          = $pc ; }
    if ($equipment->location   () != $location_name) { $num_updates++; $properties2update['location']    = $location_name ; }
    if ($equipment->room       () != $room_name    ) { $num_updates++; $properties2update['room']        = $room_name ; }
    if ($equipment->rack       () != $rack         ) { $num_updates++; $properties2update['rack']        = $rack ; }
    if ($equipment->elevation  () != $elevation    ) { $num_updates++; $properties2update['elevation']   = $elevation ; }
    if ($equipment->custodian  () != $custodian    ) { $num_updates++; $properties2update['custodian']   = $custodian ; }
    if ($num_updates) {
        $equipment = $SVC->irep()->update_equipment (
            $equipment_id ,
            $properties2update ,
            $comment
        ) ;
    }
    foreach ($tags2add           as $tag_name)      $equipment->add_tag          ($tag_name) ;
    foreach ($tags2remove        as $tag_id)        $equipment->delete_tag_by_id ($tag_id) ;
    foreach ($attachments2remove as $attachment_id) $equipment->delete_attachment($attachment_id) ;

    $SVC->finish(\Irep\IrepUtils::equipment2array(array($equipment))) ;
}) ;

?>
