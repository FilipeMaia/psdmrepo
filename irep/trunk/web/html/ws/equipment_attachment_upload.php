<?php

/**
 * This service will upload attachments for the specified equipment and return
 * an updated equipment object.
 * 
 * Parameters:
 * 
 *   <equipment_id> [<file2attach> [<file2attach> ...]]
 *
 * Note, that the resulting JSON object is warpped into the textarea. See detail
 * in the implementation of class ServiceJSON.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $SVC->irep()->can_edit_inventory() or
        $SVC->abort('your account not authorized for the operation') ;

    $equipment_id = $SVC->required_int  ('equipment_id') ;
    $files        = $SVC->optional_files() ;

    $equipment = $SVC->irep()->find_equipment_by_id($equipment_id) ;
    if (is_null($equipment)) $SVC->abort("no equipment found for ID: {$equipment_id}") ;

    foreach ($files as $file)
        $equipment->add_attachment($file, $SVC->authdb()->authName()) ;

    $SVC->finish (\Irep\IrepUtils::equipment2array (
        array (
            $SVC->irep()->find_equipment_by_id($equipment_id))
        )
    ) ;

} , array ('wrap_in_textarea' => True)) ;

?>
