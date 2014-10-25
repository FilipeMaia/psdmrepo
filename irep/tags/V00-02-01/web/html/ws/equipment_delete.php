<?php

/**
 * This service will delete the specified equipment item from the database
 * for the new item.
 * 
 * Parameters:
 * 
 *   <equipment_id>
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->can_edit_inventory() or
        $SVC->abort('your account not authorized for the operation') ;

    $SVC->irep()->delete_equipment($SVC->required_int('equipment_id')) ;
    $SVC->finish() ;
}) ;

?>
