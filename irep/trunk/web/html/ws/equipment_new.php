<?php

/**
 * This service will add a new equipment and return a database entry information
 * for the new item.
 * 
 * Parameters:
 * 
 *   <model_id> <serial> <pc> <slacid> [<location_id>] <custodian> <description>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $SVC->irep()->can_edit_inventory() or
        $SVC->abort('your account not authorized for the operation') ;

    $model_id    = $SVC->required_int('model_id') ;
    $serial      = $SVC->required_str('serial') ;
    $pc          = $SVC->required_str('pc') ;
    $slacid      = $SVC->required_str('slacid') ;
    $location_id = $SVC->optional_int('location_id', 0) ;
    $custodian   = $SVC->required_str('custodian') ;
    $description = $SVC->required_str('description') ;

    $model = $SVC->irep()->find_model_by_id($model_id) ;
    if (is_null($model)) $SVC->abort("no model found for id: {$model_id}") ;

    $location_name = '' ;
    if ($location_id) {
        $location = $SVC->irep()->find_location_by_id($location_id) ;
        if (is_null($location)) $SVC->abort("no location found for id: {$location_id}") ;
        $location_name = $location->name() ;
    }

    $equipment = $SVC->irep()->add_equipment (
        $model->manufacturer()->name() ,
        $model->name() ,
        $serial ,
        $description ,
        $pc ,
        $slacid ,
        $location_name ,
        $custodian
    ) ;

    $SVC->finish(\Irep\IrepUtils::equipment2array(array($equipment))) ;
}) ;

?>
