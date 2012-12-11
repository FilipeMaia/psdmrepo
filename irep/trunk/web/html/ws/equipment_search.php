<?php

/**
 * This service will search equipment and return a database entry information
 * for the new item. The search can be exact if a specific identifier which will
 * uniquely identify the equipment item such as: equipment id, or a SLAC id
 * or a Property Control number. Otherwise the search will be conducted based on
 * partial values of equipment properties. If no paramaters are provided then
 * a list of all known items will be returned.
 *
 * Parameters:
 * 
 *   Exact search parameters:
 *
 *      <equipment_id> || <slacid> || || <slacid_range_id> || <pc> || <status_id> || <status2_id>
 *
 *   Partial search parameters:
 *
 *      [<status> [<status2>]]
 *      [<manufacturer> || <manufacturer_id>]
 *      [<model>        || <model_id>]
 *      [<serial>]
 *      [<location> || <location_id>]
 *      [<room>     || <room_id>]
 *      [<custodian>]
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    // Check for exact search parameters and trigger the search if
    // any is found.

    $equipment_id = $SVC->optional_int('equipment_id', null) ;
    if (!is_null($equipment_id)) {
        $equipment = $SVC->irep()->find_equipment_by_id($equipment_id) ;
        if (is_null($equipment)) $SVC->abort("no equipment for id: {$equipment_id}") ;
        $SVC->finish(\Irep\IrepUtils::equipment2array(array($equipment))) ;
    }

    $slacid = $SVC->optional_int('slacid', null) ;
    if (!is_null($slacid)) {
        $equipment = $SVC->irep()->find_equipment_by_slacid($slacid) ;
        if (is_null($equipment)) $SVC->abort("no equipment for SLACid: {$slacid}") ;
        $SVC->finish(\Irep\IrepUtils::equipment2array(array($equipment))) ;
    }

    $slacid_range_id = $SVC->optional_int('slacid_range_id', null) ;
    if (!is_null($slacid_range_id)) {
        $SVC->finish (\Irep\IrepUtils::equipment2array (
            $SVC->irep()->find_equipment_by_slacid_range($slacid_range_id))) ;
    }

    $pc = $SVC->optional_str('pc', '') ;
    if ($pc != '') {
        $equipment = $SVC->irep()->find_equipment_by_pc($pc) ;
        if (is_null($equipment)) $SVC->abort("no equipment for Property Control (PC) number: {$pc}") ;
        $SVC->finish(\Irep\IrepUtils::equipment2array(array($equipment))) ;
    }

    $status_id = $SVC->optional_int('status_id', null) ;
    if (!is_null($status_id)) {
        $SVC->finish (\Irep\IrepUtils::equipment2array (
            $SVC->irep()->find_equipment_by_status_id($status_id))) ;
    }
    $status2_id = $SVC->optional_int('status2_id', null) ;
    if (!is_null($status2_id)) {
        $SVC->finish (\Irep\IrepUtils::equipment2array (
            $SVC->irep()->find_equipment_by_status2_id($status2_id))) ;
    }

    // Harvest optional parameters of the partial search. Note that for some of
    // those parameters we have alternatives such as identifiers.

    $status            = $SVC->optional_str('status', '') ;
    $status2           = $SVC->optional_str('status2', '') ;
    $manufacturer_id   = $SVC->optional_int('manufacturer_id', 0) ;
    $manufacturer_name = $SVC->optional_str('manufacturer', '') ;
    if ($manufacturer_id) {
        if ($manufacturer_name == '') {
            $manufacturer = $SVC->irep()->find_manufacturer_by_id($manufacturer_id) ;
            if (is_null($manufacturer)) $SVC->abort("no manufacturer found for id: {$manufacturer_id}") ;
            $manufacturer_name = $manufacturer->name() ;
        } else {
            $SVC->abort("conflicting parameters for a manufacturer") ;
        }
    }
    $model_id   = $SVC->optional_int('model_id', 0) ;
    $model_name = $SVC->optional_str('model', '') ;
    if ($model_id) {
        if ($model_name == '') {
            $model = $SVC->irep()->find_model_by_id($model_id) ;
            if (is_null($model)) $SVC->abort("no model found for id: {$model_id}") ;
            $model_name = $model->name() ;
        } else {
            $SVC->abort("conflicting parameters for a model") ;
        }
    }
    $serial        = $SVC->optional_str('serial', '') ;
    $location_id   = $SVC->optional_int('location_id', 0) ;
    $location_name = $SVC->optional_str('location', '') ;
    if ($location_id) {
        if ($location_name == '') {
            $location = $SVC->irep()->find_location_by_id($location_id) ;
            if (is_null($location)) $SVC->abort("no location found for id: {$location_id}") ;
            $location_name = $location->name() ;
        } else {
            $SVC->abort("conflicting parameters for a location") ;
        }
    }
    $room_id   = $SVC->optional_int('room_id', 0) ;
    $room_name = $SVC->optional_str('room', '') ;
    if ($room_id) {
        if ($room_name == '') {
            $room = $SVC->irep()->find_room_by_id($room_id) ;
            if (is_null($room)) $SVC->abort("no room found for id: {$room_id}") ;
            $room_name = $room->name() ;
        } else {
            $SVC->abort("conflicting parameters for a room") ;
        }
    }
    $custodian = $SVC->optional_str('custodian', '') ;

    $SVC->finish(\Irep\IrepUtils::equipment2array (
        $SVC->irep()->search_equipment (
            $status ,
            $status2 ,
            $manufacturer_name ,
            $model_name ,
            $serial ,
            $location_name ,
            $custodian
        )
    )) ;
}) ;

?>
