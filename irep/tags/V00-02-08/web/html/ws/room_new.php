<?php

/**
 * This service will create a new room and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   <location_name> <room_name>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $location_name = $SVC->required_str('location_name') ;
    $room_name     = $SVC->required_str('room_name') ;

    $location = $SVC->irep()->find_location_by_name($location_name) ;
    if (is_null($location)) $SVC->abort("no such location exists: {$location_name}") ;

    $location->add_room($room_name) ;

    $SVC->finish(\Irep\IrepUtils::locations2array($SVC->irep())) ;
}) ;

?>
