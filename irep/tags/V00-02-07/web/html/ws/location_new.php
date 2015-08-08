<?php

/**
 * This service will create a new location and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   <name>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $name = $SVC->required_str('name') ;

    $location = $SVC->irep()->find_location_by_name($name) ;
    if (!is_null($location)) $SVC->abort("the location already exists: {$name}") ;

    $SVC->irep()->add_location($name) ;

    $SVC->finish(\Irep\IrepUtils::locations2array($SVC->irep())) ;
}) ;

?>
