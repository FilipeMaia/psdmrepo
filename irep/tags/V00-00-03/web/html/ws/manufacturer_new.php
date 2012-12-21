<?php

/**
 * This service will create a new manufacturer and return an updated dictionary.
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

    $manufacturer = $SVC->irep()->find_manufacturer_by_name($name) ;
    if (!is_null($manufacturer)) $SVC->abort("the manufacturer already exists: {$name}") ;

    $SVC->irep()->add_manufacturer($name) ;

    $SVC->finish(\Irep\IrepUtils::manufacturers2array($SVC->irep())) ;
}) ;

?>
