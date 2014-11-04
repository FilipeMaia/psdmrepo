<?php

/**
 * This service will updated properties of the specified manufacturer and return
 * an updated dictionary.
 * 
 * Parameters:
 * 
 *   <id> [<description>]
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $id          = $SVC->required_int('id') ;
    $description = $SVC->optional_str('description', null) ;

    $manufacturer = $SVC->irep()->find_manufacturer_by_id($id) ;
    if (is_null($manufacturer))
        $SVC->abort("no manufacturer exists for id: {$id}") ;

    if (!is_null($description)) $manufacturer->update_description($description) ;

    $SVC->finish(\Irep\IrepUtils::manufacturers2array($SVC->irep())) ;
}) ;

?>
