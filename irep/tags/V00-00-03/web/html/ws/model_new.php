<?php

/**
 * This service will create a new model and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   <manufacturer_name> <model_name>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $manufacturer_name = $SVC->required_str('manufacturer_name') ;
    $model_name        = $SVC->required_str('model_name') ;

    $manufacturer = $SVC->irep()->find_manufacturer_by_name($manufacturer_name) ;
    if (is_null($manufacturer)) $SVC->abort("no such manufacturer exists: {$manufacturer_name}") ;

    $manufacturer->add_model($model_name) ;

    $SVC->finish(\Irep\IrepUtils::manufacturers2array($SVC->irep())) ;
}) ;

?>
