<?php

/**
 * This service will delete the specified model and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   <id>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $id = $SVC->required_int('id') ;

    if ($SVC->irep()->find_model_by_id($id))
        $SVC->irep()->delete_model_by_id($id) ;

    $SVC->finish(\Irep\IrepUtils::manufacturers2array($SVC->irep())) ;
}) ;

?>
