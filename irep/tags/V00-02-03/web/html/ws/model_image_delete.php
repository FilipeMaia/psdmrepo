<?php

/**
 * This service will delete the specified attachment image from a model and return an updated dictionary.
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

    $id = $SVC->required_int ('id') ;

    $attachment = $SVC->irep()->find_model_attachment_by_id($id) ;
    if (is_null($attachment)) $SVC->abort("no attachment found for ID: {$id}") ;

    $attachment->model()->delete_attachment($id) ;

    $SVC->finish(\Irep\IrepUtils::manufacturers2array($SVC->irep())) ;

}) ;

?>
