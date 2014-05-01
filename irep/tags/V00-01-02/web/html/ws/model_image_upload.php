<?php

/**
 * This service will upload an image of the specified model and return an updated dictionary.
 * 
 * Parameters:
 * 
 *   <model_id> <file2attach>
 *
 * Note, that the resulting JSON object is warpped into the textarea. See detail
 * in the implementation of class ServiceJSON.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $SVC->irep()->has_dict_priv() or
        $SVC->abort('your account not authorized for the operation') ;

    $model_id    = $SVC->required_int ('model_id') ;
    $file2attach = $SVC->required_file() ;

    $model = $SVC->irep()->find_model_by_id($model_id) ;
    if (is_null($model)) $SVC->abort("no model found for ID: {$model_id}") ;

    $attachment = $model->add_default_attachment($file2attach, $SVC->authdb()->authName()) ;

    $SVC->finish(\Irep\IrepUtils::manufacturers2array($SVC->irep())) ;

} , array ('wrap_in_textarea' => True)) ;

?>
