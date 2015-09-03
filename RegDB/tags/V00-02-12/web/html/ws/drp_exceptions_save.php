<?php

/*
 * Save experiment's exceptions for the Data Retention Policy parameters
 * 
 * AUTHORIZATION:
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int ('exper_id') ;
    $policy   = $SVC->required_json('policy') ;

    \DataPortal\Config::set_experiment_policy_exceptions (
        $SVC ,
        $SVC->safe_assign (
            $SVC->regdb()->find_experiment_by_id($exper_id) ,
            "no experiment found for id: {$exper_id}") ,
        $policy) ;

    return array (
        'experiments' => \DataPortal\Config::policy_exceptions2array($SVC)
    ) ;
}) ;
?>