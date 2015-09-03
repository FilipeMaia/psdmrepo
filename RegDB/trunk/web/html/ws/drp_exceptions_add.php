<?php

/*
 * Allow Data Retention Policy exceptions for an experiment. Initialize
 * its policies with the default values of the general policy.
 * 
 * AUTHORIZATION:
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_name = $SVC->required_str ('exper_name') ;

    \DataPortal\Config::set_experiment_policy_exceptions (
        $SVC ,
        $SVC->safe_assign (
            $SVC->regdb()->find_experiment_by_name($exper_name) ,
            "no experiment found for name: {$exper_name}") ,
        \DataPortal\Config::general_policy2array($SVC)) ;
        
    return array (
        'experiments' => \DataPortal\Config::policy_exceptions2array($SVC)
    ) ;
}) ;
?>