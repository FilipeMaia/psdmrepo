<?php

/*
 * Return Data Retention Policy exceptions. If a particular experiment
 * identifier is provided via an optional parameter then a scope of
 * the operation will be limited to that experiment only. Otherwise
 * all experiments having exceptions will be reportd.
 * 
 * PARAMETERS:
 * 
 *   [<exper_id>]
 * 
 * AUTHORIZATION: not required
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {    
    return array (
        'experiments' => \DataPortal\Config::policy_exceptions2array (
            $SVC ,
            $SVC->optional_int('exper_id', null))
    ) ;
}) ;
?>