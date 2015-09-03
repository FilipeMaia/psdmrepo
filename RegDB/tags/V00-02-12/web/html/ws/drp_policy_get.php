<?php

/*
 * Return Data Retention Policy parameters
 * 
 * AUTHORIZATION: not required
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    return array (
        'policy' => \DataPortal\Config::general_policy2array($SVC)
    ) ;
}) ;
?>