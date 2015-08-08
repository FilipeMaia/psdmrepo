<?php

/*
 * Update the Data Retention Policy parameters and return their latest state
 * 
 * AUTHORIZATION: data administrator
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    // Parameters which can be turned into LusiTime
    foreach (array('SHORT-TERM', 'MEDIUM-TERM') as $storage_class_name) {
        $param_name = 'CTIME' ;
        $str = $SVC->optional_str("{$storage_class_name}:{$param_name}", null) ;
        if (!is_null($str)) {
            $ctime = $str === '' ?
                $str :
                $SVC->safe_assign (
                    \LusiTime\LusiTime::parse($str) ,
                    "illegal value of parameter: {$storage_class_name}:{$param_name}") ;

            $SVC->configdb()->set_policy_param (
                $storage_class_name ,
                $param_name ,
                $ctime === '' ? $ctime : $ctime->toStringDay()) ;
        }
    }

    // Numeric parameters
    foreach (array('SHORT-TERM', 'MEDIUM-TERM') as $storage_class_name) {
        foreach (array('RETENTION', 'QUOTA') as $param_name) {
            $value = $SVC->optional_int("{$storage_class_name}:{$param_name}", null) ;
            if (!is_null($value)) {
                $SVC->configdb()->set_policy_param (
                    $storage_class_name ,
                    $param_name ,
                    "{$value}") ;
            }
        }
    }

    // The most up-to-date status of the parameters
    return array (
        'policy' => \DataPortal\Config::general_policy2array($SVC)
    ) ;
}) ;