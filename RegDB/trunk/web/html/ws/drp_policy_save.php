<?php

/*
 * Update the Data Retention Policy parameters and return their latest state
 * 
 * AUTHORIZATION: data administrator
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime ;

function parse_parameter ($str, $default_value) {
    return (!isset($str) || is_null($str)) ?
        $default_value :
        (is_int($default_value) ?
            intval($str) :
            $str) ;
}
DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    // Parameters which can be turned into LusiTime
    foreach (array('SHORT-TERM', 'MEDIUM-TERM') as $storage_class_name) {
        $param_name = 'CTIME' ;
        $str = $SVC->optional_str("{$storage_class_name}:{$param_name}", null) ;
        if (!is_null($str)) {
            $ctime = $str === '' ?
                $str :
                $SVC->safe_assign (
                    LusiTime::parse($str) ,
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
        'policy' => array (
            'SHORT-TERM'  => array (
                'ctime'     => parse_parameter($SVC->configdb()->get_policy_param('SHORT-TERM',  'CTIME'),    '') ,
                'retention' => parse_parameter($SVC->configdb()->get_policy_param('SHORT-TERM',  'RETENTION'), 0)) ,
            'MEDIUM-TERM' => array (
                'ctime'     => parse_parameter($SVC->configdb()->get_policy_param('MEDIUM-TERM', 'CTIME'),    '') ,
                'retention' => parse_parameter($SVC->configdb()->get_policy_param('MEDIUM-TERM', 'RETENTION'), 0) ,
                'quota'     => parse_parameter($SVC->configdb()->get_policy_param('MEDIUM-TERM', 'QUOTA'),     0)) ,
        )
    ) ;
}) ;