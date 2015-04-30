<?php

/*
 * Return Data Retention Policy parameters
 * 
 * AUTHORIZATION: not required
 */
require_once 'dataportal/dataportal.inc.php' ;

function parse_parameter ($str, $default_value) {
    return (!isset($str) || is_null($str)) ?
        $default_value :
        (is_int($default_value) ?
            intval($str) :
            $str) ;
}
DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
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
?>