<?php

/**
 * This is an example of hwo to use the JSON Web services framework
 * in the functional way. The sample code will demonstrate framework's
 * service for parsing script parameters.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish (array(
        'id'               => $SVC->required_int     ('id') ,
        'required_bool'    => $SVC->required_bool    ('required_bool') ,
        'optional_bool'    => $SVC->optional_bool    ('optional_bool', false) ,
        'optional_flag'    => $SVC->optional_flag    ('optional_flag') ,
        'required_str'     => $SVC->required_str     ('required_str') ,
        'optional_str'     => $SVC->optional_str     ('optional_str') ,
        'required_time_32' => $SVC->required_time_32 ('required_time_32')                  ->toStringShort() ,
        'optional_time_32' => $SVC->optional_time_32 ('optional_time_32', LusiTime::now()) ->toStringShort() ,
        'required_time_64' => $SVC->required_time_64 ('required_time_64')                  ->toStringShort() ,
        'optional_time_64' => $SVC->optional_time_64 ('optional_time_64', LusiTime::now()) ->toStringShort() ,
        'required_time'    => $SVC->required_time    ('required_time')                     ->toStringShort()
    )) ;
}) ;

?>
