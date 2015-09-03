<?php

/**
 * This service will check if the specified SLAC ID is in use and if so return
 * a descriptor.
 * 
 * Parameters:
 * 
 *   <slacid>
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $slacid = $SVC->required_int('slacid') ;
    $slacid_descr = $SVC->irep()->find_slacid($slacid) ;
    if (is_null($slacid_descr))
        $SVC->finish(array (
            'slacid' => array (
                'in_use'   => 0 ,
                'is_valid' => is_null($SVC->irep()->find_slacid_range_for($slacid)) ? 0 : 1))) ;
    else
        $SVC->finish(array (
            'slacid' => array (
                'in_use' => 1 ,
                'descr'  => $slacid_descr )));
}) ;

?>