<?php

/**
 * This service will delete the specified SLACid range and return an updated object.
 * 
 * Parameters:
 * 
 *   <range_id>
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $range_id = $SVC->required_int('range_id') ;
    $SVC->irep()->delete_slacid_range($range_id) ;
    $SVC->finish(array('range' => $SVC->irep()->slacid_ranges()));
}) ;


?>