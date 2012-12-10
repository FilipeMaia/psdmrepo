<?php

/**
 * This service will update parameters of the specified SLACid range
 * allocation and return an updated object.
 * 
 * Parameters:
 * 
 *   <ranges>
 * 
 * Syntax:
 * 
 *   <ranges> := [<range>[,<ranges>]
 *   <range>  := <id>:<first>:<last>
 *
 * Where:
 * 
 *   <id>    : a numeric identifier of a range. If 0 then it's a new range creation
 *             request. Otherwise an existing range needs to be updated.
 *   <first> : the first number in the range
 *   <last>  : the last number in the range
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $ranges = $SVC->required_str('ranges') ;
    foreach (explode(',', $ranges) as $range) {
        $id_first_last = explode(':', $range) ;
        $range_id = intval($id_first_last[0]) ;
        $first    = intval($id_first_last[1]) ;
        $last     = intval($id_first_last[2]) ;
        if (!$first && !$last) continue ;
        if ($last <= $first)
            $SVC->abort("illegal range: last number {$last} must be greater than the first one {$first}") ;
        if (!$range_id) {
            $SVC->irep()->add_slacid_range($first, $last) ;
        } else {
            $SVC->irep()->update_slacid_range($range_id, $first, $last) ;
            }
    }
    $SVC->finish(array('range' => $SVC->irep()->slacid_ranges()));
}) ;

?>