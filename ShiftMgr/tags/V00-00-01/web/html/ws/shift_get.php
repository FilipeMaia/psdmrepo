<?php

/**
 * This service will a list of shifts matching the specified criteria.
 * 
 * PARAMETERS:
 * 
 *   [ instr=[<name>] [last] ]
 * 
 *    instr=[<name>]
 *
 *      - an optional parameter to narrow a scope of the operation
 *      to an instrument passed as a value of the parameter.
 *      All instruments will be assumed if the parameter is not
 *      used or it has an empty value.
 * 
 *    last
 * 
 *      - an optional flag which can follow a  non-empty instrument
 *      to indicate if the last shift of that instrument is requested.
 *      Note, that this parameter will be ignored if no instrument is
 *      provided.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instrument_name = $SVC->optional_str('instr', '') ;
    $last_shift_only = $SVC->optional_bool('last', false) ;
    
    $shifts = array() ;
    if (($instrument_name != '') && $last_shift_only) {
        $last_shift = $SVC->shiftmgr()->last_shift($instrument_name) ;
        if (!is_null($last_shift)) array_push($shifts, $last_shift) ;
    } else {
        $shifts = $SVC->shiftmgr()->shifts($instrument_name) ;
    }       
    $SVC->finish ( array (
        'shifts' => \ShiftMgr\ShiftMgrUtils::shifts2array($shifts))
    ) ;
}) ;

?>
