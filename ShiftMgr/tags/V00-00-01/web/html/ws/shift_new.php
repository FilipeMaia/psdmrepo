<?php

/**
 * This service will open a new shift at the specified instrument.
 * 
 * PARAMETERS:
 * 
 *   instr=<name>

 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instrument_name = $SVC->required_str('instr') ;

    if (!$SVC->shiftmgr()->is_manager($instrument_name))
        $SVC->abort("your account is not authorized to manage shifts for instrument: {$instrument_name}") ;

    $SVC->shiftmgr()->new_shift($instrument_name) ;

    $SVC->finish (array (
        'shifts' => \ShiftMgr\ShiftMgrUtils::shifts2array ($SVC->shiftmgr()->shifts($instrument_name))
    )) ;
}) ;

?>
