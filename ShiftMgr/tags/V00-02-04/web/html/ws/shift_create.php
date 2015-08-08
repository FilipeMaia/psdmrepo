<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use ShiftMgr\ShiftMgr ;
use LusiTime\LusiTime ;

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;
    $begin_time = $SVC->required_time('begin_time') ;
    $end_time   = $SVC->required_time('end_time') ;
    $type       = $SVC->required_str('type') ;

    $min_begin_time = ShiftMgr::min_begin_time() ;
    if ($begin_time->less($min_begin_time))
        $SVC->abort("The begin time can't be older than: {$min_begin_time->toStringShort()}") ;

    if ($SVC->shiftmgr()->find_shift_by_begin_time($instr_name, $begin_time))
        $SVC->abort("There is another shift which has exactly the same begin time as the requested one") ;

    if (!ShiftMgr::is_valid_shift_type($type)) $SVC->abort("Unknown shift type: {$type}") ;

    $shift = $SVC->shiftmgr()->create_shift($instr_name, $begin_time, $end_time, $type) ;
    $shifts = array($shift) ;

    $SVC->finish(array('shifts' => \ShiftMgr\Utils::shifts2array($shifts))) ;
});

?>
