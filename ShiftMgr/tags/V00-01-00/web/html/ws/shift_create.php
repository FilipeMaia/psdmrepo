<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;
    $begin_time = $SVC->required_time('begin_time') ;
    $end_time   = $SVC->required_time('end_time') ;

    $min_begin_time = \ShiftMgr\ShiftMgr::min_begin_time() ;
    if ($begin_time->less($min_begin_time))
        $SVC->abort("The begin time can't be older than: {$min_begin_time->toStringShort()}") ;

    if ($SVC->shiftmgr()->find_shift_by_begin_time($instr_name, $begin_time))
        $SVC->abort("There is another shift which has exactly the same begin time as the requested one") ;

    $shifts = array (
        $SVC->shiftmgr()->create_shift($instr_name, $begin_time, $end_time)
    ) ;
    $SVC->finish(array('shifts' => \ShiftMgr\Utils::shifts2array($shifts))) ;
});

?>
