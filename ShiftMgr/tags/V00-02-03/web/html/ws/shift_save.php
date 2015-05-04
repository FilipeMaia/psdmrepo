<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;

\DataPortal\ServiceJSON::run_handler('POST', function($SVC) {

    $shift_id   = $SVC->required_int ('shift_id') ;
    $begin      = $SVC->required_time('begin') ;
    $end        = $SVC->required_time('end') ;
    $type       = $SVC->required_str('type') ;
    $area       = $SVC->required_JSON('area') ;
    $allocation = $SVC->required_JSON('allocation') ;
    $notes      = $SVC->required_str ('notes') ;

    $shifts = array (
        $SVC->shiftmgr()->update_shift($shift_id, $begin, $end, $type, $notes, $area, $allocation)
    ) ;
    $SVC->finish(array('shifts' => \ShiftMgr\Utils::shifts2array($shifts))) ;
});

?>
