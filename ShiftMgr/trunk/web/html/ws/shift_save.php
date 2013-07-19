<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;

\DataPortal\ServiceJSON::run_handler('POST', function($SVC) {

    $shift_id = $SVC->required_int ('shift_id') ;
    $begin    = $SVC->required_time('begin') ;
    $end      = $SVC->required_time('end') ;
    $area     = $SVC->required_JSON('area') ;
    $activity = $SVC->required_JSON('activity') ;
    $notes    = $SVC->required_str ('notes') ;

    $ival = new LusiInterval($begin, $end) ;

    $shifts = array (

        array (
            'id'       => $shift_id ,
            'begin'    => array('day' => $begin->toStringDay(), 'hm' => $begin->toStringHM(), 'hour' => $begin->hour(), 'minute' => $begin->minute(), 'full' => $begin->toStringShort()) ,
            'end'      => array('day' => $end  ->toStringDay(), 'hm' => $end  ->toStringHM(), 'hour' => $end  ->hour(), 'minute' => $end  ->minute(), 'full' => $end  ->toStringShort()) ,
            'duration'     => $ival->toStringHM() ,
            'duration_min' => $ival->toMinutes() ,
            'stopper'  => 0.0 ,
            'door'     => 0.0 ,
            'area'     => $area ,
            'activity' => $activity ,
            'notes'    => $notes ,
            'editor'   => $SVC->authdb()->authName() ,
            'modified' => LusiTime::now()->toStringShort()
        )
    ) ;
    $SVC->finish(array('shifts' => $shifts));
});

?>
