<?php

namespace ShiftMgr;

require_once( 'shiftmgr.inc.php' );
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiInterval ;

/**
 * Class ShiftMgrUtils is a utility class accomodating a collection of
 * functions used by Web services.
 *
 * @author gapon
 */
class Utils {

    /**
     * Convert an input array of objects of class ShiftMgrShift into a portable array
     * of plain objects which can be serialized into JSON format.
     *
     * @param array $shifts
     * @return array
     */
    public static function shifts2array($shifts) {
        $result = array() ;
        foreach ($shifts as $s) {
            $begin = $s->begin_time() ;
            $end   = $s->end_time() ;
            $ival  = new LusiInterval($begin, $end) ;

            array_push (
                $result ,
                array (
                    'id'           => $s->id() ,
                    'begin'        => array('day' => $begin->toStringDay(), 'hm' => $begin->toStringHM(), 'hour' => $begin->hour(), 'minute' => $begin->minute(), 'full' => $begin->toStringShort()) ,
                    'end'          => array('day' => $end  ->toStringDay(), 'hm' => $end  ->toStringHM(), 'hour' => $end  ->hour(), 'minute' => $end  ->minute(), 'full' => $end  ->toStringShort()) ,
                    'duration'     => $ival->toStringHM() ,
                    'duration_min' => $ival->toMinutes() ,
                    'stopper_min'  => $s->stopper() ,
                    'door_min'     => $s->door() ,
                    'area'         => $s->areas() ,
                    'allocation'   => $s->allocations() ,
                    'notes'        => $s->notes() ,
                    'editor'       => $s->modified_uid()  ? $s->modified_uid() : '' ,
                    'modified'     => $s->modified_time() ? $s->modified_time()->toStringShort() : ''
                )
            ) ;
        }
        return $result ;
    }
}
?>
