<?php

namespace ShiftMgr;

require_once( 'shiftmgr.inc.php' );

/**
 * Class ShiftMgrUtils is a utility class accomodating a collection of
 * functions used by Web services.
 *
 * @author gapon
 */
class ShiftMgrUtils {

    /**
     * Convert an input array of objects of class ShiftMgrShift into a portable array
     * of plain objects which can be serialized into JSON format.
     *
     * @param array $shifts
     * @return array
     */
    public static function shifts2array ($shifts) {
        $result = array() ;
        foreach ($shifts as $s) {
            array_push (
                $result ,
                array (
                    'id'              => $s->id() ,
                    'instrument_name' => $s->instrument_name() ,
                    'begin_time'      => $s->begin_time()->toStringShort() ,
                    'begin_time_sec'  => $s->begin_time()->sec ,
                    'end_time'        => $s->is_closed() ? $s->end_time()->toStringShort() : '' ,
                    'end_time_sec'    => $s->is_closed() ? $s->end_time()->sec : 0 ,
                    'is_closed'       => $s->is_closed() ? 1 : 0
                )
            );
        }
        return $result ;
    }
}
?>
