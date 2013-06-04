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
    public static function shifts2array($shifts) {
        $result = array();
        foreach ($shifts as $s) {
            array_push(
                $result ,
                array(
                    'id'                  => $s->id(),
                    'username'            => $s->username(),
                    'hutch'               => $s->hutch(),
                    'start_time'          => $s->start_time()->sec,
                    'end_time'            => $s->end_time() ? $s->end_time()->sec : 0,
                    'last_modified_time'  => $s->last_modified_time()->sec,
                    'stopper_out'         => $s->stopper_out(),
                    'door_open'           => $s->door_open(),
                    'total_shots'         => $s->total_shots(),
                    'other_notes'         => $s->other_notes()
                )
            );
        }
        return $result;
    }
}
?>
