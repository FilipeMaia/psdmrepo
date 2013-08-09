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
     * Convert an input array of objects of class Shift into a portable array
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

            $areas = array() ;
            foreach ($s->areas() as $name => $a)
                $areas[$name] = array (
                    'id'           => $a->id() ,
                    'name'         => $a->name() ,
                    'problems'     => $a->problems() ,
                    'downtime_min' => $a->downtime_min() ,
                    'comments'     => $a->comments()) ;

            $allocations = array() ;
            foreach ($s->allocations() as $name => $a)
                $allocations[$name] = array (
                    'id'           => $a->id() ,
                    'name'         => $a->name() ,
                    'duration_min' => $a->duration_min() ,
                    'comments'     => $a->comments()) ;

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
                    'area'         => $areas ,
                    'allocation'   => $allocations ,
                    'notes'        => $s->notes() ,
                    'editor'       => $s->modified_uid()  ? $s->modified_uid() : '' ,
                    'modified'     => $s->modified_time() ? $s->modified_time()->toStringShort() : ''
                )
            ) ;
        }
        return $result ;
    }
    
    public static function history2array ($events) {
        $result = array() ;
        foreach ($events as $e) {
            $shift = $e->shift() ;
            $begin = $shift->begin_time() ;
            $modified = $e->modified_time() ;
            array_push (
                $result ,
                array (
                    'editor'      => $e->modified_uid() ,
                    'modified'    => array (
                        'sec'     => $modified->sec ,
                        'full'    => $modified->toStringShort()) ,
                    'instr_name'  => $shift->instr_name() ,
                    'shift_id'    => $shift->id() ,
                    'shift_begin' => array (
                        'day'     => $begin->toStringDay() ,
                        'hm'      => $begin->toStringHM() ,
                        'hour'    => $begin->hour() ,
                        'minute'  => $begin->minute() ,
                        'full'    => $begin->toStringShort()) ,
                    'scope'       => $e->scope() ,      // SHIFT, AREA, TIME
                    'scope2'      => $e->scope2() ,     // '' for SHIFT, area name for AREA, and time allocation name for TIME
                    'operation'   => $e->operation() ,  // CREATE, MODIFY
                    'event_id'    => $e->id() ,         // in the given scope
                    'parameter'   => $e->parameter() ,
                    'old_value'   => $e->old_value() ,
                    'new_value'   => $e->new_value()
                )
            ) ;
        }
        return $result ;
    }
}
?>
