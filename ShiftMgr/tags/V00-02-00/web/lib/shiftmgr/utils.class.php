<?php

namespace ShiftMgr;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;
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

        foreach ($shifts as $s)
            array_push (
                $result ,
                Utils::shift2array($s)) ;

        return $result ;
    }

    /**
     * Convert an input object of class Shift into a portable array
     * of plain objects which can be serialized into JSON format.
     *
     * @param Shift $s
     * @return array
     */
    public static function shift2array($s) {

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

        $experiments = array() ;
        foreach ($s->experiments() as $e)
            array_push($experiments, array (
                'id' => $e['id'] ,
                'name' => $e['name'])) ;

        return array (
            'id'           => $s->id() ,
            'instr_name'   => $s->instr_name() ,
            'begin'        => array('day' => $begin->toStringDay(), 'hm' => $begin->toStringHM(), 'hour' => $begin->hour(), 'minute' => $begin->minute(), 'full' => $begin->toStringShort()) ,
            'end'          => array('day' => $end  ->toStringDay(), 'hm' => $end  ->toStringHM(), 'hour' => $end  ->hour(), 'minute' => $end  ->minute(), 'full' => $end  ->toStringShort()) ,
            'type'         => $s->type() ,
            'duration'     => $ival->toStringHM() ,
            'duration_min' => $ival->toMinutes() ,
            'stopper_min'  => $s->stopper() ,
            'door_min'     => $s->door() ,
            'area'         => $areas ,
            'allocation'   => $allocations ,
            'experiments'  => $experiments ,
            'notes'        => $s->notes() ,
            'editor'       => $s->modified_uid()  ? $s->modified_uid() : '' ,
            'modified'     => $s->modified_time() ? $s->modified_time()->toStringShort() : ''
        ) ;
    }
    
    public static function days2array($days) {
        return $days ;
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

    /**
     * This is teh helper function which is meant to consolidate multi-parametric
     * range query operations for locating shifts matching teh specified criterias.
     * The crirerias are expected to be passed as parameters to the Web service.
     * The Web service handler is passed into the function.
     * 
     * Here is the definiton of parameters expected by the function:
     * 
     *   <range> [<begin>] [<end>] [<stopper>] [<door>] [<lcls>] [<daq>] [<instruments>] [<types>]
     *
     * @param object $SVC
     * @return array of Shift objects
     */
    public static function query_shifts($SVC) {

        $range           = $SVC->required_str ('range') ;
        $begin_time      = $SVC->optional_time('begin', LusiTime::parse('2013-07-01 00:00:00')) ;
        $end_time        = $SVC->optional_time('end',   null) ;
        $stopper         = $SVC->optional_int ('stopper', null) ;
        $door            = $SVC->optional_int ('door', null) ;
        $lcls            = $SVC->optional_int ('lcls', null) ;
        $daq             = $SVC->optional_int ('daq', null) ;
        $instruments_str = $SVC->optional_str ('instruments', '') ;
        $types_str       = $SVC->optional_str ('types', '') ;

        switch ($range) {
            case 'week' :
                $begin_time = LusiTime::minus_week() ;
                $end_time   = null ;
                break ;
            case 'month' :
                $begin_time = LusiTime::minus_month() ;
                $end_time   = null ;
                break ;
            case 'range' :
                if (!is_null($end_time) && !$begin_time->less($end_time))
                    $end_time = LusiTime::parse($begin_time->in24hours()->toStringDay().' 00:00:00') ;
                break ;
            default :
                $SVC->abort('unknown range type requested from the shifts service') ;
        }
        $instruments = $instruments_str ? explode(':', $instruments_str) : array() ;
        $types       = $types_str       ? explode(':', $types_str)       : array() ;

        // First find shifts matching the search criteria

        $shifts = array() ;

        foreach ($instruments as $instr_name) {

            $instr = $SVC->regdb()->find_instrument_by_name($instr_name) ;
            if (!$instr) $SVC->abort("Unknown instrument name: {$instr_name}") ;

            if ($instr->is_standard()) {

                $SVC->shiftmgr()->precreate_shifts_if_needed($instr->name(), $begin_time, $end_time) ;

                foreach ($SVC->shiftmgr()->shifts($instr->name(), $begin_time, $end_time) as $shift) {

                    if (!in_array($shift->type(), $types)) continue ;

                    $duration = $shift->duration() ;
                    if ($duration) {
                        if (!is_null($stopper) && ($shift->stopper()   / $duration * 100. <= $stopper)) continue ;
                        if (!is_null($door)    && ($shift->door_open() / $duration * 100. >= $door))    continue ;
                        if (!is_null($lcls))                                                                     ;   // TODO: implement this
                        if (!is_null($daq))                                                                      ;   // TODO: implement this
                    }
                    array_push ($shifts, $shift) ;
                }
            }
        }
        return $shifts ;
    }
}
?>
