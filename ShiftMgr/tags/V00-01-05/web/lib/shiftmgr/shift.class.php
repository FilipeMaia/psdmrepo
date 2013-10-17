<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'dataportal/dataportal.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;
use DataPortal\ExpTimeMon ;

/**
 * Class Shift is an abstraction for shifts.
 *
 * @author gapon
 */
class Shift {

    // Data members

    private $shiftmgr ;

    public $attr ;

    public function __construct ($shiftmgr, $attr) {
        $this->shiftmgr = $shiftmgr ;
        $this->attr = $attr ;
    }

    // Accessors

    public function shiftmgr   () { return                  $this->shiftmgr ; }
    public function id         () { return           intval($this->attr['id']) ; }
    public function instr_name () { return  strtoupper(trim($this->attr['instr_name'])) ; }
    public function begin_time () { return LusiTime::from64($this->attr['begin_time']) ; }
    public function end_time   () { return LusiTime::from64($this->attr['end_time']) ; }
    public function type       () { return  strtoupper(trim($this->attr['type'])) ; }
    public function notes      () { return                  $this->attr['notes'] ; }

    public function modified_time () {
        $str = $this->attr['modified_time'] ;
        if ($str) return LusiTime::from64($str) ;
        return null ;
    }

    public function modified_uid () {
        $str = $this->attr['modified_uid'] ;
        if ($str) return strtolower(trim($str)) ;
        return null ;
    }

    /**
     * Return the shift duration expressed in minutes
     *
     * @return int
     */
    public function duration () {
        return intval(($this->end_time()->sec - $this->begin_time()->sec) / 60) ;
    }

    /**
     * The number of minutes the X-ray beam stopper into the hutch was open during the shift
     * 
     * TODO: This information is fetched from teh corresponding EPICS PV based
     *       on the shift interval.
     *
     * @return int
     */
    public function stopper () {
        $stopper = 0 ;
        ExpTimeMon::instance()->begin() ;
        foreach (ExpTimeMon::instance()->beamtime_beam_status('XRAY_DESTINATIONS', $this->begin_time(), $this->end_time()) as $triplet) {
            if ($triplet['status'] & ExpTimeMon::beam_destination_mask($this->instr_name())) {
                $ival = new LusiInterval($triplet['begin_time'], $triplet['end_time']) ;
                $stopper += $ival->toMinutes() ;
            }
        }
        return $stopper ;
    }

    /**
     * The number of minutes the hutch door was "secured" during the shift
     * 
     * TODO: This information is fetched from the corresponding EPICS PV based
     *       on the shift interval.
     *
     * @return int
     */
    public function door () {
        $door = 0 ;
        ExpTimeMon::instance()->begin() ;
        $pv = ExpTimeMon::door_secured_pv($this->instr_name()) ;
        foreach (ExpTimeMon::instance()->beamtime_beam_status($pv, $this->begin_time(), $this->end_time()) as $triplet) {
            if ($triplet['status']) {
                $ival = new LusiInterval($triplet['begin_time'], $triplet['end_time']) ;
                $door += $ival->toMinutes() ;
            }
        }
        return $door ;
    }

    /**
     * The number of minutes the hutch door was open during the shift
     *
     * @return int
     */
    public function door_open () {
        $minutes = $this->duration() - $this->door() ;
        return $minutes < 0 ? 0 : $minutes ;
    }

    // Area evaluations
    
    public function areas () {
        $areas = array() ;
        $result = $this->shiftmgr()->query("SELECT * FROM {$this->shiftmgr()->database}.shift_area_evaluation WHERE shift_id={$this->id()}") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $areas[trim($attr['name'])] = new ShiftAreaEvaluation($this, $attr) ;
        }
        return $areas ;
    }
    
    // Time allocations

    public function allocations () {
        $allocations = array() ;
        $result = $this->shiftmgr()->query("SELECT * FROM {$this->shiftmgr()->database}.shift_time_allocation WHERE shift_id={$this->id()}") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $allocations[trim($attr['name'])] = new ShiftTimeAllocation($this, $attr) ;
        }
        return $allocations ;
    }
    
    /**
     * Return all history events in a scope of this shift, its areas and time allocations
     * 
     * NOTE: The elements will be unsorted.
     *
     * @return array(ShiftHistoryEvent)
     */
    public function history () {
        $history = array();
        $result = $this->shiftmgr()->query("SELECT * FROM {$this->shiftmgr()->database}.shift_history WHERE shift_id={$this->id()} ORDER BY modified_time DESC") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            array_push (
                $history ,
                new ShiftHistoryEvent (
                    $this ,             // shift-specific informaton willbe extracted from here
                    'SHIFT' ,           // scope
                    '' ,                // scope2
                    $attr               // event-specific information will be extracted from here
                )
            ) ;
        }
        foreach ($this->areas()       as $name => $a) $history = array_merge ($history, $a->history()) ;
        foreach ($this->allocations() as $name => $a) $history = array_merge ($history, $a->history()) ;
        return $history ;
    }
}
?>
