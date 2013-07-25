<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'dataportal/dataportal.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;
use DataPortal\ExpTimeMon ;

/**
 * Class ShiftMgrShift is an abstraction for shifts.
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
     * TODO: This information is fetched from teh corresponding EPICS PV based
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

    // Area evaluations
    
    public function areas () {
        $areas = array() ;
        $sql = "SELECT * FROM {$this->shiftmgr()->database}.shift_area_evaluation WHERE shift_id={$this->id()}" ;
        $result = $this->shiftmgr()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $areas[trim($attr['name'])] = array (
                'problems'     => $attr['problems'] ? 1 : 0 ,
                'downtime_min' => intval($attr['downtime_min']) ,
                'comments'     => $attr['comments']
            ) ;
        }
        return $areas ;
    }
    
    // Time allocations

    public function allocations () {
        $allocations = array() ;
        $sql = "SELECT * FROM {$this->shiftmgr()->database}.shift_time_allocation WHERE shift_id={$this->id()}" ;
        $result = $this->shiftmgr()->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $allocations[trim($attr['name'])] = array (
                'duration_min' => intval($attr['duration_min']) ,
                'comments'     => $attr['comments']
            ) ;
        }
        return $allocations ;
    }
}
?>
