<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'regdb/regdb.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \AuthDB\AuthDB ;
use \FileMgr\DbConnection ;
use \LusiTime\LusiTime ;
use \LusiTime\LusiInterval ;

use \RegDB\RegDB ;

/**
 * Class ShiftMgr encapsulates operations with the PCDS Shift Management database
 *
 * @author gapon
 */
class ShiftMgr extends DbConnection {

    // ------------------------
    // --- STATIC INTERFACE ---
    // ------------------------

    private static $instance = null ;

    public static $area_names       = array('FEL', 'BMLN', 'CTRL', 'DAQ', 'LASR', 'TIME', 'HALL', 'OTHR') ;
    public static $allocation_names = array('tuning', 'alignment', 'daq', 'access', 'other') ;

    private static $min_begin_time = null ;
    public static function min_begin_time () {
        if (is_null(ShiftMgr::$min_begin_time)) ShiftMgr::$min_begin_time = LusiTime::parse('2013-07-24 00:00:00') ;
        return ShiftMgr::$min_begin_time ;
    }

    /**
     * Singleton to simplify certain operations.
     *
     * @return ShiftMgr
     */
    public static function instance() {
        if (is_null(ShiftMgr::$instance)) ShiftMgr::$instance =
            new ShiftMgr (
                SHIFTMGR_DEFAULT_HOST ,
                SHIFTMGR_DEFAULT_USER ,
                SHIFTMGR_DEFAULT_PASSWORD ,
                SHIFTMGR_DEFAULT_DATABASE) ;
        return ShiftMgr::$instance ;
    }

    /**
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database
     */
    public function __construct ($host, $user, $password, $database) {
        parent::__construct($host, $user, $password, $database) ;
    }

    /**
     * Throw an exception if the specified name doesn't correspond to any known hutch.
     *
     * @param string $name - hutch name
     * @throws ShiftMgrException
     */
    private function assert_hutch ($name) {
        
        // The implementation of the method will check if the specified
        // name matches any known hutch in the Experiment Registry Database.
        // This will may require to begin a separate transaction.

        RegDB::instance()->begin() ;
        if (is_null(RegDB::instance()->find_instrument_by_name($name)))
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "no such hutch found in the Experiment Registry Database: '{$name}'") ;
    }

    public function current_user () {
        AuthDb::instance()->begin() ;
        return AuthDB::instance()->authName() ;
    }

    /**
     * Return a list of shifts in a scope of teh specified instrument and an optional time interval
     * 
     * @param string $instr_name
     * @param LusiTime $begin_time
     * @param LusiTime $end_time
     * @return array
     */
    public function shifts ($instr_name, $begin_time=null, $end_time=null) {
 
        $shifts = array() ;

        $this->assert_hutch($instr_name) ;
        $instr_name_escaped = $this->escape_string(strtoupper(trim($instr_name))) ;

        // Nothing to return for an empty or incorrect interval. Don't even bother
        // to report this as a problem.

        if (($begin_time && $end_time) && !$begin_time->less($end_time)) return $shifts ;

        $opt = "instr_name='{$instr_name_escaped}'" ;
        if ($begin_time) $opt .= " AND begin_time >= {$begin_time->to64()}" ;
        if ($end_time)   $opt .= " AND begin_time <  {$end_time->to64()}" ;
        return $this->find_shifts_by_($opt) ;
    }

    public function find_shift_by_id ($id) {
        $shifts = $this->find_shifts_by_("id={$id}") ;
        $nrows = count($shifts) ;
        if ($nrows == 0) return null ;
        if ($nrows == 1) return $shifts[0] ;
        throw new ShiftMgrException (
              __class__.'::'.__METHOD__ ,
              "Invalid number of entries returned when looking for shift by id={$id}") ;
    }
    public function find_shift_by_begin_time ($instr_name, $begin_time) {
        $instr_name_escaped = $this->escape_string(strtoupper(trim($instr_name))) ;
        $shifts = $this->find_shifts_by_("instr_name='{$instr_name_escaped}' AND begin_time={$begin_time->to64()}") ;
        $nrows = count($shifts) ;
        if ($nrows == 0) return null ;
        if ($nrows == 1) return $shifts[0] ;
        throw new ShiftMgrException (
              __class__.'::'.__METHOD__ ,
              "Invalid number of entries returned when looking for shift by begin_time={$begin_time->toStringShort()}") ;
    }
    private function find_shifts_by_ ($condition=null) {
        $shifts = array() ;
        $condition = $condition ? "WHERE {$condition}" : '' ;
        $result = $this->query("SELECT * FROM {$this->database}.shift {$condition} ORDER BY instr_name, begin_time DESC") ;
        for ($i = 0, $nrows = mysql_numrows($result); $i < $nrows; $i++) {
             array_push (
                $shifts ,
                new Shift($this, mysql_fetch_array($result, MYSQL_ASSOC))
             ) ;
        }
        return $shifts ;
    }

    public function precreate_shifts_if_needed ($instr_name, $begin_time, $end_time) {

        $this->assert_hutch($instr_name) ;

        // Never go before that date

        $min_begin_time = ShiftMgr::min_begin_time() ;
        $begin_time = $begin_time ? ($begin_time->less($min_begin_time) ? $min_begin_time : $begin_time) : $min_begin_time ;
        $begin_time = LusiTime::parse("{$begin_time->toStringDay()} 00:00:00") ;
              
        // Never go beyond the comming midnight of the present day.

        $now = LusiTime::now() ;
        $max_end_time = LusiTime::parse("{$now->in24hours()->toStringDay()} 00:00:00") ;
        $end_time = $end_time ? ($max_end_time->less($end_time) ? $max_end_time : LusiTime::parse("{$end_time->in24hours()->toStringDay()} 00:00:00")) : $max_end_time ;

//       throw new ShiftMgrException (
//            __class__.'::'.__METHOD__ ,
//            "begin_time: {$begin_time->toStringShort()}, end_time: {$end_time->toStringShort()}") ;

        // Create up to 2 shifts per each day, one before 12:00, and another one after.
        // Don't create the first shift if before 9:00, and don't create the second one if before 21:00.

        $range = new LusiInterval($begin_time, $end_time) ;
        
        foreach ($range->splitIntoDays() as $day) {
            $shift_before_noon = null ;
            $shift_after_noon  = null ;
            foreach ($this->shifts($instr_name, $day->begin, $day->end) as $shift) {
                if ($shift->begin_time()->hour() < 12) $shift_before_noon = $shift ;
                else $shift_after_noon = $shift ;
            }
            if ($day->begin->less($end_time)) {
                if (is_null($shift_before_noon)) {
                    if (!(($day->begin->toStringDay() == $now->toStringDay()) && $now->hour() < 9)) {
                        $shift = $this->create_shift (
                            $instr_name ,
                            LusiTime::parse("{$day->begin->toStringDay()} 09:00:00") ,
                            LusiTime::parse("{$day->begin->toStringDay()} 21:00:00")
                        ) ;
                    }
                }
                if (is_null($shift_after_noon)) {
                    if (!(($day->begin->toStringDay() == $now->toStringDay()) && ($now->hour() < 21))) {
                        $shift = $this->create_shift (
                            $instr_name ,
                            LusiTime::parse("{$day->begin->toStringDay()} 21:00:00") ,
                            LusiTime::parse("{$day->begin->in24hours()->toStringDay()}  09:00:00")
                        ) ;
                    }
                }
            }
        }
    }

/*

INSERT INTO shiftmgr.shift (id,instr_name,begin_time,end_time,notes) VALUES(1,'AMO', (SELECT UNIX_TIMESTAMP('2013-07-19 21:00:00')*1000000000), (SELECT UNIX_TIMESTAMP('2013-07-20 09:00:00')*1000000000),'');

INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'FEL', 1, 15,'');
INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'BMLN',0,  0,'');
INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'CTRL',0,  0,'');
INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'DAQ', 1,125,'Something went wrong with the DAQ');
INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'LASR',0,  0,'');
INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'HALL',1, 12,'');
INSERT INTO shiftmgr.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES (1,'OTHR',0,  0,'');

INSERT INTO shiftmgr.shift_time_allocation (shift_id,name,duration_min,comments) VALUES (1,'tuning',   120,'');
INSERT INTO shiftmgr.shift_time_allocation (shift_id,name,duration_min,comments) VALUES (1,'alignment', 55,'');
INSERT INTO shiftmgr.shift_time_allocation (shift_id,name,duration_min,comments) VALUES (1,'daq',      345,'');
INSERT INTO shiftmgr.shift_time_allocation (shift_id,name,duration_min,comments) VALUES (1,'access',     0,'');
INSERT INTO shiftmgr.shift_time_allocation (shift_id,name,duration_min,comments) VALUES (1,'other',    200,'');

*/
    public function create_shift ($instr_name, $begin_time, $end_time) {

        $this->assert_hutch($instr_name) ;
        $instr_name_escaped = $this->escape_string(strtoupper(trim($instr_name))) ;

        $sql = "INSERT INTO {$this->database}.shift (instr_name,begin_time,end_time,notes) VALUES('{$instr_name_escaped}',{$begin_time->to64()},{$end_time->to64()},'')" ;
        $this->query($sql) ;

        $shifts = $this->find_shifts_by_('id=LAST_INSERT_ID()') ;
        if (count($shifts) != 1)
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                'Failed to create the new shift') ;
        $shift = $shifts[0] ;

        foreach (ShiftMgr::$area_names as $area_name) {
            $area_name_escaped = $this->escape_string(trim($area_name)) ;
            $sql = "INSERT INTO {$this->database}.shift_area_evaluation (shift_id,name,problems,downtime_min,comments) VALUES ({$shift->id()},'{$area_name_escaped}',0,0,'')" ;
            $this->query($sql) ;
        }
        foreach (ShiftMgr::$allocation_names as $allocation_name) {
            $allocation_name_escaped = $this->escape_string(trim($allocation_name)) ;
            $sql = "INSERT INTO {$this->database}.shift_time_allocation (shift_id,name,duration_min,comments) VALUES ({$shift->id()},'{$allocation_name_escaped}',0,'')" ;
            $this->query($sql) ;
        }
        return $shift ;
    }

    public function update_shift ($id, $begin_time=null, $end_time=null, $notes=null, $areas=null, $allocations=null) {

        // The shift object before applying first modfs

        $shift = $this->find_shift_by_id($id) ;
        if (!$shift)
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "No shift found for id={$id}") ;

        if (($begin_time && $end_time) && !$begin_time->less($end_time))
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "the begin time of the interval must be strictly less than the end one") ;

        $modified_time = LusiTime::now() ;

        $opt = " modified_uid='{$this->current_user()}', modified_time={$modified_time->to64()}" ;

        if ($begin_time) $opt .= ", begin_time={$begin_time->to64()}" ;
        if ($end_time)   $opt .= ", end_time={$end_time->to64()}" ;
        if ($notes)      $opt .= ", notes='".$this->escape_string($notes)."'" ;

        $sql = "UPDATE {$this->database}.shift SET {$opt} WHERE id={$shift->id()}" ;
        $this->query($sql) ;

        // The updated version of the shift

        $shift = $this->find_shift_by_id($id) ;

        if ($areas) {
            foreach ($areas as $area_name => $area) {
                $area_name_escaped = $this->escape_string($area_name) ;
                $opt = ' problems='.($area->problems ? 1 : 0).', downtime_min='.intval($area->downtime_min).", comments='".$this->escape_string($area->comments)."'" ;
                $sql = "UPDATE {$this->database}.shift_area_evaluation SET {$opt} WHERE shift_id={$shift->id()} AND name='{$area_name_escaped}'" ;
                $this->query($sql) ;
            }
        }
        if ($allocations) {
            function update_allocation ($shiftmgr, $shift, $allocation_name, $allocation) {
                $allocation_name_escaped = $shiftmgr->escape_string($allocation_name) ;
                $duration_min = intval($allocation->duration_min) ;
                $opt = '  duration_min='.$duration_min.", comments='".$shiftmgr->escape_string($allocation->comments)."'" ;
                $sql = "UPDATE {$shiftmgr->database}.shift_time_allocation SET {$opt} WHERE shift_id={$shift->id()} AND name='{$allocation_name_escaped}'" ;
                $shiftmgr->query($sql) ;
            }

            $ival = new LusiInterval($shift->begin_time(), $shift->end_time()) ;
            $allocations->other->duration_min = $ival->toMinutes() ;

            foreach ($allocations as $allocation_name => $allocation) {
                if ($allocation_name != 'other') {
                    update_allocation($this, $shift, $allocation_name, $allocation) ;
                    $allocations->other->duration_min -= $allocation->duration_min ;
                }
            }
            if ($allocations->other->duration_min < 0) $allocations->other->duration_min = 0 ;

            update_allocation($this, $shift, 'other', $allocations->other) ;
        }
        return $this->find_shift_by_id($id) ;
    }
    
    public function delete_shift_by_id ($id) {

        $shift = $this->find_shift_by_id($id) ;
        if (!$shift) return ;

        $sql = "DELETE FROM {$this->database}.shift WHERE id={$shift->id()}" ;
        $this->query($sql) ;
    }
}

?>
