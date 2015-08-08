<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'regdb/regdb.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \AuthDB\AuthDB ;
use \FileMgr\DbConnection ;
use \FileMgr\FileMgrException ;
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
    public static $allocation_names = array('tuning', 'alignment', 'daq', 'access', 'machine', 'other') ;

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

    /**
     * Check if the specified string represents a valid shift type
     *
     * @param string $type
     * @return boolean
     */
    public static function is_valid_shift_type ($type) {
        $type = strtoupper($type) ;
        return in_array($type, array('USER', 'MD', 'IN-HOUSE'), true) ;
    }

    /**
     * Throw an exception if the specified name doesn't correspond to any known shift type.
     * 
     * @param string $type
     * @throws ShiftMgrException
     */
    public function assert_shift_type ($type) {
        if (!ShiftMgr::is_valid_shift_type($type))
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "no such shift type supported by the application: '{$type}'") ;
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
        $id = intval($id) ;
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
    public function find_last_shift ($instr_name) {
        $instr_name_escaped = $this->escape_string(strtoupper(trim($instr_name))) ;
        $result = $this->query("SELECT * FROM {$this->database}.shift WHERE instr_name='{$instr_name_escaped}' ORDER BY begin_time DESC LIMIT 1") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows == 1) return new Shift($this, mysql_fetch_array($result, MYSQL_ASSOC)) ;
        throw new ShiftMgrException (
              __class__.'::'.__METHOD__ ,
              "Invalid number of entries returned when looking for the last shift at {$instr_name}") ;
    }
    public function find_shift_area_by_id ($id) {
        $id = intval($id) ;
        $result = $this->query("SELECT * FROM {$this->database}.shift_area_evaluation WHERE id={$id}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows == 1) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            return new ShiftAreaEvaluation (
                $this->find_shift_by_id($attr['shift_id']) ,
                $attr) ;
        }
        throw new ShiftMgrException (
            __class__.'::'.__METHOD__ ,
            "Invalid number of entries returned when looking for shift area by id={$id}") ;
    }
    public function find_shift_time_by_id ($id) {
        $id = intval($id) ;
        $result = $this->query("SELECT * FROM {$this->database}.shift_time_allocation WHERE id={$id}") ;
        $nrows = mysql_numrows($result) ;
        if ($nrows == 0) return null ;
        if ($nrows == 1) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            return new ShiftTimeAllocation (
                $this->find_shift_by_id($attr['shift_id']) ,
                $attr) ;
        }
        throw new ShiftMgrException (
            __class__.'::'.__METHOD__ ,
            "Invalid number of entries returned when looking for shift time allocation by id={$id}") ;
    }

    public function precreate_shifts_if_needed ($instr_name, $begin_time, $end_time) {

        $this->assert_hutch($instr_name) ;

        // Never go before that "very first" date of this database, or before the end time
        // of the last known shift of the instrument.

        $min_begin_time = ShiftMgr::min_begin_time() ;
        $begin_time = $begin_time ? ($begin_time->less($min_begin_time) ? $min_begin_time : $begin_time) : $min_begin_time ;

        $last_shift = $this->find_last_shift($instr_name) ;
        if ($last_shift) {
            $last_shift_end_time = $last_shift->end_time() ;
            $begin_time = $begin_time->less($last_shift_end_time) ? $last_shift_end_time : $begin_time ;
        }
        $begin_time = LusiTime::parse("{$begin_time->toStringDay()} 00:00:00") ;

        // Never go beyond the comming midnight of the present day.
        //
        // NOTE: that we're looking +25 hours ahead just in case if we hit
        // the Daylight savings time switch which happens at midnight.
        // If that will happen to be in November then the day will be 25 hours
        // in length. So we would need to have an interval in 25 hours. This should still
        // work for normal days as well as for March when an opposite Daylight savings
        // time switch happens.

        $now = LusiTime::now() ;
        $max_end_time = LusiTime::parse("{$now->in25hours()->toStringDay()} 00:00:00") ;
        $end_time = $end_time ? ($max_end_time->less($end_time) ? $max_end_time : $end_time) : $max_end_time ;

        if (!($begin_time->less($end_time))) return ;   // no room for new shifts

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
                        try {
                            $this->create_shift (
                                $instr_name ,
                                LusiTime::parse("{$day->begin->toStringDay()} 09:00:00") ,
                                LusiTime::parse("{$day->begin->toStringDay()} 21:00:00")
                            ) ;
                        } catch (FileMgrException $e) {

                            // To deal with race conditions in case if another user created
                            // the same shift after the current transaction began.

                            if (is_null($e->errno) || ($e->errno != DbConnection::$ER_DUP_ENTRY)) throw $e;
                        }
                    }
                }
                if (is_null($shift_after_noon)) {
                    if (!(($day->begin->toStringDay() == $now->toStringDay()) && ($now->hour() < 21))) {
                        try {
                            $this->create_shift (
                                $instr_name ,
                                LusiTime::parse("{$day->begin->toStringDay()} 21:00:00") ,
                                LusiTime::parse("{$day->begin->in25hours()->toStringDay()}  09:00:00")  // NOTE: see previous notes on the Daylight savings time switch
                            ) ;
                        } catch (FileMgrException $e) {

                            // To deal with race conditions in case if another user created
                            // the same shift after the current transaction began.

                            if (is_null($e->errno) || ($e->errno != DbConnection::$ER_DUP_ENTRY)) throw $e;
                        }
                    }
                }
            }
        }
    }

    public function create_shift ($instr_name, $begin_time, $end_time, $type=null) {

        $this->assert_hutch($instr_name) ;
        $instr_name_escaped = $this->escape_string(strtoupper(trim($instr_name))) ;

        // If no type provided then deduce it based on the day of week
        // of the begin shift timestamp.

        if (is_null($type)) {
            switch ($begin_time->day_of_week()) {
                case 2:
                case 3:
                    $type = 'MD' ; break ;
                default:
                    $type = 'USER' ; break ;
            }
        } else {
            $this->assert_shift_type($type) ;
        }
        $type_escaped = $this->escape_string(strtoupper(trim($type))) ;

        $sql = "INSERT INTO {$this->database}.shift (instr_name,type,begin_time,end_time,notes) VALUES('{$instr_name_escaped}','{$type_escaped}',{$begin_time->to64()},{$end_time->to64()},'')" ;
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
        $this->add_create_shift_event($shift->id()) ;
        return $shift ;
    }

    public function update_shift ($id, $begin_time=null, $end_time=null, $type=null, $notes=null, $areas=null, $allocations=null) {

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

        $this->assert_shift_type($type) ;

        $modified_time = LusiTime::now() ;

        $opt = " modified_uid='{$this->current_user()}', modified_time={$modified_time->to64()}" ;

        if ($begin_time)      $opt .= ", begin_time={$begin_time->to64()}" ;
        if ($end_time)        $opt .= ", end_time={$end_time->to64()}" ;
        if (!is_null($notes)) $opt .= ", notes='".$this->escape_string($notes)."'" ;
        if (!is_null($type))  $opt .= ", type='".$this->escape_string(strtoupper(trim($type)))."'" ;

        $sql = "UPDATE {$this->database}.shift SET {$opt} WHERE id={$shift->id()}" ;
        $this->query($sql) ;

        if ($begin_time      && !$begin_time->equal($shift->begin_time())) $this->add_modify_shift_event ($shift->id(), $modified_time, 'begin_time', $shift->begin_time()->toStringShort(), $begin_time->toStringShort()) ;
        if ($end_time        && !$end_time  ->equal($shift->end_time()))   $this->add_modify_shift_event ($shift->id(), $modified_time, 'end_time',   $shift->end_time()  ->toStringShort(), $end_time->toStringShort()) ;
        if (!is_null($type)  && $type            != $shift->type())        $this->add_modify_shift_event ($shift->id(), $modified_time, 'type',       $shift->type(),                        $type) ;
        if (!is_null($notes) && $notes           != $shift->notes())       $this->add_modify_shift_event ($shift->id(), $modified_time, 'notes',      $shift->notes(),                       $notes) ;

        // The updated version of the shift

        $shift = $this->find_shift_by_id($id) ;

        if ($areas) {
            $old_areas = $shift->areas() ;
            foreach ($areas as $area_name => $area) {
                $area_name_escaped = $this->escape_string($area_name) ;
                $opt = ' problems='.($area->problems ? 1 : 0).', downtime_min='.intval($area->downtime_min).", comments='".$this->escape_string($area->comments)."'" ;
                $sql = "UPDATE {$this->database}.shift_area_evaluation SET {$opt} WHERE shift_id={$shift->id()} AND name='{$area_name_escaped}'" ;
                $this->query($sql) ;
                $old_area = $old_areas[$area_name] ;
                if ($old_area->problems()     != $area->problems)     $this->add_modify_area_event($old_area->id(), $modified_time, 'problems',     $old_area->problems(),     $area->problems) ;
                if ($old_area->downtime_min() != $area->downtime_min) $this->add_modify_area_event($old_area->id(), $modified_time, 'downtime_min', $old_area->downtime_min(), $area->downtime_min) ;
                if ($old_area->comments()     != $area->comments)     $this->add_modify_area_event($old_area->id(), $modified_time, 'comments',     $old_area->comments(),     $area->comments) ;
            }
        }
        if ($allocations) {
            $old_allocations = $shift->allocations() ;

            function update_allocation ($shiftmgr, $shift, $allocation_name, $modified_time, $old_allocation, $allocation) {
                $allocation_name_escaped = $shiftmgr->escape_string($allocation_name) ;
                $duration_min = intval($allocation->duration_min) ;
                $opt = '  duration_min='.$duration_min.", comments='".$shiftmgr->escape_string($allocation->comments)."'" ;
                $sql = "UPDATE {$shiftmgr->database}.shift_time_allocation SET {$opt} WHERE shift_id={$shift->id()} AND name='{$allocation_name_escaped}'" ;
                $shiftmgr->query($sql) ;
                if ($old_allocation->duration_min() != $allocation->duration_min) $shiftmgr->add_modify_time_event($old_allocation->id(), $modified_time, 'duration_min', $old_allocation->duration_min(), $allocation->duration_min) ;
                if ($old_allocation->comments()     != $allocation->comments)     $shiftmgr->add_modify_time_event($old_allocation->id(), $modified_time, 'comments',     $old_allocation->comments(),     $allocation->comments) ;
            }

            $ival = new LusiInterval($shift->begin_time(), $shift->end_time()) ;
            $allocations->other->duration_min = $ival->toMinutes() ;

            foreach ($allocations as $allocation_name => $allocation) {
                if ($allocation_name != 'other') {
                    update_allocation($this, $shift, $allocation_name, $modified_time, $old_allocations[$allocation_name], $allocation) ;
                    $allocations->other->duration_min -= $allocation->duration_min ;
                }
            }
            if ($allocations->other->duration_min < 0) $allocations->other->duration_min = 0 ;

            update_allocation($this, $shift, 'other', $modified_time, $old_allocations['other'], $allocations->other) ;
        }
        return $this->find_shift_by_id($id) ;
    }
    public function delete_shift_by_id ($id) {

        $shift = $this->find_shift_by_id($id) ;
        if (!$shift) return ;

        $sql = "DELETE FROM {$this->database}.shift WHERE id={$shift->id()}" ;
        $this->query($sql) ;
    }
    private function add_create_shift_event ($shift_id) {
        $modified_uid_escaped = $this->escape_string(AuthDB::instance()->authName()) ;
        $modified_time        = LusiTime::now() ;

        $sql = "INSERT INTO {$this->database}.shift_history VALUES(NULL,{$shift_id},'{$modified_uid_escaped}',{$modified_time->to64()},'CREATE','','','')" ;
        $this->query($sql) ;
    }
    private function add_modify_shift_event ($shift_id, $modified_time, $parameter, $old_value, $new_value) {
        $modified_uid_escaped = $this->escape_string(AuthDB::instance()->authName()) ;
        $parameter_escaped    = $this->escape_string($parameter) ;
        $old_value_escaped    = $this->escape_string($old_value) ;
        $new_value_escaped    = $this->escape_string($new_value) ;

        $sql = "INSERT INTO {$this->database}.shift_history VALUES(NULL,{$shift_id},'{$modified_uid_escaped}',{$modified_time->to64()},'MODIFY','{$parameter_escaped}','{$old_value_escaped}','{$new_value_escaped}')" ;
        $this->query($sql) ;
    }
    private function add_modify_area_event($area_id, $modified_time, $parameter, $old_value, $new_value) {
        $modified_uid_escaped = $this->escape_string(AuthDB::instance()->authName()) ;
        $parameter_escaped    = $this->escape_string($parameter) ;
        $old_value_escaped    = $this->escape_string($old_value) ;
        $new_value_escaped    = $this->escape_string($new_value) ;

        $sql = "INSERT INTO {$this->database}.shift_area_history VALUES(NULL,{$area_id},'{$modified_uid_escaped}',{$modified_time->to64()},'{$parameter_escaped}','{$old_value_escaped}','{$new_value_escaped}')" ;
        $this->query($sql) ;
    }
    public function add_modify_time_event($time_id, $modified_time, $parameter, $old_value, $new_value) {
        $modified_uid_escaped = $this->escape_string(AuthDB::instance()->authName()) ;
        $parameter_escaped    = $this->escape_string($parameter) ;
        $old_value_escaped    = $this->escape_string($old_value) ;
        $new_value_escaped    = $this->escape_string($new_value) ;

        $sql = "INSERT INTO {$this->database}.shift_time_history VALUES(NULL,{$time_id},'{$modified_uid_escaped}',{$modified_time->to64()},'{$parameter_escaped}','{$old_value_escaped}','{$new_value_escaped}')" ;
        $this->query($sql) ;
    }

    public function history ($instr_name=null, $begin_time=null, $end_time=null, $since_time=null) {

        $history = array();

        $instr_opt = $instr_name ? "instr_name='".$this->escape_string(strtoupper(trim($instr_name)))."'" : '' ;

        if (($begin_time && $end_time) && !$begin_time->less($end_time))
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "the begin time of the interval must be strictly less than the end one") ;

        $begin_time_opt = $begin_time ? " modified_time >= {$begin_time->to64()}" : '' ;
        $end_time_opt   = $end_time   ? " modified_time <  {$end_time->to64()}"   : '' ;
        $since_time_opt = $since_time ? " modified_time >= {$since_time->to64()}" : '' ;

        // Fetch events for shifts

        $opt = '' ;
        if ($instr_opt)      $opt = "WHERE shift_id IN (SELECT id FROM {$this->database}.shift WHERE {$instr_opt}) " ;
        if ($begin_time_opt) $opt .= ($opt ? ' AND ' : 'WHERE ').$begin_time_opt ;
        if ($end_time_opt)   $opt .= ($opt ? ' AND ' : 'WHERE ').$end_time_opt ;
        if ($since_time_opt) $opt .= ($opt ? ' AND ' : 'WHERE ').$since_time_opt ;

        $shifts = array() ;     // shifts cached by their identifiers

        $sql = "SELECT * FROM {$this->database}.shift_history {$opt} ORDER BY modified_time DESC" ;

        $result = $this->query($sql) ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $shift_id = intval($attr['shift_id']) ;
            $shift = null ;
            if (array_key_exists($shift_id, $shifts)) {
                $shift = $shifts[$shift_id] ;
            } else {
                $shift = $this->find_shift_by_id($shift_id) ;
                if (!$shift)
                    throw new ShiftMgrException (
                        __class__.'::'.__METHOD__ ,
                        "the shift with id={$shift_id} no longer exists inthe database") ;
                $shifts[$shift->id()] = $shift ;
            }
            array_push (
                $history ,
                new ShiftHistoryEvent (
                    $shift ,    // shift-specific informaton willbe extracted from here
                    'SHIFT' ,   // scope
                    '' ,        // scope2
                    $attr       // event-specific information will be extracted from here
                )
            ) ;
        }

        // Fetch events for areas & time allocations

        $opt = '' ;
        if ($instr_opt)      $opt = "WHERE area_id IN (SELECT id FROM {$this->database}.shift_area_evaluation WHERE shift_id IN (SELECT id FROM {$this->database}.shift WHERE {$instr_opt})) " ;
        if ($begin_time_opt) $opt .= ($opt ? ' AND ' : 'WHERE ').$begin_time_opt ;
        if ($end_time_opt)   $opt .= ($opt ? ' AND ' : 'WHERE ').$end_time_opt ;
        if ($since_time_opt) $opt .= ($opt ? ' AND ' : 'WHERE ').$since_time_opt ;

        $areas = array() ;     // areas cached by their identifiers

        $result = $this->query("SELECT * FROM {$this->database}.shift_area_history {$opt} ORDER BY modified_time DESC") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $area_id = intval($attr['area_id']) ;
            $area = null ;
            if (array_key_exists($area_id, $areas)) {
                $area = $areas[$area_id] ;
            } else {
                $area = $this->find_shift_area_by_id($area_id) ;
                if (!$area)
                    throw new ShiftMgrException (
                        __class__.'::'.__METHOD__ ,
                        "the shift area with id={$area_id} no longer exists inthe database") ;
                $areas[$area->id()] = $area ;
            }
            array_push (
                $history ,
                new ShiftHistoryEvent (
                    $area->shift() ,    // shift-specific informaton willbe extracted from here
                    'AREA' ,            // scope
                    $area->name() ,     // scope2
                    $attr               // event-specific information will be extracted from here
                )
            ) ;
        }

        $opt = '' ;
        if ($instr_opt)      $opt = "WHERE time_id IN (SELECT id FROM {$this->database}.shift_time_allocation WHERE shift_id IN (SELECT id FROM {$this->database}.shift WHERE {$instr_opt})) " ;
        if ($begin_time_opt) $opt .= ($opt ? ' AND ' : 'WHERE ').$begin_time_opt ;
        if ($end_time_opt)   $opt .= ($opt ? ' AND ' : 'WHERE ').$end_time_opt ;
        if ($since_time_opt) $opt .= ($opt ? ' AND ' : 'WHERE ').$since_time_opt ;

        $times = array() ;     // times cached by their identifiers

        $result = $this->query("SELECT * FROM {$this->database}.shift_time_history {$opt} ORDER BY modified_time DESC") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            $time_id = intval($attr['time_id']) ;
            $time = null ;
            if (array_key_exists($time_id, $times)) {
                $time = $times[$time_id] ;
            } else {
                $time = $this->find_shift_time_by_id($time_id) ;
                if (!$time)
                    throw new ShiftMgrException (
                        __class__.'::'.__METHOD__ ,
                        "the shift time allocation with id={$time_id} no longer exists inthe database") ;
                $times[$time->id()] = $time ;
            }
            array_push (
                $history ,
                new ShiftHistoryEvent (
                    $time->shift() ,    // shift-specific informaton willbe extracted from here
                    'TIME' ,            // scope
                    $time->name() ,     // scope2
                    $attr               // event-specific information will be extracted from here
                )
            ) ;
        }
        return $history ;
    }
}

?>
