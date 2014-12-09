<?php

namespace LogBook ;

require_once 'logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/*
 * The class representing experimental shifts.
 */
class LogBookShift {

    /* Data members
     */
    private $logbook ;
    private $experiment ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($logbook, $experiment, $attr) {
        $this->logbook    = $logbook ;
        $this->experiment = $experiment ;
        $this->attr       = $attr ;
    }

    /* Accessors
     */
    public function parent     () { return $this->experiment ; }
    public function id         () { return intval($this->attr['id']) ; }
    public function exper_id   () { return $this->attr['exper_id'] ; }
    public function begin_time () { return LusiTime::from64($this->attr['begin_time']) ; }
    public function end_time   () { return is_null($this->attr['end_time']) ? null : LusiTime::from64($this->attr['end_time']) ; }
    public function leader     () { return $this->attr['leader'] ; }

    public function is_closed () { return !is_null($this->end_time()) ; }

    public function in_interval ($time) {
        return LusiTime::in_interval (
            $time ,
            $this->begin_time() ,
            $this->end_time()) ;
    }

    /* Close the open-ended shift
     */
    public function close ($end_time) {

        if (!is_null($this->end_time()))
            throw new LogBookException(__METHOD__, "the shift is already closed") ;

        $this->set_end_time($end_time) ;
    }

    /**
     * Change the begin time of the shift
     *
     * @param LusiTime $time
     */
    public function set_begin_time ($time) {

        if (is_null($time))
            throw new LogBookException(__METHOD__, "the time can't be null") ;

        if (!is_null($this->end_time()) && !$time->less($this->end_time()))
            throw new LogBookException(__METHOD__, "the requested begin time '{$time}' must be strictly less than the shift's end time '{$this->end_time()}'") ;

        $time_64 = $time->to64() ;
        $this->logbook->query (
            "UPDATE {$this->logbook->database}.shift SET begin_time={$time_64} WHERE id={$this->id()}") ;

        $this->attr['begin_time'] = $time_64 ;
    }

    /**
     * Change the end time of the shift
     *
     * @param LusiTime $time
     */
    public function set_end_time ($time) {

        if (is_null($time))
            throw new LogBookException(__METHOD__, "the time can't be null") ;

        if (!$this->begin_time()->less($time))
            throw new LogBookException(__METHOD__, "the begin time '{$this->begin_time()}' must be strictly less than the requested end time '{$time}'") ;

        $time_64 = $time->to64() ;
        $this->logbook->query (
            "UPDATE {$this->logbook->database}.shift SET end_time={$time_64} WHERE id={$this->id()}") ;

        $this->attr['end_time'] = $time_64 ;
    }

    /**
     * Get a complete crew (names of all members including the leader)
     *
     * @return Array
     */
    public function crew () {

        $list = array() ;

        $result = $this->logbook->query (
            "SELECT member FROM {$this->logbook->database}.shift_crew WHERE shift_id={$this->id()}") ;

        for ($nrows = mysql_numrows($result), $i = 0; $i < $nrows; $i++) {
            $member =  mysql_result($result, $i) ;
            array_push($list, $member) ;
            if ($member === $this->leader())
                $leader_in_crew = true ;
        }
        if (!$leader_in_crew)
            array_push($list, $this->leader()) ;

        return $list ;
    }

    /**
     * The number of runs which began within the shift's boundaries
     *
     * @return Number
     */
    public function num_runs () {
        $condition = 'begin_time >= '.$this->attr['begin_time'] ;
        if (!is_null($this->attr['end_time']))
            $condition .= ' AND begin_time < '.$this->attr['end_time'] ;
        return $this->parent()->num_runs($condition) ;
    }

    /**
     * The list of runs which began within the shift's boundaries
     *
     * @return Array
     */
    public function runs () {
        return $this->parent()->runs (
            "begin_time >= {$this->begin_time()->to64()}" .
            (is_null($this->end_time()) ? '' : " AND begin_time < {$this->end_time()->to64()}")
        );
    }

    /**
     * Messages which were directly associuated with thr shift
     * @return Array
     */
    public function entries () {
        return $this->parent()->entries_of_shift($this->id()) ;
    }

    /**
     * Find the previous shift (if any)
     *
     * @return LogBookShift
     */
    public function previous () {
        return $this->experiment->find_prev_shift_for($this) ;
    }

    /**
     * Find the next shift (if any)
     *
     * @return LogBookShift
     */
    public function next () {
        return $this->experiment->find_next_shift_for($this) ;
    }
}
?>
