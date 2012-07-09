<?php

namespace LogBook;

require_once( 'logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

/*
 * The class representing experimental shifts.
 */
class LogBookShift {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent() {
        return $this->experiment; }

    public function id() {
        return $this->attr['id']; }

    public function exper_id() {
        return $this->attr['exper_id']; }

    public function begin_time() {
        return LusiTime::from64( $this->attr['begin_time'] ); }

    public function end_time() {
        if( is_null( $this->attr['end_time'] )) return null;
        return LusiTime::from64( $this->attr['end_time'] ); }

    public function leader() {
        return $this->attr['leader']; }

    public function in_interval ( $time ) {
        return LusiTime::in_interval (
            $time,
            $this->begin_time(),
            $this->end_time() ); }

    /* Close the open-ended shift
     */
    public function close( $end_time ) {

        if( !is_null( $this->attr['end_time'] ))
            throw new LogBookException(
                __METHOD__, "the shift is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            throw new LogBookException(
                __METHOD__, "end time can't be null" );

        if( !$this->begin_time()->less( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$this->begin_time()."' isn't less than the requested end time '".$end_time."'" );

        /* Make the update
         */
        $end_time_64 = LusiTime::to64from( $end_time );
        $this->connection->query (
            "UPDATE {$this->connection->database}.shift SET end_time=".$end_time_64.
            ' WHERE exper_id='.$this->exper_id().' AND begin_time='.$this->attr['begin_time'] );

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
    }

    /**
     * Get a complete crew (names of all members including the leader)
     *
     * @return array
     */
    public function crew() {

        $list = array();

        $result = $this->connection->query(
            "SELECT member FROM {$this->connection->database}.shift_crew WHERE shift_id=".$this->id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $member =  mysql_result( $result, $i );
            array_push( $list, $member );
            if( $member == $this->leader())
                $leader_in_crew = true;
        }
        if( !$leader_in_crew )
            array_push( $list, $this->leader());

        return $list;
    }

    /* =====================
     *   FREE-FORM ENTRIES
     * =====================
     */

    public function num_runs() {
        $condition = 'begin_time >= '.$this->attr['begin_time'];
        if( !is_null( $this->attr['end_time'] ))
            $condition .= ' AND begin_time < '.$this->attr['end_time'];
        return $this->parent()->num_runs( $condition );
    }

    public function runs() {
        $condition = 'begin_time >= '.$this->begin_time()->to64();
        if( !is_null( $this->end_time()))
            $condition .= ' AND begin_time < '.$this->end_time()->to64();
        return $this->parent()->runs( $condition );
    }

    /* =====================
     *   FREE-FORM ENTRIES
     * =====================
     */
    public function entries () {
        return $this->parent()->entries_of_shift( $this->id());
    }}
?>
