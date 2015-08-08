<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarJobNumberAlloc is an abstraction for job number allocations stored
 * in the database.
 *
 * @author gapon
 */
class NeoCaptarJobNumberAlloc {

   /* Data members
     */
    private $connection;
    private $neocaptar;

    private $attr;
    private $num_in_use;
    private $last_allocation;

    /* Constructor
     */
    public function __construct ($connection, $neocaptar, $attr, $num_in_use, $last_allocation) {
        $this->connection = $connection;
        $this->neocaptar = $neocaptar;
        $this->attr = $attr;
        $this->num_in_use = $num_in_use;
        $this->last_allocation = $last_allocation;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function neocaptar     () { return $this->neocaptar; }
    public function id            () { return intval($this->attr['id']); }
    public function owner         () { return $this->attr['owner']; }
    public function prefix        () { return $this->attr['prefix']; }
    public function first         () { return intval($this->attr['first']); }
    public function last          () { return intval($this->attr['last']); }
    public function num_in_use    () { return intval($this->num_in_use); }
    public function num_available () { return $this->last() - $this->first() - $this->num_in_use(); }
    public function next_available() {

        // Nothing has been allocated so far/ If so then return the first
        // number in the range.
        //
        if(is_null($this->recently_allocated())) return $this->first();

        // If the last number is within the range and if we still have at least
        // one number in that range then return the one.
        //
        $last_number = $this->recently_allocated();
        if(( $this->first() <= $last_number ) && ( $last_number < $this->last())) return $last_number + 1;

        // Sorry, nothing is available
        //
        return null;
    }
    public function recently_allocated          () { return is_null($this->last_allocation) ? null :           intval($this->last_allocation['jobnumber'       ]); }
    public function recent_allocation_uid       () { return is_null($this->last_allocation) ? null :                  $this->last_allocation['allocated_by_uid'] ; }
    public function recent_allocation_time      () { return is_null($this->last_allocation) ? null : LusiTime::from64($this->last_allocation['allocated_time'  ]); }
    public function recent_allocation_project_id() { return is_null($this->last_allocation) ? null :           intval($this->last_allocation['project_id'      ]); }
    public function recent_allocation_project   () {
        return is_null($this->last_allocation) ? null : $this->neocaptar->find_project_by_id(intval($this->last_allocation['project_id']));
    }

    public function find_jobnumber($project_id) {
        $sql    = "SELECT jobnumber FROM {$this->connection->database}.jobnumber_allocation WHERE jobnumber_id='{$this->id()}' AND project_id={$project_id}";
        $result = $this->connection->query($sql);
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return $attr['jobnumber'];
    }

    public function jobnumbers() {
        $list = array();
        //$project_id = $this->recent_allocation_project_id();
        //if(!is_null($project_id)) {
            //$sql    = "SELECT * FROM {$this->connection->database}.jobnumber_allocation WHERE jobnumber_id='{$this->id()}' AND project_id={$project_id}";
            $sql    = "SELECT * FROM {$this->connection->database}.jobnumber_allocation WHERE jobnumber_id='{$this->id()}'";
            $result = $this->connection->query($sql);
            for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
                array_push (
                    $list,
                    new NeoCaptarJobNumber (
                        $this->connection,
                        $this,
                        mysql_fetch_array( $result, MYSQL_ASSOC )));
        //}
        return $list;

    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function update_self($first,$last,$prefix) {
        return $this->neocaptar->update_jobnumber($this,$first,$last,$prefix);
    }
}
?>
