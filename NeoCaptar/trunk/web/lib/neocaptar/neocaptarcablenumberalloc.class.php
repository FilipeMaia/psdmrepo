<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarCableNumberAlloc is an abstraction for cable number allocations stored
 * in the database.
 *
 * @author gapon
 */
class NeoCaptarCableNumberAlloc {

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
    public function location      () { return $this->attr['location']; }
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
    public function recently_allocated    () { return is_null($this->last_allocation) ? null : intval($this->last_allocation['cablenumber']); }
    public function recent_allocation_uid () { return is_null($this->last_allocation) ? null : $this->last_allocation['allocated_by_uid']; }
    public function recent_allocation_time() { return is_null($this->last_allocation) ? null : LusiTime::from64( $this->last_allocation['allocated_time'] ); }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function update_self($first,$last,$prefix) {
        return $this->neocaptar->update_cablenumber($this,$first,$last,$prefix);
    }
}
?>
