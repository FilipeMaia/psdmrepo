<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;
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

    /* Constructor
     */
    public function __construct ($connection, $neocaptar, $attr) {
        $this->connection = $connection;
        $this->neocaptar = $neocaptar;
        $this->attr = $attr;
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

    /*
     * ==================================
     *   NON-TRIVIAL DATABASE SELECTORS
     * ==================================
     */
    public function num_in_use() {
        $result = $this->connection->query("SELECT COUNT(*) AS 'num_in_use' FROM {$this->connection->database}.cablenumber_allocation WHERE cablenumber_id={$this->id()}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return intval($attr['num_in_use']);
    }
    public function num_available () {
        $num = $this->last() - $this->first() - $this->num_in_use();
        return $num < 0 ? 0 : $num;
    }
    public function next_available() {

        if( !$this->num_available()) return null;   // no more numbers available in this range

        // Scan the range for the smallest availble number (if any)
        //
        $result = $this->connection->query("SELECT cablenumber FROM {$this->connection->database}.cablenumber_allocation WHERE cablenumber_id={$this->id()} ORDER BY cablenumber ASC");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return $this->first();
        $prev = null;
        for( $i = 0; $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            $cablenumber = intval($attr['cablenumber']);
            if(is_null($prev)) {
                if($cablenumber != $this->first()) return $this->first();   // a hole found at the beginning of the range
            } else {
                if($cablenumber != $prev + 1 ) return $prev + 1;            // a hole between the previous and this one
            }
            $prev = $cablenumber;
        }
        // Our last chance is a hole between the end of the sequence and the last allowed
        // number in the range.
        //
        return $prev == $this->last() ? null : $prev + 1;
    }
    public function recent_allocation() {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.cablenumber_allocation WHERE cablenumber_id={$this->id()} ORDER BY allocated_time DESC LIMIT 1");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        $attr['allocated_time'] = LusiTime::from64($attr['allocated_time']);
        return $attr;
    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */

    /**
     * The method (if successfull) return a number from a range asociated
     * with this allocation. The method should return null if no more numbers
     * are left in the range.
     * 
     * NOTE: If the cable was already registered then it should get a previously
     *       allocated number.
     *
     * @param integer $cable_id
     * @return integer
     */
    public function allocate($cable_id) {

        // Check if a number was previously allocated for the the cable,
        // and return that number of so.
        //
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.cablenumber_allocation WHERE cablenumber_id={$this->id()} AND cable_id={$cable_id}");
        $nrows = mysql_numrows( $result );
        switch( $nrows ) {
        case 0:
            $cablenumber = $this->next_available();
            if( is_null($cablenumber)) return null;
            $time_64     = LusiTime::now()->to64();
            $uid_escaped = $this->connection->escape_string( trim( AuthDB::instance()->authName()));
            $this->connection->query("INSERT {$this->connection->database}.cablenumber_allocation VALUES ({$this->id()},{$cablenumber},{$cable_id},{$time_64},'{$uid_escaped}')");
            return $cablenumber;
        case 1:
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            return intval($attr['cablenumber']);
        default:
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        }
        return null;
    }
    public function update_self($first,$last,$prefix) {
        return $this->neocaptar->update_cablenumber($this,$first,$last,$prefix);
    }
}
?>
