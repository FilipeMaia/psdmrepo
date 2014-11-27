<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;
use LusiTime\LusiTime;

/**
 * Class NeoCaptarCableNumberPrefix is an abstraction for cable number prefix stored
 * in the database.
 *
 * @author gapon
 */
class NeoCaptarCableNumberPrefix {

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
    public function neocaptar() { return $this->neocaptar; }
    public function id       () { return intval($this->attr['id']); }
    public function name     () { return $this->attr['prefix']; }

    /*
     * ==================================
     *   NON-TRIVIAL DATABASE SELECTORS
     * ==================================
     */
    public function locations() {
        $list = array();
        $result = $this->connection->query("SELECT location FROM {$this->connection->database}.cablenumber_location WHERE prefix_id={$this->id()} ORDER BY location");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push( $list, $attr['location'] );
        }
        return $list;
    }
    public function ranges() {
        $list = array();
        $result = $this->connection->query("SELECT id,first,last FROM {$this->connection->database}.cablenumber_range WHERE prefix_id={$this->id()} ORDER BY first");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            $range = array (
                'id'    => intval($attr['id']),
                'first' => intval($attr['first']),
                'last'  => intval($attr['last'] ));
            $range['available'] = $this->find_available_numbers($range);
            array_push($list, $range);
        }
        return $list;
    }
    public function find_range_for($cablenumber) {
        $result = $this->connection->query("SELECT id,first,last FROM {$this->connection->database}.cablenumber_range WHERE prefix_id={$this->id()} AND first <= {$cablenumber} AND {$cablenumber} <= last");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt.");
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC ); 
        $range = array (
            'id'    => intval($attr['id']),
            'first' => intval($attr['first']),
            'last'  => intval($attr['last'] ));
        $range['available'] = count($this->find_available_numbers($range));
        return $range;
    }
    private function find_available_numbers($range) {
        $list      = array();
        $range_id  = $range['id'];   
        $first     = $range['first'];   
        $last      = $range['last'];
        $allocated = array();
        $result = $this->connection->query("SELECT cablenumber FROM {$this->connection->database}.cablenumber_allocated WHERE range_id={$range_id} AND {$first} <= cablenumber AND cablenumber <= {$last} ORDER BY cablenumber");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push($allocated, intval($attr['cablenumber']));
        }
        return array_diff(range($first, $last), $allocated);
    }
    public function find_cablenumber_for($cable_id) {
        foreach( $this->ranges() as $range ) {
            $range_id = $range['id'];
            $sql = "SELECT cablenumber,range_id FROM {$this->connection->database}.cablenumber_allocated WHERE range_id={$range_id} AND cable_id={$cable_id}";
            $result = $this->connection->query($sql);
            $nrows = mysql_numrows( $result );
            if( $nrows == 0 ) continue;
            if( $nrows != 1 )
                throw new NeoCaptarException (
                    __METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}");
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            return array(
                'cable_id'    => $cable_id,
                'cablenumber' => sprintf("%2s%05d", $this->name(), intval($attr['cablenumber'])),
                'range_id'    => intval($attr['range_id']),
                'number'      => intval($attr['cablenumber']),
                'prefix'      => $this->name()
            );
        }
        return null;
    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function add_location($name) {
        $name_escaped = $this->connection->escape_string(trim($name));
        $this->connection->query("INSERT {$this->connection->database}.cablenumber_location VALUES ({$this->id()},'{$name_escaped}')");
    }
    public function add_range($first,$last) {
        $this->connection->query("INSERT {$this->connection->database}.cablenumber_range VALUES (NULL,{$this->id()},{$first},{$last})");
    }
    public function update_range($range_id,$first,$last) {
        $this->connection->query("UPDATE {$this->connection->database}.cablenumber_range SET first={$first}, last={$last} WHERE id={$range_id} AND prefix_id={$this->id()}");
    }
    public function delete_range($range_id) {
        $this->connection->query("DELETE FROM {$this->connection->database}.cablenumber_range WHERE id={$range_id} AND prefix_id={$this->id()}");
    }
    public function synchronize_cable($cable, $uid) {
        $range_id = $cable['range_id'];   
        $number   = $cable['number'];
        $cable_id = $cable['cable_id'];
        $allocated_time_64 = LusiTime::now()->to64();
        $this->connection->query("INSERT {$this->connection->database}.cablenumber_allocated VALUES ({$range_id},{$number},{$cable_id},{$allocated_time_64},'{$uid}')");
    }
    public function allocate_cable_number($cable_id, $uid) {

        $cable_id = intval($cable_id);

        // First check if a number has already been allocated for this
        // cable in one of this prefix's ranges.
        //
        $cablenumber = $this->find_cablenumber_for($cable_id);
        if( !is_null($cablenumber)) return $cablenumber['cablenumber'];

        // Otherwise go for real allocation
        //
        foreach( $this->ranges() as $range ) {
            foreach( $this->find_available_numbers($range) as $number ) {
                // Note that the code below will intercept an attempt to create duplicate
                // cable name. If a conflict will be detected then the code will return null
                // to indicate a proble. Then it's up to the caller how to deal with this
                // situation. Usually, a solution is to commit the current transaction,
                // start another one and make a read attempt for the desired cable within
                // that (new) transaction.
                //
                try {
                    $this->synchronize_cable (
                        array(
                            'range_id' => $range['id'],
                            'number'   => $number,
                            'cable_id' => intval($cable_id)),
                        $uid
                    );
                    return sprintf("%3s%05d", $this->name(), $number);
                } catch( NeoCaptarException $e ) {
                    if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) continue;
                    throw $e;
                }
            }
        }
        throw new NeoCaptarException (
            __METHOD__, "ran out of cable numbers in prefix {$this->name()}");
    }
    public function free_cable($cable) {
        $range_id = $cable['range_id'];   
        $number   = $cable['number'];
        $cable_id = $cable['cable_id'];
        $this->connection->query("DELETE FROM {$this->connection->database}.cablenumber_allocated WHERE range_id={$range_id} AND cablenumber={$number} AND cable_id={$cable_id}");
    }
}
?>
