<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarUser is an abstraction for instrs stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarUser {

    /* Data members
     */
    private $connection;
    private $neocaptar;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $neocaptar, $attr ) {
        $this->connection = $connection;
        $this->neocaptar = $neocaptar;
        $this->attr = $attr;
    }

    /* Properties
     */
    public function neocaptar()         { return $this->neocaptar; }
    public function uid()               { return $this->attr['uid']; }
    public function role()              { return strtolower ($this->attr['role']); }
    public function name()              { return $this->attr['name']; }
    public function added_time()        { return LusiTime::from64( $this->attr['added_time'] ); }
    public function added_uid()         { return $this->attr['added_uid']; }
    public function last_active_time()  { return $this->attr['last_active_time'] == '' ? '' : LusiTime::from64( $this->attr['last_active_time'] ); }

    /* Helper functions
     */
    public function is_administrator() { return 'administrator' == $this->role(); }
    public function is_projmanager  () { return 'projmanager'   == $this->role(); }
    public function is_other        () { return 'other'         == $this->role(); }
}
?>
