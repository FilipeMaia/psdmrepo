<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictRack is an abstraction for racks stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictRack {

   /* Data members
     */
    private $connection;
    private $location;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $location, $attr ) {
        $this->connection = $connection;
        $this->location = $location;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function location     () { return $this->location; }
    public function id           () { return $this->attr['id']; }
    public function name         () { return $this->attr['name']; }
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function created_uid  () { return $this->attr['created_uid']; }
}
?>
