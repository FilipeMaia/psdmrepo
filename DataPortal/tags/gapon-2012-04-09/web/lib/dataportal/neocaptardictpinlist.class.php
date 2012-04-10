<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictCable is an abstraction for pinlist types stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictPinlist {

   /* Data members
     */
    private $connection;
    private $connector;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $connector, $attr ) {
        $this->connection = $connection;
        $this->connector = $connector;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function connector    () { return $this->connector; }
    public function id           () { return $this->attr['id']; }
    public function name         () { return $this->attr['name']; }
    public function documentation() { return $this->attr['documentation']; }
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function created_uid  () { return $this->attr['created_uid']; }
}
?>
