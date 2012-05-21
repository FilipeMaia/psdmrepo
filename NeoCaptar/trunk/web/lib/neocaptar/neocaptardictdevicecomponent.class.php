<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictDeviceComponent is an abstraction for device regions stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictDeviceComponent {

   /* Data members
     */
    private $connection;
    private $region;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $region, $attr ) {
        $this->connection = $connection;
        $this->region = $region;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function region       () { return $this->region; }
    public function id           () { return $this->attr['id']; }
    public function name         () { return $this->attr['name']; }
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function created_uid  () { return $this->attr['created_uid']; }
}
?>
