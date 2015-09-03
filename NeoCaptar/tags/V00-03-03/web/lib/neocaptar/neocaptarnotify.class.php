<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarNotify is an abstraction for notification configurations stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarNotify {

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
    public function neocaptar()   { return $this->neocaptar; }
    public function id()          { return $this->attr['id']; }
    public function uid()         { return $this->attr['uid']; }
    public function event_type()  { return $this->neocaptar()->find_notify_event_type_by_id($this->attr['event_type_id']); }
    public function enabled()     { return 'ON' == $this->attr['enabled']; }
}
?>
