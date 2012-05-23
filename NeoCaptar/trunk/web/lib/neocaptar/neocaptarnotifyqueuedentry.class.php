<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarNotifyQueuedEntry is an abstraction for a notification event waiting
 * in the pending queue.
 *
 * @author gapon
 */
class NeoCaptarNotifyQueuedEntry {

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
    public function neocaptar()      { return $this->neocaptar; }
    public function id()             { return $this->attr['id']; }
    public function event_type()     { return $this->neocaptar()->find_notify_event_type_by_id($this->attr['event_type_id']); }
    public function event_time()     { return LusiTime::from64($this->attr['event_time']); }
    public function originator_uid() { return $this->attr['event_originator_uid']; }
    public function recipient_uid()  { return $this->attr['recipient_uid']; }
}
?>
