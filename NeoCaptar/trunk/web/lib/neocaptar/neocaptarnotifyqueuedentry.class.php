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

    public function extra() {

        switch($this->event_type()->scope()) {

            case 'CABLE':

                $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_queue_cable WHERE notify_queue_id={$this->id()}");
                $nrows = mysql_numrows( $result );
                if( $nrows == 0 ) return null;
                if( $nrows != 1 )
                    throw new NeoCaptarException (
                        __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );

                $extra_parameters = mysql_fetch_array( $result, MYSQL_ASSOC );

                return $extra_parameters;

            case 'PROJECT':

                $result = $this->connection->query("SELECT project_id,project_title,project_owner_uid,project_due_time FROM {$this->connection->database}.notify_queue_project WHERE notify_queue_id={$this->id()}");
                $nrows = mysql_numrows( $result );
                if( $nrows == 0 ) return null;
                if( $nrows != 1 )
                    throw new NeoCaptarException (
                        __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );

                $extra_parameters = mysql_fetch_array( $result, MYSQL_ASSOC );
                $extra_parameters['project_due_time'] = LusiTime::from64($extra_parameters['project_due_time']);

                return $extra_parameters;
        }
        return null;
    }
}
?>
