<?php

namespace Irep ;

require_once 'irep.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepNotifyQueuedEntry is an abstraction for a notification event waiting
 * in the pending queue.
 *
 * @author gapon
 */
class IrepNotifyQueuedEntry {

    /* Data members
     */
    private $irep ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($irep, $attr) {
        $this->irep = $irep ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function irep           () { return $this->irep ; }
    public function id             () { return intval($this->attr['id']) ; }
    public function event_type     () { return $this->irep()->find_notify_event_type_by_id(intval($this->attr['event_type_id'])) ; }
    public function event_time     () { return LusiTime::from64($this->attr['event_time']) ; }
    public function originator_uid () { return $this->attr['event_originator_uid'] ; }
    public function recipient_uid  () { return $this->attr['recipient_uid'] ; }
    public function extra          () {

        switch ($this->event_type()->scope()) {

            case 'EQUIPMENT' :

                $result = $this->irep()->query("SELECT * FROM {$this->irep()->database}.notify_queue_cable WHERE notify_queue_id={$this->id()}") ;
                $nrows = mysql_numrows($result) ;
                if($nrows == 0) return null ;
                if($nrows != 1)
                    throw new IrepException (
                        __METHOD__, "inconsistent result returned by the query. Database may be corrupt.") ;

                return mysql_fetch_array ($result, MYSQL_ASSOC) ;
        }
        return null ;
    }
}
?>
