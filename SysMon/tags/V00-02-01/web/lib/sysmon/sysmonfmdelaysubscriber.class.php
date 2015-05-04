<?php

namespace SysMon ;

require_once 'sysmon.inc.php' ;

require_once 'lusitime/lusitime.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use \LusiTime\LusiTime ;
use \RegDB\RegDB ;

if (PHP_VERSION_ID < 50400) {
    /*
     * This interface was formally introduced in PHP 5.4 for
     * better control over what gets serialized into the JSON format.
     */
    if (!interface_exists ('\SysMon\JsonSerializable', false)) {
        interface JsonSerializable {
            public function jsonSerialize () ;
        }
    }
}

/**
 * An abstraction for the file migration delay subscribers.
 */
class SysMonFMDelaySubscriber implements JsonSerializable  {

    // Object parameters

    private $sysmon = null ;

    // Public attributes

    public $uid   = null ;
    public $gecos = null ;
    public $subscribed_uid   = null ;
    public $subscribed_gecos = null ;
    public $subscribed_time  = null ;
    public $subscribed_host  = null ;
    public $instr     = null ;
    public $last_sec  = null ;
    public $delay_sec = null ;

    /**
     * Constructor
     * 
     * @param \SysMon\SysMon $sysmon
     * @param array $attr
     */
    public function __construct ($sysmon, $attr) {

        $this->sysmon = $sysmon ;

        $this->uid = trim($attr['uid']) ;

        RegDB::instance()->begin() ;
        $user = RegDB::instance()->find_user_account($this->uid) ;
        $this->gecos = $user ? $user['gecos'] : $this->uid ;

        $this->subscribed_uid = trim($attr['subscribed_uid']) ;
        $subscribed_user = RegDB::instance()->find_user_account($this->subscribed_uid) ;
        $this->subscribed_gecos = $subscribed_user ? $subscribed_user['gecos'] : $this->subscribed_uid ;

        $subscribed_time = LusiTime::from64(trim($attr['subscribed_time'])) ;
        $this->subscribed_time = new \stdClass ;
        $this->subscribed_time->sec  = $subscribed_time->sec ;
        $this->subscribed_time->nsec = $subscribed_time->nsec ;
        $this->subscribed_time->day  = $subscribed_time->toStringDay() ;
        $this->subscribed_time->hms  = $subscribed_time->toStringHMS() ;
        $this->subscribed_host = trim($attr['subscribed_host']) ;

        $this->instr     = is_null($attr['instr']) ? '' : strtoupper(trim($attr['instr'])) ;
        $this->last_sec  =  intval($attr['last_sec']) ;
        $this->delay_sec =  intval($attr['delay_sec']) ;
    }

    /**
     * Make a simple object which can be serialized into JSON or any other
     * external representation.
     *
     *    Attr            | Type     | Description
     *   -----------------+----------+----------------------------------------------------------------------------
     *    uid             | string   | user account of a person subscribed
     *    gecos           | string   | full user name fr the user account (if available)
     *    subscribed_uid  | string   | user account of a person who requested the subscription 
     *    subscribed_time | stdClass | time when the subscription was made:
     *      sec           | integer  | - the nuumber of seconds
     *      nsec          | integer  | - the nuumber of nanoseconds
     *      day           | string   | - the day like 2014-04-12
     *      hms           | string   | - the hour-minute-second like 20:23:05
     *    subscribed_host | string   | host (IP address or DNS name) name from which the operation was requested
     *    instr           | string   | the name of an instrument (optional)
     *    last_sec        | integer  | the number of last seconds to take into account
     *    delay_sec       | integer  | the minimum duration of delays to take into account
     *    events          | array    | events to which the subscriber is subscribed for
     * 
     * @return \stdClass
     */
    public function jsonSerialize () {

        $obj = new \stdClass ;

        $obj->uid   = $this->uid;
        $obj->gecos = $this->gecos ;

        $obj->subscribed_uid   = $this->subscribed_uid ;
        $obj->subscribed_gecos = $this->subscribed_gecos ;
        $obj->subscribed_time  = $this->subscribed_time ;
        $obj->subscribed_host  = $this->subscribed_host ;

        $obj->instr     = $this->instr ;
        $obj->last_sec  = $this->last_sec ;
        $obj->delay_sec = $this->delay_sec ;

        if (PHP_VERSION_ID < 50400) {
            /*
             * JSON-ready object serialization control is provided
             * through a special interface JsonSerializable as
             * of PHP 5.4. The method jsonSerialize will return \stdClass
             * object with members ready for the JSON serialization.
             * Until that we have to call this method explicitly.
             */
            $obj->events = array() ;
            foreach ($this->events() as $e)
                array_push($obj->events, $e->jsonSerialize()) ;
        } else {
            $obj->events = $this->events() ;
        }
        return $obj ;
    }
    
    /**
     * Return events this user is subscribed for
     *
     * @return array
     */
    public function events () {

        $list = array();

        $sql =
            "SELECT * FROM {$this->sysmon->database}.fm_delay_event" .
            "  WHERE name IN (" .
            "    SELECT event_name FROM {$this->sysmon->database}.fm_delay_event_subscriber" .
            "      WHERE subscriber_uid='{$this->uid}')".
            "  ORDER BY name" ;

        $result = $this->sysmon->query($sql) ;
        for ($nrows = mysql_numrows($result), $i = 0; $i < $nrows; $i++) {
            array_push (
                $list ,
                new SysMonFMDelayEvent (
                    $this->sysmon ,
                    mysql_fetch_array (
                        $result ,
                        MYSQL_ASSOC))) ;
        }
        return $list ;
    }

    // Managing subscriptions for specific events

    public function subscribe   ($event_name) { $this->subscribe_if(true,  $event_name) ; }
    public function unsubscribe ($event_name) { $this->subscribe_if(false, $event_name) ; }

    /**
     * Subscribe/unsubscribe for teh specified event
     *
     * @param boolean $on
     * @param string $event_name
     */
    public function subscribe_if ($on, $event_name) {
        $event_escaped = $this->sysmon->escape_string(trim($event_name)) ;
        $sql = $on ?
            "INSERT INTO {$this->sysmon->database}.fm_delay_event_subscriber" .
            "  VALUES('{$event_escaped}','{$this->uid}')" .
            "  ON DUPLICATE KEY UPDATE event_name='{$event_escaped}'" :

            "DELETE FROM {$this->sysmon->database}.fm_delay_event_subscriber" .
            "  WHERE event_name='{$event_escaped}' AND subscriber_uid='{$this->uid}'" ;

        $this->sysmon->query($sql) ;
    }
}

?>

