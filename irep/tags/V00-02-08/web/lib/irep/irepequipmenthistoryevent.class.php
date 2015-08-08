<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'authdb/authdb.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepEquipmentHistoryEvent is an abstraction for equipment history events.
 *
 * @author gapon
 */
class IrepEquipmentHistoryEvent {

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
    public function irep        () { return                       $this->irep ; }
    public function id          () { return                intval($this->attr['id']) ; }
    public function equipment_id() { return                intval($this->attr['equipment_id']) ; }
    public function event_time  () { return LusiTime::from64(trim($this->attr['event_time'])) ; }
    public function event_uid   () { return                  trim($this->attr['event_uid']) ; }
    public function event       () { return                  trim($this->attr['event']) ; }

    /* Comments
     */
    public function comments () {
        $list = array () ;
        $result = $this->irep()->query("SELECT comment FROM {$this->irep()->database}.equipment_history_comments WHERE equipment_history_id={$this->id()}") ;
        for ($i = 0, $nrows = mysql_numrows( $result ) ; $i < $nrows ; $i++) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC) ;
            array_push (
                $list,
                $attr['comment']) ;
        }
        return $list ;
    }
}
?>
