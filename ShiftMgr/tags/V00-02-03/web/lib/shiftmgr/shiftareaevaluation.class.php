<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'dataportal/dataportal.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;
use DataPortal\ExpTimeMon ;

/**
 * Class ShiftAreaEvaluation is an abstraction for shift area evaluations.
 *
 * @author gapon
 */
class ShiftAreaEvaluation {

    // Data members

    private $shift ;

    public $attr ;

    public function __construct ($shift, $attr) {
        $this->shift = $shift ;
        $this->attr  = $attr ;
    }

    // Accessors

    public function shift        () { return        $this->shift ; }
    public function id           () { return intval($this->attr['id']) ; }
    public function name         () { return   trim($this->attr['name']) ; }
    public function problems     () { return        $this->attr['problems'] ? 1 : 0 ; }
    public function downtime_min () { return intval($this->attr['downtime_min']) ; }
    public function comments     () { return        $this->attr['comments'] ; }
    
    /**
     * Return all history events in a scope of this area
     * 
     * NOTE: The elements will be unsorted.
     *
     * @return array(ShiftHistoryEvent)
     */
    public function history () {
        $history = array();
        $result = $this->shift()->shiftmgr()->query("SELECT * FROM {$this->shift()->shiftmgr()->database}.shift_area_history WHERE area_id={$this->id()} ORDER BY modified_time DESC") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            array_push (
                $history ,
                new ShiftHistoryEvent (
                    $this->shift() ,    // shift-specific informaton willbe extracted from here
                    'AREA' ,            // scope
                    $this->name() ,     // scope2
                    $attr               // event-specific information will be extracted from here
                )
            ) ;
        }
        return $history ;
    }
}
?>
