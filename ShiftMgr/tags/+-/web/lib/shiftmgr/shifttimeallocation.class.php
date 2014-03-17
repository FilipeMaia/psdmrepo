<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'dataportal/dataportal.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;
use DataPortal\ExpTimeMon ;

/**
 * Class ShiftTimeAllocation is an abstraction for shift time allocations.
 *
 * @author gapon
 */
class ShiftTimeAllocation {

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
    public function duration_min () { return intval($this->attr['duration_min']) ; }
    public function comments     () { return        $this->attr['comments'] ; }

    /**
     * Return all history events in a scope of this time allocation
     *
     * NOTE: The elements will be unsorted.
     *
     * @return array(ShiftHistoryEvent)
     */
    public function history () {
        $history = array();
        $result = $this->shift()->shiftmgr()->query("SELECT * FROM {$this->shift()->shiftmgr()->database}.shift_time_history WHERE time_id={$this->id()} ORDER BY modified_time DESC") ;
        for ($i = 0, $nrows = mysql_numrows($result) ; $i < $nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            array_push (
                $history ,
                new ShiftHistoryEvent (
                    $this->shift() ,    // shift-specific informaton willbe extracted from here
                    'TIME' ,            // scope
                    $this->name() ,     // scope2
                    $attr               // event-specific information will be extracted from here
                )
            ) ;
        }
        return $history ;
    }
}
?>
