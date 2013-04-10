<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class ShiftMgrShift is an abstraction for shifts.
 *
 * @author gapon
 */
class ShiftMgrShift {

    // Data members

    private $manager ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($manager, $attr) {
        $this->manager = $manager ;
        $this->attr    = $attr ;
    }

    /* Properties
     */
    public function manager () { return $this->manager ; }

    public function id              () { return                   intval($this->attr['id']) ; }
    public function instrument_name () { return                     trim($this->attr['instrument']) ; }
    public function begin_time      () { return new LusiTime(intval(trim($this->attr['begin_time']))) ; }
    public function end_time        () {

        // Note that the the end time is optional.

        if (!$this->attr['end_time']) return null ;
        return new LusiTime(intval(trim($this->attr['end_time']))) ;
    }

    public function is_closed () { return !is_null($this->end_time()) ; }

    /**
     * Close the shift now
     *
     * ATTENTION: this operation may change the state of teh object.
     *
     * @throws ShiftMgrException
     */
    public function close () {

        if ($this->is_closed())
            throw new ShiftMgrException (
                __class__.'::'.__METHOD__ ,
                "the shift id={$this->id()} is already closed.") ;

        $end_time_sec = LusiTime::now()->sec ;

        $this->manager()->query (
            "UPDATE {$this->manager()->database}.shift SET end_time={$end_time_sec} WHERE id={$this->id()}") ;

        $this->attr['end_time'] = $end_time_sec ;
    }
}
?>
