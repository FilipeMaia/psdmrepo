<?php

namespace DataPortal;

require_once 'dataportal.inc.php' ;

use LusiTime\LusiTime;

/**
 * Class ExpTimeMonBeamTimeGap is an abstraction for gaps stored
 * in the beam-time usage monitoring database.
 *
 * @author gapon
 */
class ExpTimeMonBeamTimeGap {

   /* Data members
     */
    private $exptimemon;

    public $attr;

    /* Constructor
     */
    public function __construct ( $exptimemon, $attr ) {
        $this->exptimemon = $exptimemon;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function exptimemon() { return                   $this->$exptimemon; }
    public function begin_time() { return LusiTime::from64( $this->attr['begin_time'] ); }
    public function end_time  () { return LusiTime::from64( $this->attr['end_time'] ); }
    public function instr_name() { return                   $this->attr['instr_name']; }
}
?>
