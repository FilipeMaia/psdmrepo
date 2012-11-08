<?php

namespace DataPortal;

require_once 'dataportal.inc.php' ;

use LusiTime\LusiTime;

/**
 * Class ExpTimeMonBeamTimeRun is an abstraction for runs stored
 * in the beam-time usage monitoring database.
 *
 * @author gapon
 */
class ExpTimeMonBeamTimeRun {

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
    public function exper_id  () { return intval(           $this->attr['exper_id']); }
    public function runnum    () { return intval(           $this->attr['runnum']); }
    public function exper_name() { return                   $this->attr['exper_name']; }
    public function instr_name() { return                   $this->attr['instr_name']; }
}
?>
