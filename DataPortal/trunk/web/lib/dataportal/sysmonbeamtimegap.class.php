<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class SysMonBeamTimeGap is an abstraction for gaps stored
 * in the beam-time usage monitoring database.
 *
 * @author gapon
 */
class SysMonBeamTimeGap {

   /* Data members
     */
    private $sysmon;

    public $attr;

    /* Constructor
     */
    public function __construct ( $sysmon, $attr ) {
        $this->sysmon = $sysmon;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function sysmon    () { return                   $this->$sysmon; }
    public function begin_time() { return LusiTime::from64( $this->attr['begin_time'] ); }
    public function end_time  () { return LusiTime::from64( $this->attr['end_time'] ); }
}
?>
