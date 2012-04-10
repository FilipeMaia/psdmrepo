<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class SysMonBeamTimeComment is an abstraction for comments stored
 * in the beam-time usage monitoring database.
 *
 * @author gapon
 */
class SysMonBeamTimeComment {

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
    public function sysmon        () { return                   $this->$sysmon; }
    public function gap_begin_time() { return LusiTime::from64( $this->attr['gap_begin_time'] ); }
    public function instr_name    () { return                   $this->attr['instr_name']; }
    public function post_time     () { return LusiTime::from64( $this->attr['post_time'] ); }
    public function posted_by_uid () { return                   $this->attr['posted_by_uid']; }
    public function comment       () { return                   $this->attr['comment']; }
    public function system        () { return                   $this->attr['system']; }
}
?>
