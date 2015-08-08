<?php

namespace SysMon ;

require_once 'sysmon.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

class SysMonPsanaMonPlot {

    /* Data members
     */
    private $sysmon ;

    public $attr;

    /** Constructor
     */
    public function __construct ($sysmon, $attr) {
        $this->sysmon = $sysmon ;
        $this->attr   = $attr ;
    }

    /* Trivial aAccessors
     */
    public function sysmon      () { return                  $this->sysmon ; }
    public function id          () { return           intval($this->attr['id']) ; }
    public function exper_id    () { return           intval($this->attr['exper_id']); }
    public function name        () { return             trim($this->attr['name']) ; }
    public function type        () { return             trim($this->attr['type']) ; }
    public function descr       () { return             trim($this->attr['descr']) ; }
    public function update_time () { return LusiTime::from64($this->attr['update_time']) ; }
    public function update_uid  () { return             trim($this->attr['update_uid']) ; }
    public function data_size   () { return           intval($this->attr['data_size']) ; }

    /**
     * Fetch the image data
     *
     * @return string
     * @throws SysMonException
     */
    public function data () {
        $result = $this->sysmon()->query("SELECT data FROM {$this->sysmon()->database}.psanamon_plot_m WHERE id={$this->id()}") ;
        if (mysql_numrows($result) != 1) throw new SysMonException (
            __class__.'::'.__METHOD__ ,
            'inconsistent result returned from the database. Wrong schema?') ;
        return mysql_result($result, 0) ;
    }
}
?>
