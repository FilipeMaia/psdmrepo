<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

class LogBookRunParam {

    /* Data members
     */
    private $logbook;
    private $experiment;

    public $attr;

    /** Constructor
     */
    public function __construct ( $logbook, $experiment, $attr ) {
        $this->logbook    = $logbook;
        $this->experiment = $experiment;
        $this->attr       = $attr;
    }

    /* Accessors
     */
    public function parent      () { return        $this->experiment; }
    public function id          () { return intval($this->attr['id']); }
    public function name        () { return        $this->attr['param']; }
    public function exper_id    () { return intval($this->attr['exper_id']); }
    public function type_name   () { return        $this->attr['type']; }
    public function description () { return   trim($this->attr['descr']); }
}
?>
