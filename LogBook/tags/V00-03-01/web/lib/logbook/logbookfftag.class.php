<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookFFTag is an abstraction for tags associated
 * with free-form entries.
 *
 * @author gapon
 */
class LogBookFFTag {

    /* Data members
     */
    private $logbook;
    private $entry;

    public $attr;

    /* Constructor
     */
    public function __construct ( $logbook, $entry, $attr ) {
        $this->logbook = $logbook;
        $this->exntry  = $entry;
        $this->attr    = $attr;
    }

    public function parent       () { return $this->entry; }
    public function entry_hdr_id () { return $this->attr['hdr_id']; }
    public function tag          () { return $this->attr['tag']; }
    public function value        () { return $this->attr['value']; }
}
?>
