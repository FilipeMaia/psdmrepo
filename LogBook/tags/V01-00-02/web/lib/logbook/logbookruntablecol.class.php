<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

class LogBookRunTable {

    /* Data members
     */
    private $logbook;
    private $run;

    private $attr;

    /** Constructor
     */
    public function __construct ( $logbook, $run, $attr ) {
        $this->logbook = $logbook;
        $this->run     = $run;
        $this->attr    = $attr;
    }

    /* Accessors
     */
    public function run           () { return                   $this->run; }
    public function id            () { return                   $this->attr['id']; }
    public function name          () { return                   $this->attr['name']; }
    public function description   () { return                   $this->attr['descr']; }
    public function is_private    () { return         'YES' === $this->attr['is_private']; }
    public function created_uid   () { return                   $this->attr['created_uid']; }
    public function created_time  () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function modified_uid  () { return                   $this->attr['modified_uid']; }
    public function modified_time () { return LusiTime::from64( $this->attr['modified_time'] ); }
}
?>
