<?php

namespace LogBook;

require_once( 'LogBook.inc.php' );
require_once( 'LusiTime/LusiTime.inc.php' );

use LusiTime\LusiTime;

class LogBookRunVal {

    /* Data members
     */
    private $connection;
    private $run;
    private $name;
    private $type;

    public $attr;

    /* Constructor
     */
    public function __construct( $connection, $run, $name, $type, $attr ) {
        $this->connection = $connection;
        $this->run  = $run;
        $this->name = $name;
        $this->type = $type;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent  () { return $this->run; }
    public function run_id  () { return $this->attr['run_id']; }
    public function param_id() { return $this->attr['param_id']; }
    public function name    () { return $this->name; }
    public function type    () { return $this->type; }
    public function source  () { return $this->attr['source']; }
    public function updated () { return LusiTime::from64( $this->attr['updated'] ); }
    public function value   () { return $this->attr['val']; }
}
?>
