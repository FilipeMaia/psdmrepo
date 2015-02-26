<?php

namespace RegDB ;

require_once 'regdb.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class RegDBRun an abstraction for experimental runs.
 *
 * @author gapon
 */
class RegDBRun {

    private $connection ;
    private $experiment ;

    public $attr ;

    public function __construct ($connection, $experiment, $attr) {
        $this->connection = $connection ;
        $this->experiment = $experiment ;
        $this->attr = $attr ;
    }

    public function parent       () { return $this->experiment ; }
    public function num          () { return $this->attr['num'] ; }
    public function request_time () { return LusiTime::from64($this->attr['request_time']) ; }
}
?>
