<?php

namespace SysMon ;

require_once 'sysmon.inc.php' ;

/**
 * An abstraction for the file migration delay events.
 */
class SysMonFMDelayEvent {

    // Object parameters

    private $sysmon = null ;

    // Public attributes

    public $name = null ;
    public $descr = null ;

    /**
     * Constructor
     * 
     * @param \SysMon\SysMon $sysmon
     * @param array $attr
     */
    public function __construct ($sysmon, $attr) {
        $this->sysmon = $sysmon ;
        $this->name   = trim($attr['name']) ;
        $this->descr  = trim($attr['descr']) ;
    }

    /**
     * Make a simple object which can be serialized into JSON or any other
     * external representation.
     *
     *    Attr  | Type   | Description
     *   -------+--------+-----------------------
     *    name  | string | the name of the event
     *    descr | string | its description
     */
    public function jsonSerialize () {
        $obj = new \stdClass ;
        $obj->name  = $this->name;
        $obj->descr = $this->descr ;
        return $obj ;
    }
}

?>

