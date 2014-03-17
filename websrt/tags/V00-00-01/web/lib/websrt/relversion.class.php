<?php

namespace websrt ;

require_once 'websrt/websrt.inc.php' ;

class RelVersion {
    public $major  = 0 ;
    public $middle = 0 ;
    public $minor  = 0 ;
    public function __construct($major, $middle, $minor) {
        $this->major  = $major ;
        $this->middle = $middle ;
        $this->minor  = $minor ;
    }
    public function as_string () { return "{$this->major}.{$this->middle}.{$this->minor}" ; }
    public function as_number () { return 100 * 100 * $this->major + 100 * $this->middle + $this->minor ; }

    public function less ($rhs) { return $this->as_number() < $rhs->as_number() ? true : false ; }

    /**
     * Return the array representation of the object suitable for converting into JSON.
     *
     * @return array
     */
    public function export2array () {
        return array (
            'major'      => $this->major ,
            'middle'     => $this->middle ,
            'minor'      => $this->minor ,
            'as_string'  => $this->as_string() ,
            'as_number'  => $this->as_number()
        ) ;
    }
}
?>