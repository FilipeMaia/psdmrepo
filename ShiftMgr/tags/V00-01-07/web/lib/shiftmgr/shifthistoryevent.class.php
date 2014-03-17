<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'dataportal/dataportal.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class ShiftHistoryEvent is an abstraction for history events.
 *
 * @author gapon
 */
class ShiftHistoryEvent {

    // Data members

    private $shift ;
    private $scope ;
    private $scope2 ;
    private $attr ;

    public function __construct ($shift, $scope, $scope2, $attr) {
        $this->shift  = $shift ;
        $this->scope  = $scope ;
        $this->scope2 = $scope2 ;
        $this->attr   = $attr ;
    }

    // Accessors

    public function shift         () { return $this->shift ; }
    public function modified_uid  () { return trim($this->attr['modified_uid']) ; }
    public function modified_time () { return LusiTime::from64(intval($this->attr['modified_time'])) ; }
    public function scope         () { return $this->scope ; }
    public function scope2        () { return $this->scope2 ; }
    public function operation     () { return $this->scope() == 'SHIFT' ? strtoupper(trim($this->attr['event'])) : 'MODIFY' ; }
    public function id            () { return intval($this->attr['id']) ; }
    public function parameter     () { return trim($this->attr['parameter']) ; }
    public function old_value     () { return trim($this->attr['old_value']) ; }
    public function new_value     () { return trim($this->attr['new_value']) ; }
}
?>
