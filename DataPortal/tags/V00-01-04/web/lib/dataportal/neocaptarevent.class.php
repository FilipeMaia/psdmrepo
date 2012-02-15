<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarEvent is a base class for history events.
 *
 * @author gapon
 */
class NeoCaptarEvent {

   /* Data members
     */
    private $scope;
    private $scope_id;
    private $attr;

    /* Constructor
     */
    public function __construct ($scope, $scope_id, $attr) {
        $this->scope    = $scope;
        $this->scope_id = $scope_id;
        $this->attr     = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function scope      () { return $this->scope; }
    public function scope_id   () { return $this->scope_id; }
    public function event_time () { return LusiTime::from64( $this->attr['event_time'] ); }
    public function event_uid  () { return $this->attr['event_uid']; }
    public function event      () { return $this->attr['event']; }
}
?>
