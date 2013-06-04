<?php

namespace ShiftMgr ;

require_once 'shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class ShiftMgrShift is an abstraction for shifts.
 *
 * @author gapon
 */
class ShiftMgrShift {

    // Data members

    private $manager ;

    public $attr ;

    // Constructor
    public function __construct ($manager, $attr) {
        $this->manager = $manager ;
        $this->attr    = $attr ;
    }

    // Getter methods
    public function manager() {
      return $this->manager;
    }

    public function id() {
      return $this->attr['id'];
    }

    public function username() {
      return trim($this->attr['username']);
    }

    public function hutch() {
      return trim($this->attr['hutch']);
    }

    public function start_time() {
      return new LusiTime(intval(trim($this->attr['start_time'])));
    }

    public function last_modified_time() {
      return new LusiTime(intval(trim($this->attr['last_modified_time'])));
    }

    public function end_time() {
      if ($this->attr['end_time']) {
        return new LusiTime(intval(trim($this->attr['end_time'])));
      } else {
        return null;
      }
    }

    public function stopper_out() {
      return $this->attr['stopper_out'];
    }

    public function door_open() {
      return $this->attr['door_open'];
    }

    public function total_shots() {
      return $this->attr['total_shots'];
    }

    public function other_notes() {
      if ($this->attr['other_notes']) {
        return $this->attr['other_notes'];
      } else {
        return "";
      }
    }
}
?>
