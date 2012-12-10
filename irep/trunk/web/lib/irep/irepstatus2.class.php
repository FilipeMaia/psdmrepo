<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepStatus2 is an abstraction for sub-statuses.
 *
 * @author gapon
 */
class IrepStatus2 {

    /* Data members
     */
    private $status ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($status, $attr) {
        $this->status = $status ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function status       () { return                       $this->status ; }
    public function id           () { return                intval($this->attr['id']) ; }
    public function name         () { return                  trim($this->attr['name']) ; }
    public function is_locked    () { return                       $this->attr['is_locked'] == 'YES' ; }
    public function created_time () { return LusiTime::from64(trim($this->attr['created_time'])) ; }
    public function created_uid  () { return                  trim($this->attr['created_uid']) ; }
}
?>
