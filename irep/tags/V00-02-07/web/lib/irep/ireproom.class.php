<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepRoom is an abstraction for rooms.
 *
 * @author gapon
 */
class IrepRoom {

    /* Data members
     */
    private $location ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($location, $attr) {
        $this->location = $location ;
        $this->attr     = $attr ;
    }

    /* Properties
     */
    public function location     () { return                       $this->location ; }
    public function id           () { return                intval($this->attr['id']) ; }
    public function name         () { return                  trim($this->attr['name']) ; }
    public function created_time () { return LusiTime::from64(trim($this->attr['created_time'])) ; }
    public function created_uid  () { return                  trim($this->attr['created_uid']) ; }
}
?>
