<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepLocation is an abstraction for locations.
 *
 * @author gapon
 */
class IrepLocation {

    /* Data members
     */
    private $irep ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($irep, $attr) {
        $this->irep = $irep ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function irep         () { return                       $this->irep ; }
    public function id           () { return                intval($this->attr['id']) ; }
    public function name         () { return                  trim($this->attr['name']) ; }
    public function created_time () { return LusiTime::from64(trim($this->attr['created_time'])) ; }
    public function created_uid  () { return                  trim($this->attr['created_uid']) ; }
}
?>
