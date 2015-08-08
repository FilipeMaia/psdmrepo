<?php

namespace Irep ;

require_once 'irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class IrepEquipmentTag is an abstraction for equipment tags.
 *
 * @author gapon
 */
class IrepEquipmentTag {

    /* Data members
     */
    private $equipment ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($equipment, $attr) {
        $this->equipment = $equipment ;
        $this->attr = $attr ;
    }

    /* Properties
     */
    public function equipment () { return $this->equipment ; }

    public function id   () { return intval($this->attr['id']) ; }
    public function name () { return   trim($this->attr['name']) ; }

    public function create_time () { return LusiTime::from64(intval($this->attr['create_time'])) ; }
    public function create_uid  () { return                    trim($this->attr['create_uid']) ; }
}
?>
