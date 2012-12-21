<?php

namespace Irep ;

require_once 'irep.inc.php' ;

/**
 * Class IrepNotifyEventType is an abstraction for mnotification types stored
 * in the database.
 *
 * @author gapon
 */
class IrepNotifyEventType {

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
    public function irep        () { return $this->irep ; }
    public function id          () { return intval($this->attr['id']) ; }
    public function recipient   () { return $this->attr['recipient'] ; }
    public function name        () { return $this->attr['name'] ; }
    public function scope       () { return $this->attr['scope'] ; }
    public function description () { return $this->attr['description'] ; }
    
    public function recipient_role_name () {
        switch ($this->recipient()) {
            case 'ADMINISTRATOR' : return 'Administrator' ;
            case 'EDITOR'        : return 'Editor' ;
            case 'OTHER'         : return 'Other User' ;
        }
        return 'Unknown' ;
    }
}
?>
