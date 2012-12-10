<?php

namespace Irep ;

require_once 'irep.inc.php' ;

/**
 * Class IrepNotify is an abstraction for notification configurations stored
 * in the database.
 *
 * @author gapon
 */
class IrepNotify {

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
    public function irep       () { return $this->irep ; }
    public function id         () { return intval($this->attr['id']) ; }
    public function uid        () { return $this->attr['uid'] ; }
    public function event_type () { return $this->irep()->find_notify_event_type_by_id($this->attr['event_type_id']) ; }
    public function enabled    () { return 'ON' == $this->attr['enabled'] ; }
}
?>
