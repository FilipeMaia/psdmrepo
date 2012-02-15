<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

/**
 * Class NeoCaptarCableEvent is an abstraction for cables.
 *
 * @author gapon
 */
class NeoCaptarCableEvent extends NeoCaptarEvent {

   /* Data members
     */
    private $cable;

    /* Constructor
     */
    public function __construct ($cable, $attr) {
        parent::__construct('cable',$cable->id(),$attr);
        $this->cable = $cable;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function cable () { return $this->cable; }
}
?>
