<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

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
    public function __construct ($connection, $cable, $attr) {
        parent::__construct($connection,'cable',$cable->id(),$attr);
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
