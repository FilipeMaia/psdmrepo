<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarNotifyEventType is an abstraction for mnotification types stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarNotifyEventType {

    /* Data members
     */
    private $connection;
    private $neocaptar;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $neocaptar, $attr ) {
        $this->connection = $connection;
        $this->neocaptar = $neocaptar;
        $this->attr = $attr;
    }

    /* Properties
     */
    public function neocaptar()   { return $this->neocaptar; }
    public function id()          { return $this->attr['id']; }
    public function recipient()   { return $this->attr['recipient']; }
    public function name()        { return $this->attr['name']; }
    public function scope()       { return $this->attr['scope']; }
    public function description() { return $this->attr['description']; }

    /* Helper functions
     */
    public function is_administrator() { return 'ADMINISTRATOR' == $this->recipient_role(); }
    public function is_projmanager  () { return 'PROJMANAGER'   == $this->recipient_role(); }
    public function is_other        () { return 'OTHER'         == $this->recipient_role(); }
}
?>
