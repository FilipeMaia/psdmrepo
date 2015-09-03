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
    
    public function recipient_role_name() {
        switch($this->recipient()) {
            case 'ADMINISTRATOR': return 'Administrator';
            case 'PROJMANAGER':   return 'Project Manager';
            case 'OTHER':         return 'Other User';
        }
        return 'Unknown';
    }
}
?>
