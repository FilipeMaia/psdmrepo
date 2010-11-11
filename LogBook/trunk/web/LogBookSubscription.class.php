<?php

namespace LogBook;

require_once( 'LogBook.inc.php' );
require_once( 'LusiTime/LusiTime.inc.php' );

use LusiTime\LusiTime;

class LogBookSubscription {

    /* Data members
     */
    private $connection;
    private $experiment;
    private $attr;

    /* Constructor
     */
    public function __construct( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr       = $attr;
    }

    /* Accessors
     */
    public function parent () { return $this->experiment; }

    public function id () { return $this->attr['id']; }
    public function subscriber() { return $this->attr['subscriber']; }
    public function address   () { return $this->attr['address']; }
    public function subscribed_time() { return LusiTime::from64( $this->attr['subscribed_time'] ); }
    public function subscribed_host() { return $this->attr['subscribed_host']; }
}
?>
