<?php

namespace LogBook;

require_once( 'logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

class LogBookSubscription {

    /* Data members
     */
    private $logbook;
    private $experiment;
    private $attr;

    /* Constructor
     */
    public function __construct( $logbook, $experiment, $attr ) {
        $this->logbook    = $logbook;
        $this->experiment = $experiment;
        $this->attr       = $attr;
    }

    /* Accessors
     */
    public function parent          () { return $this->experiment; }
    public function id              () { return $this->attr['id']; }
    public function subscriber      () { return $this->attr['subscriber']; }
    public function address         () { return $this->attr['address']; }
    public function subscribed_time () { return LusiTime::from64( $this->attr['subscribed_time'] ); }
    public function subscribed_host () { return $this->attr['subscribed_host']; }
}
?>
