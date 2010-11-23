<?php

namespace RegDB;

require_once( 'regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

/**
 * Class RegDBFile an abstraction for a data file open/created by the DAQ system.
 *
 * @author gapon
 */
class RegDBFile {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    public function parent () {
        return $this->experiment; }

    public function run () {
        return (int)$this->attr['run']; }

    public function stream () {
        return (int)$this->attr['stream']; }

    public function chunk () {
        return (int)$this->attr['chunk']; }
        
    public function open_time () {
        return LusiTime::from64( $this->attr['open'] ); }
}
?>
