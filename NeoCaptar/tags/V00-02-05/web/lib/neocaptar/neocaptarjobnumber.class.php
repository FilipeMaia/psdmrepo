<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarJobNumber is an abstraction for cable number stored
 * in the database.
 *
 * @author gapon
 */
class NeoCaptarJobNumber {

   /* Data members
     */
    private $connection;
    private $allocation;

    private $attr;

    /* Constructor
     */
    public function __construct ($connection, $allocation, $attr) {
        $this->connection = $connection;
        $this->allocation = $allocation;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function allocation      () { return $this->allocation; }
    public function jobnumber_id    () { return intval($this->attr['jobnumber_id']); }
    public function project_id      () { return intval($this->attr['project_id']); }
    public function jobnumber       () { return intval($this->attr['jobnumber']); }
    public function jobnumber_name  () { return sprintf("%3s%03d", $this->allocation()->prefix(), $this->jobnumber()); }
    public function num_cables      () { return count($this->allocation()->neocaptar()->find_cables_by_jobnumber( $this->jobnumber_name())); }
    public function allocated_time  () { return LusiTime::from64( $this->attr['allocated_time'] ); }
    public function allocated_by_uid() { return $this->attr['allocated_by_uid']; }
}
?>
