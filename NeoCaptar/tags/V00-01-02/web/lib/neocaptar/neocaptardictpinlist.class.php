<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictCable is an abstraction for pinlist types stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictPinlist {

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

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function neocaptar    () { return $this->neocaptar; }
    public function id           () { return $this->attr['id']; }
    public function name         () { return $this->attr['name']; }
    public function documentation() { return $this->attr['documentation']; }
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function created_uid  () { return $this->attr['created_uid']; }

    /*
     * ====================================
     *   DATABASE MODIFICATION OPERATIONS
     * ====================================
     */
    public function update($documentation) {
        $documentation_escaped = $this->connection->escape_string(trim($documentation));
        $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET documentation='{$documentation_escaped}' WHERE id={$this->id()}");
        $this->attr['documentation'] = $documentation;
    }
}
?>
