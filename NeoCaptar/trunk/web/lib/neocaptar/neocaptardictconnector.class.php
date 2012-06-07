<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictConnector is an abstraction for connector types stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictConnector {

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
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function cables() {
        $list = array();
        $sql = "SELECT dict_cable.* FROM {$this->connection->database}.dict_cable, {$this->connection->database}.dict_cable_connector_link ".
                " WHERE dict_cable.id=dict_cable_connector_link.cable_id AND dict_cable_connector_link.connector_id={$this->id()}".
                " ORDER BY dict_cable.name";
        $result = $this->connection->query ( $sql );
        for( $nrows = mysql_numrows( $result ), $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictCable (
                    $this->connection,
                    $this->neocaptar(),
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function is_linked($cable_id) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_cable_connector_link WHERE cable_id={$cable_id} AND connector_id={$this->id()}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        return $nrows != 0;
    }

    /*
     * ====================================
     *   DATABASE MODIFICATION OPERATIONS
     * ====================================
     */
    public function update($documentation) {
        $documentation_escaped = $this->connection->escape_string(trim($documentation));
        $this->connection->query("UPDATE {$this->connection->database}.dict_connector SET documentation='{$documentation_escaped}' WHERE id={$this->id()}");
        $this->attr['documentation'] = $documentation;
    }
    public function link($cable_id) {
        $this->connection->query("INSERT INTO {$this->connection->database}.dict_cable_connector_link VALUES ({$cable_id},{$this->id()})" );
    }
    public function unlink($cable_id) {
        $sql = "DELETE FROM {$this->connection->database}.dict_cable_connector_link WHERE cable_id={$cable_id} AND connector_id={$this->id()}";
        $this->connection->query ( $sql );
   }
}
?>
