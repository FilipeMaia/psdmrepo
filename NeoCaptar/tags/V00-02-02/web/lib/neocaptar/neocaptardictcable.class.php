<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictCable is an abstraction for cable types stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictCable {

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
    public function connectors() {
        $list = array();
        $sql = "SELECT dict_connector.* FROM {$this->connection->database}.dict_connector, {$this->connection->database}.dict_cable_connector_link ".
                " WHERE dict_connector.id=dict_cable_connector_link.connector_id  AND dict_cable_connector_link.cable_id={$this->id()}".
                " ORDER BY dict_connector.name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictConnector (
                    $this->connection,
                    $this->neocaptar(),
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function is_linked($connector_id) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_cable_connector_link WHERE cable_id={$this->id()} AND connector_id={$connector_id}";
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
        $this->connection->query("UPDATE {$this->connection->database}.dict_cable SET documentation='{$documentation_escaped}' WHERE id={$this->id()}");
        $this->attr['documentation'] = $documentation;
    }
    public function link($connector_id) {
        $this->connection->query("INSERT INTO {$this->connection->database}.dict_cable_connector_link VALUES ({$this->id()},{$connector_id})" );
    }
    public function unlink($connector_id) {
        $sql = "DELETE FROM {$this->connection->database}.dict_cable_connector_link WHERE cable_id={$this->id()} AND connector_id={$connector_id}";
        $this->connection->query ( $sql );
   }
}
?>
