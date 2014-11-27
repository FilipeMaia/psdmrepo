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
     *   DATABASE ACCESSOR OPERATIONS
     * ====================================
     */
    public function cable() {
        $cable_id = intval( $this->attr['cable_id'] );
        if($cable_id) {
            $cable = $this->neocaptar->find_dict_cable_by_id($cable_id);
            return $cable->name();
        }
        return '';
    }
    public function origin_connector() {
        return $this->find_connector_by_id( intval( $this->attr['origin_connector_id'] ));
    }
    public function destination_connector() {
        return $this->find_connector_by_id( intval( $this->attr['destination_connector_id'] ));
    }
    private function find_connector_by_id($id) {
        if($id) {
            $connector = $this->neocaptar->find_dict_connector_by_id($id);
            return $connector->name();
        }
        return '';
    }

    /*
     * ====================================
     *   DATABASE MODIFICATION OPERATIONS
     * ====================================
     */
    public function update_documentation($documentation) {
        $documentation_escaped = $this->connection->escape_string(trim($documentation));
        $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET documentation='{$documentation_escaped}' WHERE id={$this->id()}");
        $this->attr['documentation'] = $documentation;
    }
    public function update_cable($cable_id=null) {
        if( is_null($cable_id)) {
            $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET cable_id=NULL WHERE id={$this->id()}");
            $this->attr['cable_id'] = '';
        } else {
            $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET cable_id={$cable_id} WHERE id={$this->id()}");
            $this->attr['cable_id'] = $cable_id;
        }
    }
    public function update_origin_connector($connector_id=null) {
        if( is_null($connector_id)) {
            $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET origin_connector_id=NULL WHERE id={$this->id()}");
            $this->attr['origin_connector_id'] = '';
        } else {
            $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET origin_connector_id={$connector_id} WHERE id={$this->id()}");
            $this->attr['origin_connector_id'] = $connector_id;
        }
    }
    public function update_destination_connector($connector_id=null) {
        if( is_null($connector_id)) {
            $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET destination_connector_id=NULL WHERE id={$this->id()}");
            $this->attr['origin_connector_id'] = '';
        } else {
            $this->connection->query("UPDATE {$this->connection->database}.dict_pinlist SET destination_connector_id={$connector_id} WHERE id={$this->id()}");
            $this->attr['destination_connector_id'] = $connector_id;
        }
    }
}
?>
