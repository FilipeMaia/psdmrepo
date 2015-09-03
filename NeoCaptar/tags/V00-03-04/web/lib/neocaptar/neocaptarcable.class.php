<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarCable is an abstraction for cables.
 *
 * @author gapon
 */
class NeoCaptarCable {

   /* Data members
     */
    private $connection;
    private $project;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $project, $attr ) {
        $this->connection = $connection;
        $this->project = $project;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function project() { return $this->project; }

    public function id     () { return $this->attr['id'    ]; }
    public function status () { return $this->attr['status']; }

    public function cable      () { return        $this->attr['cable'      ];  }
    public function revision   () { return intval($this->attr['revision'   ]); }
    public function description() { return        $this->attr['description'];  }

    public function device          () { return $this->attr['device'          ]; }
    public function device_location () { return $this->attr['device_location' ]; }
    public function device_region   () { return $this->attr['device_region'   ]; }
    public function device_component() { return $this->attr['device_component']; }
    public function device_counter  () { return $this->attr['device_counter'  ]; }
    public function device_suffix   () { return $this->attr['device_suffix'   ]; }

    public function func      () { return $this->attr['func'      ]; }
    public function length    () { return $this->attr['length'    ]; }
    public function cable_type() { return $this->attr['cable_type']; }
    public function routing   () { return $this->attr['routing'   ]; }

    public function origin_name    () { return $this->attr['origin_name'    ]; }
	public function origin_loc     () { return $this->attr['origin_loc'     ]; }
	public function origin_rack    () { return $this->attr['origin_rack'    ]; }
	public function origin_ele     () { return $this->attr['origin_ele'     ]; }
	public function origin_side    () { return $this->attr['origin_side'    ]; }
	public function origin_slot    () { return $this->attr['origin_slot'    ]; }
	public function origin_station () { return $this->attr['origin_station' ]; }
	public function origin_conn    () { return $this->attr['origin_conn'    ]; }
	public function origin_conntype() { return $this->attr['origin_conntype']; }
	public function origin_pinlist () { return $this->attr['origin_pinlist' ]; }
	public function origin_instr   () { return $this->attr['origin_instr'   ]; }

    public function destination_name    () { return $this->attr['destination_name'    ]; }
	public function destination_loc     () { return $this->attr['destination_loc'     ]; }
	public function destination_rack    () { return $this->attr['destination_rack'    ]; }
	public function destination_ele     () { return $this->attr['destination_ele'     ]; }
	public function destination_side    () { return $this->attr['destination_side'    ]; }
	public function destination_slot    () { return $this->attr['destination_slot'    ]; }
	public function destination_station () { return $this->attr['destination_station' ]; }
	public function destination_conn    () { return $this->attr['destination_conn'    ]; }
	public function destination_conntype() { return $this->attr['destination_conntype']; }
	public function destination_pinlist () { return $this->attr['destination_pinlist' ]; }
	public function destination_instr   () { return $this->attr['destination_instr'   ]; }

    public function dump2array() {
        return array(

            "id: {$this->id()}",
            "status: {$this->status()}",
            "cable #: {$this->cable()}",
            "device: {$this->device()}",
            "func: {$this->func()}",
            "length: {$this->length()}",
            "cable_type: {$this->cable_type()}",
            "routing: {$this->routing()}",

            "origin_name: {$this->origin_name()}",
            "origin_loc: {$this->origin_loc()}",
            "origin_rack: {$this->origin_rack()}",
            "origin_ele: {$this->origin_ele()}",
            "origin_side: {$this->origin_side()}",
            "origin_slot: {$this->origin_slot()}",
            "origin_station: {$this->origin_station()}",
            "origin_conntype: {$this->origin_conntype()}",
            "origin_pinlist: {$this->origin_pinlist()}",
            "origin_instr: {$this->origin_instr()}",

            "destination_name: {$this->destination_name()}",
            "destination_loc: {$this->destination_loc()}",
            "destination_rack: {$this->destination_rack()}",
            "destination_ele: {$this->destination_ele()}",
            "destination_side: {$this->destination_side()}",
            "destination_slot: {$this->destination_slot()}",
            "destination_station: {$this->destination_station()}",
            "destination_conn: {$this->destination_conn()}",
            "destination_conntype: {$this->destination_conntype()}",
            "destination_pinlist: {$this->destination_pinlist()}",
            "destination_instr: {$this->destination_instr()}"
        );
    }

    /*
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function history() {
        $list = array();
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.cable_history WHERE cable_id={$this->id()} ORDER BY event_time");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push(
                $list,
                new NeoCaptarCableEvent(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function history_last_entry() {
        $list = array();
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.cable_history WHERE cable_id={$this->id()} ORDER BY event_time DESC LIMIT 1");
        $nrows = mysql_numrows($result);
        if( !$nrows ) return null;
        if( 1 != $nrows )
            throw new NeoCaptarException(__METHOD__, "database schema problem");
        return new NeoCaptarCableEvent(
            $this->connection,
            $this,
            mysql_fetch_array($result, MYSQL_ASSOC));
    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function clone_self() {
        return $this->project()->clone_cable($this);
    }

    /**
     * Update cable attributes from the specified attribute/value pairs
     * passed in the input array.
     *
     * @param array $params
     * @return NeoCaptarCable
     */
    public function update_self($params, $comments="") {
        return $this->project()->update_cable($this, $params, $comments);
    }
}
?>
