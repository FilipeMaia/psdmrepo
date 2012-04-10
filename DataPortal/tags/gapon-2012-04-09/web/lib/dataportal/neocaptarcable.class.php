<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

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

    public function job       () { return $this->attr['job'       ]; }
    public function cable     () { return $this->attr['cable'     ]; }
    public function device    () { return $this->attr['device'    ]; }
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

    public function history() {
        $list = array();
        $result = $this->connection->query("SELECT event_time,event_uid,event FROM {$this->connection->database}.cable_history WHERE cable_id={$this->id()} ORDER BY event_time");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push(
                $list,
                new NeoCaptarCableEvent(
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
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
    public function update_self($params) {
        return $this->project()->update_cable($this, $params);
    }
}
?>
