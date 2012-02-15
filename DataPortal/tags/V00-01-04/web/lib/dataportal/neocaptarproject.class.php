<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarProject is an abstraction for projects.
 *
 * @author gapon
 */
class NeoCaptarProject {

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
    public function owner        () { return $this->attr['owner']; }
    public function title        () { return $this->attr['title']; }
    public function description  () { return $this->attr['description']; }
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function due_time     () { return LusiTime::from64( $this->attr['due_time'] ); }
    public function modified_time() { return LusiTime::from64( $this->attr['modified_time'] ); }

    /*
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function status() {
        $counters = array(
            'total'        => 0,
            'Planned'      => 0,
            'Registered'   => 0,
            'Labeled'      => 0,
            'Fabrication'  => 0,
            'Ready'        => 0,
            'Installed'    => 0,
            'Commissioned' => 0,
            'Damaged'      => 0,
            'Retired'      => 0
        );
        $sql = "SELECT status, count(status) as 'count' FROM {$this->connection->database}.cable WHERE project_id={$this->id()} GROUP BY status";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            $status = $attr['status'];
            $count  = $attr['count'];
            $counters[$status] = $count;
            $counters['total'] += $count;
        }
        return $counters;
    }
    public function cables() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.cable WHERE project_id={$this->id()} ORDER BY id DESC";
        $result = $this->connection->query( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarCable (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_cable_by_id( $id ) {
    	return $this->neocaptar()->find_cable_by_id( $id );
    }
    private function find_cable_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.cable WHERE project_id={$this->id()} AND {$condition}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarCable (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function add_cable() {
        $sql = "INSERT INTO {$this->connection->database}.cable VALUES(NULL,{$this->id()},'Planned','','','','','','','','','','','','','','','','','','','','','','','','','','','','','')";
        $this->connection->query($sql);
        $new_cable = $this->find_cable_by_('id IN (SELECT LAST_INSERT_ID())');
        if( is_null($new_cable)) return null;
        $this->neocaptar->add_cable_event($new_cable,'Created');
        return $new_cable;
    }
    public function clone_cable($c) {
        if(is_null($c))
            throw new NeoCaptarException (
        		__METHOD__, "wrong parameter: null object passed into the method." );
        $sql =
            "INSERT INTO {$this->connection->database}.cable VALUES(NULL,{$this->id()},".
            "'Planned',".   // status must be reset
            "'',".                              // job number can't be cloned
            "'',".                              // cable number can't be cloned
            "'{$this->connection->escape_string(trim($c->device()))}',".
            "'{$this->connection->escape_string(trim($c->func()))}',".
            "'{$this->connection->escape_string(trim($c->length()))}',".
            "'{$this->connection->escape_string(trim($c->cable_type()))}',".
            "'{$this->connection->escape_string(trim($c->routing()))}',".

            "'{$this->connection->escape_string(trim($c->origin_name()))}',".
            "'{$this->connection->escape_string(trim($c->origin_loc()))}',".
            "'{$this->connection->escape_string(trim($c->origin_rack()))}',".
            "'{$this->connection->escape_string(trim($c->origin_ele()))}',".
            "'{$this->connection->escape_string(trim($c->origin_side()))}',".
            "'{$this->connection->escape_string(trim($c->origin_slot()))}',".
            "'{$this->connection->escape_string(trim($c->origin_conn()))}',".
            "'{$this->connection->escape_string(trim($c->origin_conntype()))}',".
            "'{$this->connection->escape_string(trim($c->origin_pinlist()))}',".
            "'{$this->connection->escape_string(trim($c->origin_station()))}',".
            "'{$this->connection->escape_string(trim($c->origin_instr()))}',".

            "'{$this->connection->escape_string(trim($c->destination_name()))}',".
            "'{$this->connection->escape_string(trim($c->destination_loc()))}',".
            "'{$this->connection->escape_string(trim($c->destination_rack()))}',".
            "'{$this->connection->escape_string(trim($c->destination_ele()))}',".
            "'{$this->connection->escape_string(trim($c->destination_side()))}',".
            "'{$this->connection->escape_string(trim($c->destination_slot()))}',".
            "'{$this->connection->escape_string(trim($c->destination_conn()))}',".
            "'{$this->connection->escape_string(trim($c->destination_conntype()))}',".
            "'{$this->connection->escape_string(trim($c->destination_pinlist()))}',".
            "'{$this->connection->escape_string(trim($c->destination_station()))}',".
            "'{$this->connection->escape_string(trim($c->destination_instr()))}'".
            ")";
        $this->connection->query($sql);
        $new_cable = $this->find_cable_by_('id IN (SELECT LAST_INSERT_ID())');
        if( is_null($new_cable)) return null;
        $this->neocaptar->add_cable_event($new_cable,'Created');
        return $new_cable;
    }
    public function update_cable($c,$params) {
        if(is_null($c))
            throw new NeoCaptarException (
        		__METHOD__, "wrong parameter: null cable object passed into the method." );
        if(is_null($params))
            throw new NeoCaptarException (
        		__METHOD__, "wrong parameter: null attributes object passed into the method." );


        // Update the cable
        //
        $first = true;
        $sql = "UPDATE {$this->connection->database}.cable ";
        foreach( $params as $p => $v ) {
            $sql .= $first ? "SET " : ",";
            $sql .= "{$p}='".$this->connection->escape_string(trim($v))."'";
            $first = false;
        }
        if( $first ) return $c;  // nothing to be updated
        $sql .= " WHERE id={$c->id()}";
        $this->connection->query($sql);

        // Update the host project's modificaton time too
        //
        $now_64 = LusiTime::now()->to64();
        $sql = "UPDATE {$this->connection->database}.project SET modified_time={$now_64} WHERE id={$this->id()}";
        $this->connection->query($sql);

        $new_cable = $this->find_cable_by_id($c->id());  // fetch the updated object
        if( is_null($new_cable)) return null;
        $this->neocaptar->add_cable_event($new_cable,'Modified');
        return $new_cable;
    }
    public function delete_cable_by_id( $id ) {
    	return $this->neocaptar()->delete_cable_by_id( $id );
    }
}
?>
