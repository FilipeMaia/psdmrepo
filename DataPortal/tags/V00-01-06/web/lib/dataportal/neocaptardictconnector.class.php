<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictCable is an abstraction for connector types stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictConnector {

   /* Data members
     */
    private $connection;
    private $cable;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $cable, $attr ) {
        $this->connection = $connection;
        $this->cable = $cable;
        $this->attr = $attr;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function cable        () { return $this->cable; }
    public function id           () { return $this->attr['id']; }
    public function name         () { return $this->attr['name']; }
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function created_uid  () { return $this->attr['created_uid']; }

    /*
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function pinlists() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_pinlist WHERE connector_id={$this->id()} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictPinlist (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function find_pinlist_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal pinlist type name. A valid non-empty string is expected." );
        return $this->find_pinlist_by_( "name='{$name_escaped}'" );
    }
    public function find_pinlist_by_id( $id ) {
    	return $this->cable()->neocaptar()->find_dict_pinlist_by_id( $id );
    }
    private function find_pinlist_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_pinlist WHERE connector_id={$this->id()} AND {$condition}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictPinlist (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    
    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function add_pinlist( $name, $documentation, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal pinlist type name. A valid non-empty string is expected." );
    	$documentation_escaped = $this->connection->escape_string( trim( $documentation ));
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// pinlist name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired pinlist within
    	// that (new) transaction.
    	//
    	try {
    		$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_pinlist VALUES(NULL,{$this->id()},'{$name_escaped}','{$documentation_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
    	return $this->find_pinlist_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function update_pinlist( $pinlist_id, $documentation ) {
    	$documentation_escaped = $this->connection->escape_string( trim( $documentation ));
   		$this->connection->query (
   			"UPDATE {$this->connection->database}.dict_pinlist SET documentation='{$documentation_escaped}' WHERE id={$pinlist_id}"
   		);
        return $this->find_pinlist_by_id($pinlist_id);
    }
    public function delete_pinlist_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal pinlist type name. A valid non-empty string is expected." );
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_pinlist WHERE connector_id={$this->id()} AND name='{$name_escaped}'" );
    }
    public function delete_pinlist_by_id( $id ) { $this->cable()->neocaptar()->delete_dict_pinlist_by_id( $id ); }
}
?>
