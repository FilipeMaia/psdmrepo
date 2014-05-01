<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptarDictDeviceLocation is an abstraction for device locations stored
 * in the dictionary.
 *
 * @author gapon
 */
class NeoCaptarDictDeviceLocation {

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
    public function created_time () { return LusiTime::from64( $this->attr['created_time'] ); }
    public function created_uid  () { return $this->attr['created_uid']; }

    /*
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function regions() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_device_region WHERE location_id={$this->id()} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictDeviceRegion(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function find_region_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal region name. A valid non-empty string is expected." );
        return $this->find_region_by_( "name='{$name_escaped}'" );
    }
    public function find_region_by_id( $id ) {
    	return $his->neocaptar()->find_dict_device_region_by_id( $id );
    }
    private function find_region_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_device_region WHERE location_id={$this->id()} AND {$condition}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictDeviceRegion (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    
    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function add_region( $name, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal region name. A valid non-empty string is expected." );
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// region name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired region within
    	// that (new) transaction.
    	//
    	try {
    		$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_device_region VALUES(NULL,{$this->id()},'{$name_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
    	return $this->find_region_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_region_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal region name. A valid non-empty string is expected." );
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_device_region WHERE location_id={$this->id()} AND name='{$name_escaped}'" );
    }
    public function delete_region_by_id( $id ) { $this->neocaptar()->delete_dict_device_region_by_id( $id ); }
}
?>
