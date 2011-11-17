<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );

use LusiTime\LusiTime;

/**
 * Class NeoCaptar encapsulates operations with the 'neocaptar' database
 *
 * @author gapon
 */
class NeoCaptar {

    // ---------------------------------------------------
    // --- SIMPLIFIED INTERFACE AND ITS IMPLEMENTATION ---
    // ---------------------------------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return unknown_type
     */
    public static function instance() {
        if( is_null( NeoCaptar::$instance )) NeoCaptar::$instance = new NeoCaptar();
        return NeoCaptar::$instance;
    }

    // -----------------------------------------
    // --- CORE CLASS AND ITS IMPLEMENTATION ---
    // -----------------------------------------
    
    /* Data members
     */
    private $connection;

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct (
        $host     = null,
        $user     = null,
        $password = null,
        $database = null ) {

        $this->connection =
            new NeoCaptarConnection (
                is_null($host)     ? NEOCAPTAR_DEFAULT_HOST : $host,
                is_null($user)     ? NEOCAPTAR_DEFAULT_USER : $user,
                is_null($password) ? NEOCAPTAR_DEFAULT_PASSWORD : $password,
                is_null($database) ? NEOCAPTAR_DEFAULT_DATABASE : $database);
    }

    /*
     * ==========================
     *   TRANSACTION MANAGEMENT
     * ==========================
     */
    public function begin    () { $this->connection->begin    (); }
    public function commit   () { $this->connection->commit   (); }
    public function rollback () { $this->connection->rollback (); }

    /*
     * =================================
     *   INFORMATION REQUEST OPERATION
     * =================================
     */
    public function dict_cables() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_cable ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictCable (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_dict_cable_by_id( $id ) {
    	return $this->find_dict_cable_by_( "id={$id}");
    }
    public function find_dict_cable_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal cable type name. A valid non-empty string is expected." );
    	return $this->find_dict_cable_by_( "name='{$name_escaped}'");
    }
    private function find_dict_cable_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_cable WHERE {$condition} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictCable (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function find_dict_connector_by_id( $id ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_connector WHERE id={$id}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarDictConnector (
        	$this->connection,
            $this->find_dict_cable_by_id( $attr['cable_id'] ),
            $attr );
    }
    public function find_dict_pinlist_by_id( $id ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_pinlist WHERE id={$id}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
		$attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarDictPinlist (
        	$this->connection,
            $this->find_dict_connector_by_id( $attr['connector_id'] ),
            $attr );
    }


    public function dict_locations() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_location ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictLocation (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_dict_location_by_id( $id ) {
    	return $this->find_dict_location_by_( "id={$id}");
    }
    public function find_dict_location_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal location name. A valid non-empty string is expected." );
    	return $this->find_dict_location_by_( "name='{$name_escaped}'");
    }
    private function find_dict_location_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_location WHERE {$condition} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictLocation (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function find_dict_rack_by_id( $id ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_rack WHERE id={$id}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarDictRack (
        	$this->connection,
            $this->find_dict_location_by_id( $attr['location_id'] ),
            $attr );
    }
    
    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function add_dict_cable( $name, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal cable type name. A valid non-empty string is expected." );
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// cable name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired cable within
    	// that (new) transaction.
    	//
    	try {
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_cable VALUES(NULL,'{$name_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
	    return $this->find_dict_cable_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_dict_cable_by_id( $id ) {
    	$this->delete_dict_cable_by_( "id={$id}" );
    }
    public function delete_dict_cable_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal cable type name. A valid non-empty string is expected." );
    	$this->delete_dict_cable_by_( "name='{$name_escaped}'" );
    }
    private function delete_dict_cable_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_cable WHERE {$condition}" );
    }
    public function delete_dict_connector_by_id( $id ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_connector WHERE id={$id}" );
    }
    public function delete_dict_pinlist_by_id( $id ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_pinlist WHERE id={$id}" );
    }


    public function add_dict_location( $name, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal location name. A valid non-empty string is expected." );
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// location name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired location within
    	// that (new) transaction.
    	//
    	try {
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_location VALUES(NULL,'{$name_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
	    return $this->find_dict_location_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_dict_location_by_id( $id ) {
    	$this->delete_dict_location_by_( "id={$id}" );
    }
    public function delete_dict_location_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal location name. A valid non-empty string is expected." );
    	$this->delete_dict_location_by_( "name='{$name_escaped}'" );
    }
    private function delete_dict_location_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_location WHERE {$condition}" );
    }
    public function delete_dict_rack_by_id( $id ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_rack WHERE id={$id}" );
    }
    



}

/* ==============
 *   Unit tests
 * ==============
 *
 * WARNING: Never run these tests against the production database!
 *          They will put a lot of garbage into database tables.
 *
function dump_dict_pinlist( $pinlist ) {
	print <<<HERE
            <table cellpadding=4><tbody>
              <tr><td>id:</td><td>{$pinlist->id()}</td></tr>
              <tr><td>name:</td><td>{$pinlist->name()}</td></tr>
              <tr><td>created_time:</td><td>{$pinlist->created_time()->toStringShort()}</td></tr>
              <tr><td>created_uid:</td><td>{$pinlist->created_uid()}</td></tr>
            </tbody></table>
HERE;
}
function dump_dict_connector( $connector ) {
	print <<<HERE
      <table cellpadding=4><tbody>
		<tr><td>id:</td><td>{$connector->id()}</td></tr>
        <tr><td>name:</td><td>{$connector->name()}</td></tr>
        <tr><td>created_time:</td><td>{$connector->created_time()->toStringShort()}</td></tr>
        <tr><td>created_uid:</td><td>{$connector->created_uid()}</td></tr>
        <tr><td>pinlist:</td>
          <td>
HERE;
	foreach( $connector->pinlists() as $pinlist ) dump_dict_pinlist( $pinlist );
	print <<<HERE
          </td>
        </tr>
      </tbody></table>
HERE;
}

function dump_dict_cable( $cable ) {
	print <<<HERE
<table cellpadding=4><tbody>
  <tr><td>id:</td><td>{$cable->id()}</td></tr>
  <tr><td>name:</td><td>{$cable->name()}</td></tr>
  <tr><td>created_time:</td><td>{$cable->created_time()->toStringShort()}</td></tr>
  <tr><td>created_uid:</td><td>{$cable->created_uid()}</td></tr>
  <tr><td>connector:</td>
    <td>
HERE;
	foreach( $cable->connectors() as $connector ) dump_dict_connector( $connector );
	print <<<HERE
    </td>
  </tr>
</tbody></table>
HERE;
}

function dump_dict( $neocaptar ) {
	foreach( $neocaptar->dict_cables() as $cable ) dump_dict_cable( $cable );
}

print <<<HERE
<html>
<head>
<style>
td {
  border: 1px solid #c0c0c0;
  border-right: 0;
  border-bottom: 0;
}
</style>
</head>
<body>
HERE;

try {
    $neocaptar = new NeoCaptar();
    $neocaptar->begin();

    print <<<HERE

<h3>Find cable by its name</h3>

HERE;
	$cable_name = 'TEST:CABLE:2011-10-18 17:55:49';
	$cable = $neocaptar->find_dict_cable_by_name( $cable_name );
	if( is_null( $cable )) print "<br>cable '{$cable_name}' not found";
	else dump_dict_cable( $cable );

    print <<<HERE

<h3>Find cable by its unique identifier</h3>

HERE;
	$cable_id = 22;
	$cable = $neocaptar->find_dict_cable_by_id( $cable_id );
	if( is_null( $cable )) print "<br>cable id={$cable_id} not found";
	else dump_dict_cable( $cable );

	print <<<HERE

<h3>Find connector by its unique identifier</h3>

HERE;
	$connector_id = 22;
	$connector = $neocaptar->find_dict_connector_by_id( $connector_id );
	if( is_null( $connector )) print "<br>connector id={$connector_id} not found";
	else dump_dict_connector( $connector );

	print <<<HERE

<h3>Find pinlist by its unique identifier</h3>

HERE;
	$pinlist_id = 22;
	$pinlist = $neocaptar->find_dict_pinlist_by_id( $pinlist_id );
	if( is_null( $pinlist )) print "<br>pinlist id={$pinlist_id} not found";
	else dump_dict_pinlist( $pinlist );

	print <<<HERE

<h3>Test adding and removal operations (try finding elements originated by user 'perazzo')</h3>

HERE;
	$cable_name     = 'TEST:2DELETE_CABLE:'    .LusiTime::now()->toStringShort();
	$connector_name = 'TEST:2DELETE_CONNECTOR:'.LusiTime::now()->toStringShort();
	$pinlist_name   = 'TEST:2DELETE_PINLIST:'  .LusiTime::now()->toStringShort();

	$cable     = $neocaptar->add_dict_cable( $cable_name,     LusiTime::now(), 'perazzo' );
	$connector = $cable    ->add_connector ( $connector_name, LusiTime::now(), 'perazzo' );
	$pinlist   = $connector->add_pinlist   ( $pinlist_name,   LusiTime::now(), 'perazzo' );

	$connector->delete_pinlist_by_name   ( $pinlist  ->name());
	$cable    ->delete_connector_by_name ( $connector->name());
	$neocaptar->delete_dict_cable_by_name( $cable    ->name());

	$cable     = $neocaptar->add_dict_cable( 'TEST:2DELETE_CABLE:'    .LusiTime::now()->toStringShort(), LusiTime::now(), 'perazzo' );
	$connector = $cable    ->add_connector ( 'TEST:2DELETE_CONNECTOR:'.LusiTime::now()->toStringShort(), LusiTime::now(), 'perazzo' );
	$pinlist   = $connector->add_pinlist   ( 'TEST:2DELETE_PINLIST:'  .LusiTime::now()->toStringShort(), LusiTime::now(), 'perazzo' );

	$neocaptar->delete_dict_pinlist_by_id  ( $pinlist  ->id());
	$neocaptar->delete_dict_connector_by_id( $connector->id());
	$neocaptar->delete_dict_cable_by_id    ( $cable    ->id());

	print <<<HERE

<h3>Pass I: cable types dictionary before appying modifications</h3>

HERE;
    dump_dict( $neocaptar );

	$cable     = $neocaptar->add_dict_cable( 'TEST:CABLE:'    .LusiTime::now()->toStringShort(), LusiTime::now(), 'gapon' );
	$connector = $cable->add_connector     ( 'TEST:CONNECTOR:'.LusiTime::now()->toStringShort(), LusiTime::now(), 'gapon' );
	$pinlist   = $connector->add_pinlist   ( 'TEST:PINLIST:'  .LusiTime::now()->toStringShort(), LusiTime::now(), 'gapon' );
	$pinlist   = $connector->add_pinlist   ( 'TEST:PINLIST:'  .LusiTime::now()->toStringShort(), LusiTime::now(), 'gapon' );
	$pinlist   = $connector->add_pinlist   ( 'TEST:PINLIST:'  .LusiTime::now()->toStringShort(), LusiTime::now(), 'gapon' );
	
	print <<<HERE

<h3>Pass II: cable types dictionary after appying modifications</h3>

HERE;
    dump_dict( $neocaptar );

    $neocaptar->commit();

} catch( NeoCaptarException $e ) { print( $e->toHtml()); }
  catch( LusiTimeException  $e ) { print( $e->toHtml()); }

print <<<HERE
</body>
</html>
HERE;

 *
 */
?>
