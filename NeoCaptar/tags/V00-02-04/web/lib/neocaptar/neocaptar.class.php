<?php

namespace NeoCaptar;

require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
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

    // -----------------------------------
    // --- E-mail notification service ---
    // -----------------------------------

    public static function notify( $address, $subject, $body ) {
        $tmpfname = tempnam("/tmp", "neocaptar");
        $handle = fopen( $tmpfname, "w" );
        fwrite( $handle, $body );
        fclose( $handle );

        shell_exec( "cat {$tmpfname} | mail -s '{$subject}' {$address} -- -F 'PCDS Cable Manager'" );

        // Delete the file only after piping its contents to the mailer command.
        // Otherwise its contents will be lost before we use it.
        //
        unlink( $tmpfname );
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
    public function begin    () {
        $this->connection->begin();
        $this->update_current_user_activity();
    }
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

    public function dict_connectors() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_connector ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictConnector (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_dict_connector_by_id( $id ) {
    	return $this->find_dict_connector_by_( "id={$id}");
    }
    public function find_dict_connector_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal connector type name. A valid non-empty string is expected." );
    	return $this->find_dict_connector_by_( "name='{$name_escaped}'");
    }
    private function find_dict_connector_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_connector WHERE {$condition} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictConnector (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    public function dict_pinlists() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_pinlist ORDER BY name";
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
    public function find_dict_pinlist_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal pinlist type name. A valid non-empty string is expected." );
        return $this->find_dict_pinlist_by_( "name='{$name_escaped}'" );
    }
    public function find_dict_pinlist_by_id( $id ) {
    	return $this->find_dict_pinlist_by_( "id={$id}" );
    }
    private function find_dict_pinlist_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_pinlist WHERE {$condition}";
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


    public function dict_routings() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_routing ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictRouting (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_dict_routing_by_id( $id ) {
    	return $this->find_dict_routing_by_( "id={$id}");
    }
    public function find_dict_routing_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal routing name. A valid non-empty string is expected." );
    	return $this->find_dict_routing_by_( "name='{$name_escaped}'");
    }
    private function find_dict_routing_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_routing WHERE {$condition} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictRouting (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }



    public function dict_device_locations() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_device_location ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictDeviceLocation (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_dict_device_location_by_id( $id ) {
    	return $this->find_dict_device_location_by_( "id={$id}");
    }
    public function find_dict_device_location_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal device location type name. A valid non-empty string is expected." );
    	return $this->find_dict_device_location_by_( "name='{$name_escaped}'");
    }
    private function find_dict_device_location_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_device_location WHERE {$condition} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictDeviceLocation (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function find_dict_device_region_by_id( $id ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_device_region WHERE id={$id}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarDictDeviceRegion (
        	$this->connection,
            $this->find_dict_device_location_by_id( $attr['location_id'] ),
            $attr );
    }
    public function find_dict_device_component_by_id( $id ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_device_component WHERE id={$id}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
		$attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarDictDeviceComponent (
        	$this->connection,
            $this->find_dict_device_region_by_id( $attr['region_id'] ),
            $attr );
    }




    public function dict_instrs() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.dict_instr ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarDictinstr (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_dict_instr_by_id( $id ) {
    	return $this->find_dict_instr_by_( "id={$id}");
    }
    public function find_dict_instr_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal instr name. A valid non-empty string is expected." );
    	return $this->find_dict_instr_by_( "name='{$name_escaped}'");
    }
    private function find_dict_instr_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.dict_instr WHERE {$condition} ORDER BY name";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarDictInstr (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    public function projects($title=null,$owner=null,$begin=null,$end=null) {
        $condition = '';
        if(!is_null($title)) {
            $title_escaped = $this->connection->escape_string(trim($title));
            if($title_escaped != '')
                $condition .= ($condition == '' ? '' : ' AND ')." title LIKE '%{$title_escaped}%'";
        }
        if(!is_null($owner)) {
            $owner_escaped = $this->connection->escape_string(trim($owner));
            if( $owner_escaped != '' )
                $condition .= ($condition == '' ? '' : ' AND ')." owner='{$owner_escaped}'";
        }
        if(!is_null($begin)) {
            $condition .= ($condition == '' ? '' : ' AND ')." created_time >= {$begin->to64()}";
        }
        if(!is_null($end)) {
            $condition .= ($condition == '' ? '' : ' AND ')." created_time < {$end->to64()}";
        }
        return $this->find_projects_by_($condition);
    }
    public function projects_by_coowner($uid) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
        $condition = $uid_escaped == ''
            ? ''
            : "id IN (SELECT project_id FROM {$this->connection->database}.project_shared_access WHERE uid='{$uid_escaped}')";
        return $this->find_projects_by_($condition);
    }
    public function find_projects_by_jobnumber_prefix( $prefix ) {
    	$prefix_escaped = $this->connection->escape_string(trim($prefix));
        return $this->find_projects_by_(
            "job LIKE '%{$prefix_escaped}%'");
    }
    public function find_projects_by_( $condition='' ) {
        $where_condition = $condition == '' ? '' : "WHERE {$condition}";
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.project {$where_condition} ORDER BY title";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarProject (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }
    public function find_project_by_id( $id ) {
    	return $this->find_project_by_( "id={$id}");
    }
    public function find_project_by_owner( $uid ) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
    	return $this->find_project_by_( "owner='{$uid_escaped}'");
    }
    public function find_project_by_title( $title ) {
        $title_escaped = $this->connection->escape_string( trim( $title ));
    	return $this->find_project_by_( "title='{$title_escaped}'");
    }
    public function find_project_by_jobnumber( $jobnumber ) {
    	$jobnumber_escaped = $this->connection->escape_string(trim($jobnumber));
        return $this->find_project_by_(
            "job='{$jobnumber_escaped}'");
    }
    private function find_project_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.project WHERE {$condition} ORDER BY created_time";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt. Condition: '{$condition}'" );
        return new NeoCaptarProject (
        	$this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    public function find_cable_by_id( $id ) {
        $sql = "SELECT * FROM {$this->connection->database}.cable WHERE id={$id}";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarCable (
        	$this->connection,
            $this->find_project_by_id($attr['project_id']),
            $attr);
    }
    /**
     * Case insensitive combined search based on partial values of search parameters.
     * The parameters are provided in an input dictionary with the parameter name
     * as a key and its value.
     *
     * @param type $partial_search_params
     * @return type 
     */
    public function search_cables( $partial_search_params ) {
        // NOTE: case-insensitive search in SQL queries
        //
        $condition = null;
        $operator = array_key_exists('partial_or',$partial_search_params) ? ' OR ' : ' AND ';
        foreach( $partial_search_params as $p => $v ) {
            $p = trim($p);
            if( $p == 'partial_or' ) continue;
            if( $p == 'job') continue;
            $v = $this->connection->escape_string(trim($v));
            if(($p == '') || ($v == '')) continue;
            if( !is_null($condition)) $condition .= $operator;
            $condition .= " {$p} LIKE '%{$v}%' COLLATE latin1_general_ci";
        }
        if( is_null($condition))
            $condition = 'id is NOT NULL';
        if( array_key_exists('job',$partial_search_params)) {
            $v = $this->connection->escape_string(trim($partial_search_params['job']));
            if( $v != '' )
                $condition = "(({$condition}) AND project_id IN ( SELECT id FROM {$this->connection->database}.project WHERE job LIKE '%{$v}%' COLLATE latin1_general_ci ))";
        }
        return $this->find_cables_by_($condition);
    }
    public function find_cables_by_($condition) {
        $list = array();
        $id2project = array();
        $sql = "SELECT * FROM {$this->connection->database}.cable WHERE {$condition} ORDER BY project_id, status, cable";
        $result = $this->connection->query ( $sql );
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr       = mysql_fetch_array( $result, MYSQL_ASSOC );
            $project_id = $attr['project_id'];
            $project    = null;
            if( array_key_exists($project_id, $id2project)) {
                $project = $id2project[$project_id];
            } else {
                $project = $this->find_project_by_id($attr['project_id']);
                $id2project[$project_id] = $project;
            }
            array_push (
                $list,
                new NeoCaptarCable (
                    $this->connection,
                    $project,
                    $attr));
        }
        return $list;
    }
    public function find_orphant_cables() {
        $result = array();
        $prefix2range = array();
        foreach( NeoCaptarUtils::cablenumber_prefixes2array($this) as $p ) {
            $prefix_name = $p['name'];
            $prefix = $this->find_cablenumber_prefix_by_name($prefix_name);
            if( is_null($prefix))
                throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
            $prefix2range[$prefix_name] = array(
                'prefix' => $prefix,
                'range'  => $p['range']
            );
        }
        $cables = $this->find_cables_by_("cable != ''");
        foreach( $cables as $c ) {

            $cablenumber = $c->cable();
            $prefix_name = strtoupper(substr($cablenumber,0,2));
            $number      = intval(substr($cablenumber,2));
            if( !array_key_exists($prefix_name, $result))
                $result[$prefix_name] = array(
                    'out_of_range' => array(),
                    'in_range'     => array());

            $range_id = null;
            if( array_key_exists($prefix_name, $prefix2range)) {

                if( !is_null($prefix2range[$prefix_name]['prefix']->find_cablenumber_for($c->id()))) continue;

                foreach( $prefix2range[$prefix_name]['range'] as $range )
                    if(($range['first'] <= $number) && ($number <= $range['last'])) {
                        $range_id = $range['id'];
                        break;
                    }
            }
            array_push(
                $result[$prefix_name][is_null($range_id) ? 'out_of_range' : 'in_range'],
                 array(
                    'cable_id'    => $c->id(),
                    'cablenumber' => $cablenumber,
                    'prefix'      => $prefix_name,
                    'range_id'    => is_null($range_id) ? 0 : $range_id,
                    'number'      => $number ));
        }
        return $result;
    }
    public function find_reserved_cables() {
        $result = array();

        $prefixes = array();
        foreach( $this->cablenumber_prefixes() as $prefix )
            $prefixes[$prefix->name()] = $prefix;

        $cables = $this->find_cables_by_("cable = ''");
        foreach( $cables as $cable ) {
            foreach( $prefixes as $prefix_name => $prefix ) {
                $cablenumber = $prefix->find_cablenumber_for($cable->id());
                if( is_null($cablenumber)) continue;
                if( !array_key_exists($prefix_name, $result)) $result[$prefix_name] = array();
                array_push($result[$prefix_name], $cablenumber);
            }
        }
        return $result;
    }
    public function find_cable_by_cablenumber($cablenumber) {
    	$cablenumber_escaped = $this->connection->escape_string(trim($cablenumber));
    	if( $cablenumber_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal cable number. A valid non-empty string is expected." );
        $sql = "SELECT * FROM {$this->connection->database}.cable WHERE cable='{$cablenumber_escaped}'";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt. Query: {$sql}" );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarCable (
        	$this->connection,
            $this->find_project_by_id($attr['project_id']),
            $attr);
    }
    public function find_cables_by_prefix($prefix) {
    	$prefix_escaped = $this->connection->escape_string(trim($prefix));
        return $this->find_cables_by_(
            "cable LIKE '{$prefix_escaped}%'");

    }
    public function find_cables_by_cablenumber_range_id($range_id) {
        $prefix = $this->find_cablenumber_prefix_for_range_id($range_id);
        if( is_null($prefix))
            throw new NeoCaptarException (
        		__METHOD__, "illegal cable number range identifier {$range_id}");
    	$prefix_escaped = $this->connection->escape_string(trim($prefix->name()));
        return $this->find_cables_by_(
            "cable LIKE '{$prefix_escaped}%'");
    }
    public function find_cables_by_jobnumber( $jobnumber ) {
        $project = $this->find_project_by_jobnumber($jobnumber);
        if( is_null($project)) return array();
        return $project->cables();
    }
    public function find_cables_by_dict_cable_id($dict_cable_id) {
        $dict_cable = $this->find_dict_cable_by_id($dict_cable_id);
        if( is_null($dict_cable)) return array();
        $cable_type_escaped = $this->connection->escape_string(trim($dict_cable->name()));
        return $this->find_cables_by_(
            "cable_type='{$cable_type_escaped}'");
    }
    public function find_cables_by_dict_connector_id($dict_connector_id) {
        $dict_connector = $this->find_dict_connector_by_id($dict_connector_id);
        if( is_null($dict_connector)) return array();
        $connector_type_escaped = $this->connection->escape_string(trim($dict_connector->name()));
        return $this->find_cables_by_(
            "(origin_conntype='{$connector_type_escaped}' OR destination_conntype='{$connector_type_escaped}')");

    }
    public function find_cables_by_dict_pinlist_id($dict_pinlist_id) {
        $dict_pinlist = $this->find_dict_pinlist_by_id($dict_pinlist_id);
        if( is_null($dict_pinlist)) return array();
        $pinlist_escaped = $this->connection->escape_string(trim($dict_pinlist->name()));
        return $this->find_cables_by_(
            "origin_pinlist='{$pinlist_escaped}' OR destination_pinlist='{$pinlist_escaped}'");
    }
    public function find_cables_by_dict_location_id($dict_location_id) {
        $dict_location = $this->find_dict_location_by_id($dict_location_id);
        if( is_null($dict_location)) return array();
        $location_escaped = $this->connection->escape_string(trim($dict_location->name()));
        return $this->find_cables_by_(
            "origin_loc='{$location_escaped}' OR destination_loc='{$location_escaped}'");
    }
    public function find_cables_by_dict_rack_id($dict_rack_id) {
        $dict_rack = $this->find_dict_rack_by_id($dict_rack_id);
        if( is_null($dict_rack)) return array();
        $location_escaped = $this->connection->escape_string(trim($dict_rack->location()->name()));
        $rack_escaped     = $this->connection->escape_string(trim($dict_rack->name()));
        return $this->find_cables_by_(
            "(origin_loc='{$location_escaped}' AND origin_rack='{$rack_escaped}') OR ".
            "(destination_loc='{$location_escaped}' AND destination_rack='{$rack_escaped}')");
    }
    public function find_cables_by_dict_routing_id($dict_routing_id) {
        $dict_routing = $this->find_dict_routing_by_id($dict_routing_id);
        if( is_null($dict_routing)) return array();
        $routing_escaped = $this->connection->escape_string(trim($dict_routing->name()));
        return $this->find_cables_by_(
            "routing='{$routing_escaped}'");
    }
    public function find_cables_by_dict_instr_id($dict_instr_id) {
        $dict_instr = $this->find_dict_instr_by_id($dict_instr_id);
        if( is_null($dict_instr)) return array();
        $instr_escaped = $this->connection->escape_string(trim($dict_instr->name()));
        return $this->find_cables_by_(
            "origin_instr='{$instr_escaped}' OR destination_instr='{$instr_escaped}'");
    }

    public function find_cables_by_dict_device_location_id($dict_device_location_id) {
        $dict_device_location = $this->find_dict_device_location_by_id($dict_device_location_id);
        if( is_null($dict_device_location)) return array();
        $dict_device_location_escaped = $this->connection->escape_string(trim($dict_device_location->name()));
        return $this->find_cables_by_(
            "device_location='{$dict_device_location_escaped}'");
    }
    public function find_cables_by_dict_device_region_id($dict_device_region_id) {
        $dict_device_region = $this->find_dict_device_region_by_id($dict_device_region_id);
        if( is_null($dict_device_region)) return array();
        $dict_device_location_escaped = $this->connection->escape_string(trim($dict_device_region->location()->name()));
        $dict_device_region_escaped   = $this->connection->escape_string(trim($dict_device_region->name()));
        return $this->find_cables_by_(
            "device_location='{$dict_device_location_escaped}' AND device_region='{$dict_device_region_escaped}'");
    }
    public function find_cables_by_dict_device_component_id($dict_device_component_id) {
        $dict_device_component = $this->find_dict_device_component_by_id($dict_device_component_id);
        if( is_null($dict_device_component)) return array();
        $dict_device_location_escaped  = $this->connection->escape_string(trim($dict_device_component->region()->location()->name()));
        $dict_device_region_escaped    = $this->connection->escape_string(trim($dict_device_component->region()->name()));
        $dict_device_component_escaped = $this->connection->escape_string(trim($dict_device_component->name()));
        return $this->find_cables_by_(
            "device_location='{$dict_device_location_escaped}' AND device_region='{$dict_device_region_escaped}' AND device_component='{$dict_device_component_escaped}'");
    }

    public function cablenumber_prefixes() {
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.cablenumber_prefix ORDER BY prefix";
        $result = $this->connection->query ( $sql );
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push (
                $list,
                new NeoCaptarCableNumberPrefix (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function find_cablenumber_prefix_by_name($name) {
    	$name_escaped = $this->connection->escape_string(trim($name));
    	if( $name_escaped == '' ) throw new NeoCaptarException(__METHOD__,"illegal prefix name");
        return $this->find_cablenumber_prefix_by_("prefix='{$name_escaped}'");
    }
    private function find_cablenumber_prefix_by_($condition) {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.cablenumber_prefix WHERE {$condition}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarCableNumberPrefix (
            $this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    private function find_cablenumber_prefix_for_range_id($range_id) {
        $range_id = intval($range_id);
        foreach( $this->cablenumber_prefixes() as $prefix) {
            foreach( $prefix->ranges() as $range ) {
                if( $range['id'] == $range_id) return $prefix;
            }
        }
        return null;
    }
    
    public function jobnumber_allocations() {
        $this->update_jobnumber_allocations();
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.jobnumber ORDER BY owner";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push (
                $list,
                new NeoCaptarJobNumberAlloc (
                    $this->connection,
                    $this,
                    $attr,
                    $this->jobnumber_total_allocated($attr['owner']),
                    $this->jobnumber_last_allocation($attr['owner'])));
        }
        return $list;
    }
    /**
     * Look for job numbers within  allowed range for the specified project
     * owner only.
     *
     * @param string $owner
     * @return int 
     */
    public function jobnumber_total_allocated($owner) {
        $owner_escaped = $this->connection->escape_string(trim($owner));
        if( $owner_escaped == '' )
            throw new NeoCaptarException (
                __METHOD__, "empty string passed where a valid project owner account was expected." );
        $table_j  = "{$this->connection->database}.jobnumber `j`";
        $table_ja = "{$this->connection->database}.jobnumber_allocation `ja`";
        $sql      = "SELECT COUNT(*) AS `total` FROM {$table_j}, {$table_ja} WHERE j.owner='{$owner_escaped}' AND ja.jobnumber_id=j.id AND j.first <= ja.jobnumber AND ja.jobnumber <= j.last";
        $result   = $this->connection->query($sql);
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return 0;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return $attr['total'];
    }

    /**
     * Look for job numbers within allowed range for the specified project owner only.
     *
     * @param string $owner
     * @return array (keys: 'project_id','jobnumber','allocated_time','allocated_by_uid') 
     */
    private function jobnumber_last_allocation($owner) {
        $owner_escaped = $this->connection->escape_string(trim($owner));
        if( $owner_escaped == '' )
            throw new NeoCaptarException (
                __METHOD__, "empty string passed where a valid project owner account was expected." );
        $table_j  = "{$this->connection->database}.jobnumber `j`";
        $table_ja = "{$this->connection->database}.jobnumber_allocation `ja`";
        $sql      = "SELECT ja.project_id,ja.jobnumber,ja.allocated_time,ja.allocated_by_uid FROM {$table_j}, {$table_ja} WHERE j.owner='{$owner_escaped}' AND ja.jobnumber_id=j.id AND j.first <= ja.jobnumber AND ja.jobnumber <= j.last ORDER BY jobnumber DESC LIMIT 1";
        $result   = $this->connection->query($sql);
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return mysql_fetch_array( $result, MYSQL_ASSOC );
    }

    public function find_jobnumber_allocation_by_id($id) {
        if( intval($id) == 0 ) throw new NeoCaptarException(__METHOD__,"illegal identifier");
        return $this->find_jobnumber_allocation_by_("id={$id}");

    }
    public function find_jobnumber_allocation_by_owner($owner) {
    	$owner_escaped = $this->connection->escape_string( trim($owner));
    	if( $owner_escaped == '' ) throw new NeoCaptarException(__METHOD__,"illegal project owner name");
        return $this->find_jobnumber_allocation_by_("owner='{$owner_escaped}'");
    }
    private function find_jobnumber_allocation_by_($condition) {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.jobnumber WHERE {$condition}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarJobNumberAlloc (
            $this->connection,
            $this,
            $attr,
            $this->jobnumber_total_allocated($attr['owner']),
            $this->jobnumber_last_allocation($attr['owner']));
    }

    public function cable_origin_locations() { return $this->table_property_('cable',      'origin_loc'); }
    public function cablenumber_locations () { return $this->table_property_('cablenumber','location');   }
    public function project_owners        () { return $this->table_property_('project',    'owner');      }
    public function jobnumber_owners      () { return $this->table_property_('jobnumber',  'owner');      }

    public function project_managers_and_administrators() {
        $result = array();
        foreach( $this->users() as $user ) {
            if( $user->is_projmanager() || $user->is_administrator()) {
                array_push($result, $user->uid());
            }
        }
        return $result;
    }

    private function table_property_($table,$property) {
        $list = array();
        $result = $this->connection->query("SELECT DISTINCT {$property} FROM {$this->connection->database}.{$table} WHERE {$property} != '' ORDER BY {$property}");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push($list,$attr[$property]);
        }
        return $list;
    }

    public function users() {
        $list = array();
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.user ORDER BY uid,role");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push(
                $list,
                new NeoCaptarUser(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    
    public function find_user_by_uid($uid) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.user WHERE uid='{$uid_escaped}'");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarUser(
            $this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function current_user() {
        return $this->find_user_by_uid(AuthDB::instance()->authName());
    }
    public function is_other() {
        $user = $this->current_user();
        return !is_null($user) && $user->is_other();
    }
    public function is_administrator() {
        $user = $this->current_user();
        return !is_null($user) && $user->is_administrator();
    }
    public function can_manage_projects() {
        $user = $this->current_user();
        return !is_null($user) && ($user->is_administrator() || $user->is_projmanager());
    }
    public function has_dict_priv() {
        $user = $this->current_user();
        return !is_null($user) && $user->has_dict_priv();
    }

    public function notify_event_types($recipient=null) {
        $list = array();
        $recipient_condition = is_null($recipient) ? '' : "WHERE recipient='{$this->connection->escape_string(trim($recipient))}'";
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_event_type {$recipient_condition} ORDER BY recipient,scope,id");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push(
                $list,
                new NeoCaptarNotifyEventType(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }

    public function find_notify_event_type_by_id($id) {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_event_type WHERE id={$id}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarNotifyEventType(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function find_notify_event_type($recipient,$name) {
        $recipient_escaped = $this->connection->escape_string(trim($recipient));
        $name_escaped      = $this->connection->escape_string(trim($name));
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_event_type WHERE recipient='{$recipient_escaped}' and name='{$name_escaped}'");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarNotifyEventType(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function notify_schedule() {
        $dictionary = array();
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_schedule");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            $dictionary[$attr['recipient']] = $attr['mode'];
        }
        return $dictionary;
    }
    public function notifications($uid=null) {
        $list = array();
        $uid_condition = is_null($uid) ? '' : "WHERE uid='{$this->connection->escape_string(trim($uid))}'";
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify {$uid_condition}");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push(
                $list,
                new NeoCaptarNotify(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function find_notification_by_id($id) {
        return $this->find_notification_by_("id={$id}");
    }
    public function find_notification($uid, $event_type_id) {
        $uid_escaped  = $this->connection->escape_string(trim($uid));
        return $this->find_notification_by_("uid='{$uid_escaped}' AND event_type_id={$event_type_id}");
    }
    public function find_notification_by_($condition) {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify WHERE {$condition}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarNotify(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    public function notify_queue() {
        $list = array();
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_queue ORDER BY event_time");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ )
            array_push(
                $list,
                new NeoCaptarNotifyQueuedEntry(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function find_notify_queue_entry_by_id($id) {
        return $this->find_notify_queue_entry_by_("id={$id}");
    }
    public function find_notify_queue_entry_by_($condition) {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.notify_queue WHERE {$condition}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return new NeoCaptarNotifyQueuedEntry(
            $this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    public function add_cablenumber_prefix($name) {
        $name_escaped = $this->connection->escape_string(trim($name));
        $this->connection->query("INSERT {$this->connection->database}.cablenumber_prefix VALUES (NULL,'{$name_escaped}')");
        return $this->find_cablenumber_prefix_by_("id in (SELECT LAST_INSERT_ID())");
    }

    /**
     * Scan projects to get all known owner names and create a job numbers
     * allocation for each owner not registered in the database.
     * table.
     */
    public function update_jobnumber_allocations() {
        foreach( array_diff( $this->project_managers_and_administrators(), $this->jobnumber_owners()) as $owner )
            $this->add_jobnumber_allocation($owner, '', 0, 0);
    }

    public function known_project_owners() {
        $list = array();
        $result = $this->connection->query("SELECT DISTINCT owner FROM {$this->connection->database}.project ORDER BY owner");
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push($list,$attr['owner']);
        }
        return $list;
    }
    /*
     * ======================
     *   DATABASE MODIFIERS
     * ======================
     */
    public function add_dict_cable( $name, $documentation, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal cable type name. A valid non-empty string is expected." );
    	$documentation_escaped = $this->connection->escape_string( trim( $documentation ));
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
    			"INSERT INTO {$this->connection->database}.dict_cable VALUES(NULL,'{$name_escaped}','{$documentation_escaped}',{$created->to64()},'{$uid_escaped}')"
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

    public function add_dict_connector( $name, $documentation, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal connetor type name. A valid non-empty string is expected." );
    	$documentation_escaped = $this->connection->escape_string( trim( $documentation ));
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// connector name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired connetor within
    	// that (new) transaction.
    	//
    	try {
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_connector VALUES(NULL,'{$name_escaped}','{$documentation_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
	    return $this->find_dict_connector_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_dict_connector_by_id( $id ) {
    	$this->delete_dict_connector_by_( "id={$id}" );
    }
    public function delete_dict_connector_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal connector type name. A valid non-empty string is expected." );
    	$this->delete_dict_connector_by_( "name='{$name_escaped}'" );
    }
    private function delete_dict_connector_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_connector WHERE {$condition}" );
    }

    public function add_dict_pinlist( $name, $documentation, $created, $uid, $cable_id=null, $origin_connector_id=null, $destination_connector_id=null ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal pinlist type name. A valid non-empty string is expected." );
    	$documentation_escaped = $this->connection->escape_string( trim( $documentation ));
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

        $cable_id_opt                 = is_null($cable_id)                 ? 'NULL' : intval($cable_id);
        $origin_connector_id_opt      = is_null($origin_connector_id)      ? 'NULL' : intval($origin_connector_id);
        $destination_connector_id_opt = is_null($destination_connector_id) ? 'NULL' : intval($destination_connector_id);

    	// Note that the code below will intercept an attempt to create duplicate
    	// pinlist name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired pinlist within
    	// that (new) transaction.
    	//
    	try {
    		$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_pinlist VALUES(NULL,'{$name_escaped}','{$documentation_escaped}',{$created->to64()},'{$uid_escaped}',{$cable_id_opt},{$origin_connector_id_opt},{$destination_connector_id_opt})"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
    	return $this->find_dict_pinlist_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function update_dict_pinlist( $pinlist_id, $documentation ) {
    	$documentation_escaped = $this->connection->escape_string( trim( $documentation ));
   		$this->connection->query (
   			"UPDATE {$this->connection->database}.dict_pinlist SET documentation='{$documentation_escaped}' WHERE id={$pinlist_id}"
   		);
        return $this->find_dict_pinlist_by_id($pinlist_id);
    }
    public function delete_dict_pinlist_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal pinlist type name. A valid non-empty string is expected." );
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_pinlist WHERE name='{$name_escaped}'" );
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
    

    public function add_dict_routing( $name, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal routing name. A valid non-empty string is expected." );
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// routing name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired routing within
    	// that (new) transaction.
    	//
    	try {
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_routing VALUES(NULL,'{$name_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
	    return $this->find_dict_routing_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_dict_routing_by_id( $id ) {
    	$this->delete_dict_routing_by_( "id={$id}" );
    }
    public function delete_dict_routing_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal routing name. A valid non-empty string is expected." );
    	$this->delete_dict_routing_by_( "name='{$name_escaped}'" );
    }
    private function delete_dict_routing_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_routing WHERE {$condition}" );
    }

    public function add_dict_device_location( $name, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal device location name. A valid non-empty string is expected." );
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
    			"INSERT INTO {$this->connection->database}.dict_device_location VALUES(NULL,'{$name_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
	    return $this->find_dict_device_location_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_dict_device_location_by_id( $id ) {
    	$this->delete_dict_device_location_by_( "id={$id}" );
    }
    public function delete_dict_device_location_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal device location name. A valid non-empty string is expected." );
    	$this->delete_dict_device_location_by_( "name='{$name_escaped}'" );
    }
    private function delete_dict_device_location_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_device_location WHERE {$condition}" );
    }
    public function delete_dict_device_region_by_id( $id ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_device_region WHERE id={$id}" );
    }
    public function delete_dict_device_component_by_id( $id ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_device_component WHERE id={$id}" );
    }




    public function add_dict_instr( $name, $created, $uid ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal instr name. A valid non-empty string is expected." );
    	$uid_escaped = $this->connection->escape_string( trim( $uid ));
    	if( $uid_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal UID. A valid non-empty string is expected." );

    	// Note that the code below will intercept an attempt to create duplicate
    	// instr name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired instr within
    	// that (new) transaction.
    	//
    	try {
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.dict_instr VALUES(NULL,'{$name_escaped}',{$created->to64()},'{$uid_escaped}')"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
	    return $this->find_dict_instr_by_( "id IN (SELECT LAST_INSERT_ID())" );
    }
    public function delete_dict_instr_by_id( $id ) {
    	$this->delete_dict_instr_by_( "id={$id}" );
    }
    public function delete_dict_instr_by_name( $name ) {
    	$name_escaped = $this->connection->escape_string( trim( $name ));
    	if( $name_escaped == '' )
    		throw new NeoCaptarException (
    			__METHOD__, "illegal instr name. A valid non-empty string is expected." );
    	$this->delete_dict_instr_by_( "name='{$name_escaped}'" );
    }
    private function delete_dict_instr_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.dict_instr WHERE {$condition}" );
    }


    public function add_project( $owner, $title, $description, $created_time, $due_time, $modified_time ) {

        $owner_escaped = $this->connection->escape_string( trim( $owner ));
        if($owner_escaped == '') throw new NeoCaptarException( __METHOD__, "illegal owner. A valid non-empty string is expected." );

        $title_escaped = $this->connection->escape_string( trim( $title ));
        if($title_escaped == '') throw new NeoCaptarException( __METHOD__, "illegal title. A valid non-empty string is expected." );

        $description_escaped = $this->connection->escape_string( trim( $description ));

    	// Note that the code below will intercept an attempt to create duplicate
    	// project title name. If a conflict will be detected then the code will return null
    	// to indicate a proble. Then it's up to the caller how to deal with this
    	// situation. Usually, a solution is to commit the current transaction,
    	// start another one and make a read attempt for the desired project within
    	// that (new) transaction.
    	//
    	try {
            $jobnumber = '';    // Leave it like this for now. If we'll succeed then we'll
                                // set it for real.
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.project VALUES(NULL,'{$owner_escaped}','{$title_escaped}',job='{$jobnumber}','{$description_escaped}',{$created_time->to64()},{$due_time->to64()},{$modified_time->to64()})"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
        $project = $this->find_project_by_( "id IN (SELECT LAST_INSERT_ID())" );
        $jobnumber = $this->allocate_jobnumber($project);
        $this->connection->query ("UPDATE {$this->connection->database}.project SET job='{$jobnumber}' WHERE id={$project->id()}");

        $this->add_project_event($project, 'Created', array('Empty project created'));
        $this->add_notification_event4project('on_project_create', $project);

        return $project;
    }
    public function  update_project($id, $owner, $title, $description, $due_time) {

        $project_old = $this->find_project_by_id($id);

        $owner_escaped = $this->connection->escape_string( trim( $owner ));
        if($owner_escaped == '') throw new NeoCaptarException( __METHOD__, "illegal owner. A valid non-empty string is expected." );

        $title_escaped = $this->connection->escape_string( trim( $title ));
        if($title_escaped == '') throw new NeoCaptarException( __METHOD__, "illegal title. A valid non-empty string is expected." );

        $description_escaped = $this->connection->escape_string( trim( $description ));

        $sql = "UPDATE {$this->connection->database}.project SET owner='{$owner_escaped}',title='{$title_escaped}',description='{$description_escaped}',due_time={$due_time->to64()} WHERE id={$id}";
        $this->connection->query($sql);

        $project_new = $this->find_project_by_id($id);

        $comments = array();
        if( $project_old->owner() != $project_new->owner()) {
            array_push($comments, "Owner: '{$project_old->owner()}' -> '{$project_new->owner()}'");
            $this->add_notification_event4project('on_project_deassign', $project_old);
            $this->add_notification_event4project('on_project_assign',   $project_new);
        }
        if( $project_old->title() != $project_new->title()) {
            array_push($comments, "Ttile: '{$project_old->title()}' -> '{$project_new->title()}'");
        }
        if( $project_old->description() != $project_new->description()) {
            array_push($comments, "Description: '{$project_old->description()}' -> '{$project_new->description()}'");
        }
        if( $project_old->due_time()->to64() != $project_new->due_time()->to64()) {
            array_push($comments, "Due time: '{$project_old->due_time()->toStringShort()}' -> '{$project_new->due_time()->toStringShort()}'");
        }
        $this->add_project_event($project_new, 'Updated', $comments);

        return $project_new;
    }
    public function delete_project_by_id($id) {
        $project = $this->find_project_by_id($id);
        if( is_null($project)) return;
        $this->add_notification_event4project('on_project_delete', $project);
    	$this->delete_project_by_( "id={$id}" );
    }
    private function delete_project_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.project WHERE {$condition}" );
    }

    public function delete_cable_by_id( $id ) {
        $cable = $this->find_cable_by_id($id);
        if( is_null($cable)) return;
        $this->add_project_event($cable->project(), 'Delete Cable', $cable->dump2array());
        $this->add_notification_event4cable('on_cable_delete', $cable,  $cable->dump2array());
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.cable WHERE id={$id}" );
    }

    private function allocate_cablenumber($location, $cable_id) {
        $location = trim($location);
        foreach( $this->cablenumber_prefixes() as $prefix )
            if( in_array($location, $prefix->locations()))
                return $prefix->allocate_cable_number($cable_id,AuthDB::instance()->authName());
        return null;
    }

    public function add_jobnumber_allocation($owner, $prefix, $first, $last) {
        $owner_escaped = $this->connection->escape_string( trim( $owner ));
        if($owner_escaped == '')
            throw new NeoCaptarException( __METHOD__, "illegal project owner. A valid non-empty string is expected." );

        if(strlen($prefix) > 3)
            throw new NeoCaptarException( __METHOD__, "illegal prefix. The string length must not exceed exactly 3 symbols" );

        if( intval($first) > intval($last))
            throw new NeoCaptarException( __METHOD__, "invalid range of numbers. The last number must be strictly greater than the first one."); 

        $this->connection->query("INSERT INTO {$this->connection->database}.jobnumber VALUES(NULL,'{$owner_escaped}','{$prefix}',{$first},{$last})");
        return $this->find_jobnumber_allocation_by_('id IN (SELECT LAST_INSERT_ID())');
    }
    public function update_jobnumber($j,$first,$last,$prefix) {
        $what2update = '';
        if( !is_null($first)) {
            if( $what2update != '') $what2update .= ',';
            $what2update .= " first={$first}";
        }
        if( !is_null($last)) {
            if( $what2update != '') $what2update .= ',';
            $what2update .= " last={$last}";
        }
        if( !is_null($prefix)) {
            if( $what2update != '') $what2update .= ',';
            $prefix_escaped = $this->connection->escape_string(trim($prefix));
            $what2update .= " prefix='{$prefix_escaped}'";
        }
        if( $what2update == '') return $j;
        $this->connection->query("UPDATE {$this->connection->database}.jobnumber SET {$what2update} WHERE id={$j->id()}");
        return $this->find_jobnumber_allocation_by_id($j->id());
    }
    public function allocate_jobnumber($project) {
        $owner      = $project->owner();
        $project_id = $project->id();
        $ja = $this->find_jobnumber_allocation_by_owner($owner);
        if(is_null($ja))
            throw new NeoCaptarException(__METHOD__,"project owner '{$owner}' is not configured to assume job numbers");
        if(strlen($ja->prefix()) != 3)
            throw new NeoCaptarException(__METHOD__,"job number prefix is not set for project owner: {$owner}");

        // Check if there is a number associated with the project. If not then make
        // the new allocation.
        //
        $jobnumber = $ja->find_jobnumber($project_id);
        if( is_null($jobnumber)) {

            $jobnumber = $ja->next_available();
            if(!$jobnumber) return null;

            $allocation_time_64       = LusiTime::now()->to64();
            $allocated_by_uid_escaped = $this->connection->escape_string( trim( AuthDB::instance()->authName()));

            $this->connection->query("INSERT {$this->connection->database}.jobnumber_allocation VALUES ({$ja->id()},{$project_id},{$jobnumber},{$allocation_time_64},'{$allocated_by_uid_escaped}')");
        }
        return sprintf("%3s%03d",$ja->prefix(),$jobnumber);
    }
    private static $cable_status2rank = array(
        'Planned'      => 0,
        'Registered'   => 1,
        'Labeled'      => 2,
        'Fabrication'  => 3,
        'Ready'        => 4,
        'Installed'    => 5,
        'Commissioned' => 6,
        'Damaged'      => 7,
        'Retired'      => 8
    );
    private static function cable_status2rank($status) {
        if( array_key_exists($status, NeoCaptar::$cable_status2rank)) return NeoCaptar::$cable_status2rank[$status];
        throw new NeoCaptarException(__METHOD__, "unknown cable status: {$status}");
    }
    private static function compare_cable_status_ranks($lhs_status, $rhs_status) {
        return NeoCaptar::cable_status2rank($lhs_status) -
               NeoCaptar::cable_status2rank($rhs_status);
    }
    private function change_cable_status($cable, $new_status, $user_comments="") {
        if(is_null($cable)) throw new NeoCaptarException( __METHOD__, "invalid cable passed into the method." );
        $this->connection->query("UPDATE {$this->connection->database}.cable SET status='{$new_status}' WHERE id={$cable->id()}");
        $new_cable = $this->find_cable_by_id($cable->id());
        if( is_null($new_cable)) return null;
        $comments = array("User comments: {$user_comments}");
        if( NeoCaptar::compare_cable_status_ranks($new_cable->status(), $cable->status()) < 0 )
            array_push(
                $comments,
                "Cable status reverted from {$cable->status()} back to {$new_cable->status()}"
            );
        $this->add_cable_event($new_cable,$new_cable->status(),$comments);
        return $new_cable;

    }
    public function register_cable($cable, $comments="") {
        if(is_null($cable))
            throw new NeoCaptarException( __METHOD__, "invalid cable passed into the method." );

        $cablenumber = $this->allocate_cablenumber($cable->origin_loc(), $cable->id());
        if( is_null($cablenumber))
            throw new NeoCaptarException(__METHOD__, "failed to allocate a cable number");

        $this->connection->query("UPDATE {$this->connection->database}.cable SET cable='{$cablenumber}' WHERE id={$cable->id()}");

        $new_cable = $this->change_cable_status($cable, 'Registered', $comments);
        $this->add_notification_event4cable('on_register', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }

    public function label_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Labeled', $comments);
        $this->add_notification_event4cable('on_label', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function fabricate_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Fabrication', $comments);
        $this->add_notification_event4cable('on_fabrication', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function ready_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Ready', $comments);
        $this->add_notification_event4cable('on_ready', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function install_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Installed', $comments);
        $this->add_notification_event4cable('on_install', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function commission_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Commissioned', $comments);
        $this->add_notification_event4cable('on_commission', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function damage_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Damaged', $comments);
        $this->add_notification_event4cable('on_damage', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function retire_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Retired', $comments);
        $this->add_notification_event4cable('on_retire', $cable, "Cable number:  {$new_cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_register_cable($cable, $comments="") {
        $this->connection->query("UPDATE {$this->connection->database}.cable SET cable='' WHERE id={$cable->id()}");
        $new_cable = $this->change_cable_status($cable, 'Planned', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_label_cable($cable, $comments="") {
        $new_revision = $cable->revision() + 1;
        $this->connection->query("UPDATE {$this->connection->database}.cable SET revision={$new_revision} WHERE id={$cable->id()}");
        $new_cable = $this->change_cable_status($cable, 'Registered', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_fabricate_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Labeled', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_ready_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Fabrication', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_install_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Ready', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_commission_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Installed', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_damage_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Commissioned', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    public function un_retire_cable($cable, $comments="") {
        $new_cable = $this->change_cable_status($cable, 'Damaged', $comments);
        $this->add_notification_event4cable('on_cable_state_reversed', $cable, "Cable number:  {$cable->cable()} \nUser comments: {$comments}");
        return $new_cable;
    }
    private function add_event_impl($scope, $scope_id, $event, $comments) {
        $event_time = LusiTime::now()->to64();
        $event_uid  = $this->connection->escape_string(trim( AuthDB::instance()->authName()));
        $event      = $this->connection->escape_string($event);
        $table      = $scope ? "{$scope}_history" : "history";
        if( $scope ) $this->connection->query("INSERT INTO {$this->connection->database}.{$table} VALUES(NULL,{$scope_id},{$event_time},'{$event_uid}','{$event}')");
        else         $this->connection->query("INSERT INTO {$this->connection->database}.{$table} VALUES(NULL,            {$event_time},'{$event_uid}','{$event}')");
        $attr = mysql_fetch_array( $this->connection->query("SELECT LAST_INSERT_ID() AS `id`"), MYSQL_ASSOC );
        $event_id = $attr['id'];
        foreach( $comments as $comment )
            $this->connection->query("INSERT INTO {$this->connection->database}.{$table}_comments VALUES({$event_id},'{$this->connection->escape_string($comment)}')");
    }
    public function add_event($event, $comments) {
        $this->add_event_impl(null, null, $event, $comments);
    }
    public function add_cable_event($cable, $event, $comments) {
        $this->add_event_impl('cable', $cable->id(), $event, $comments);
    }
    public function add_project_event($project, $event, $comments) {
        $this->add_event_impl('project', $project->id(), $event, $comments);
    }
    public function add_user($uid, $name, $role) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
        $role_uc = strtoupper(trim($role));
        $name_escaped = $this->connection->escape_string(trim($name));
        $added_time = LusiTime::now()->to64();
        $added_uid_escaped = $this->connection->escape_string(trim( AuthDB::instance()->authName()));
        $sql = "INSERT INTO {$this->connection->database}.user VALUES('{$uid_escaped}','{$role_uc}','{$name_escaped}',{$added_time},'{$added_uid_escaped}',NULL)";
        $this->connection->query($sql);
        return $this->find_user_by_uid($uid);
    }
    public function delete_user($uid) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
        $this->connection->query("DELETE FROM {$this->connection->database}.user WHERE uid='{$uid_escaped}'");
        $this->connection->query("DELETE FROM {$this->connection->database}.notify WHERE uid='{$uid_escaped}'");
    }
    public function update_current_user_activity() {
        $user = $this->current_user();
        if(is_null($user)) return;
        $uid_escaped = $this->connection->escape_string(trim($user->uid()));
        $current_time = LusiTime::now()->to64();
        $sql = "UPDATE {$this->connection->database}.user SET last_active_time={$current_time} WHERE uid='{$uid_escaped}'";
        $this->connection->query($sql);
    }

    public function add_notification($uid, $event_type_id, $enabled) {
        $uid_escaped  = $this->connection->escape_string(trim($uid));
        $enabled_flag = $enabled ? 'ON' : 'OFF';
        $this->connection->query("INSERT INTO {$this->connection->database}.notify VALUES(NULL,'{$uid_escaped}',{$event_type_id},'{$enabled_flag}')");
        return $this->find_notification_by_('id IN (SELECT LAST_INSERT_ID())');       
    }
    public function update_notification($id, $enabled) {
        $enabled_flag = $enabled ? 'ON' : 'OFF';
        $this->connection->query("UPDATE {$this->connection->database}.notify SET enabled='{$enabled_flag}' WHERE id={$id}");
        return $this->find_notification_by_id($id);       
    }
    public function update_notification_schedule($recipient, $policy) {
        $recipient_escaped = $this->connection->escape_string(trim($recipient));
        $policy_escaped    = $this->connection->escape_string(trim($policy));
        $this->connection->query("UPDATE {$this->connection->database}.notify_schedule SET mode='{$policy_escaped}' WHERE recipient='{$recipient_escaped}'");
    }
    public function delete_notification_event($id) {
        $this->connection->query("DELETE FROM {$this->connection->database}.notify_queue WHERE id={$id}");
    }
    public function submit_notification_event($id) {
        $entry = $this->find_notify_queue_entry_by_id($id);
        if( is_null($entry)) return;
        switch($entry->event_type()->scope()) {
            case 'CABLE':
                $cable_id      = 0;
                $cable_info    = '';
                $project_title = '<unknown>';
                $project_owner = '<unknown>';
                $extra = $entry->extra();
                if( !is_null($extra)) {
                    $cable_id      = $extra['cable_id'];
                    $cable_info    = $extra['cable_info'];
                    $project_title = $extra['project_title'];
                    $project_owner = $extra['project_owner_uid'];
                }
                $this->notify4cable (
                    $entry->recipient_uid(),
                    $entry->event_type()->description(),
                    $cable_id, $cable_info,
                    $project_title, $project_owner,
                    $entry->originator_uid(),
                    $entry->event_time());
                break;
            case 'PROJECT':
                $project_id       = 0;
                $project_title    = '<unknown>';
                $project_owner    = '<unknown>';
                $project_due_time = '<unknown>';
                $extra = $entry->extra();
                if( !is_null($extra)) {
                    $project_id       = $extra['project_id'];
                    $project_title    = $extra['project_title'];
                    $project_owner    = $extra['project_owner_uid'];
                    $project_due_time = $extra['project_due_time']->toStringDay();
                }
                $this->notify4project (
                    $entry->recipient_uid(),
                    $entry->event_type()->description(),
                    $project_id, $project_title, $project_owner, $project_due_time,
                    $entry->originator_uid(),
                    $entry->event_time());
                break;
        }
        $this->delete_notification_event($id);
    }
    public function add_notification_event4project($event_name, $project) {

        $event_name = trim($event_name);
        $event_time = LusiTime::now();

        $originator_uid = trim(AuthDB::instance()->authName());

        // Look for who might be interested in the event
        //
        $recipients = array();
        foreach( $this->notifications() as $notify ) {

            // Skip notifications which aren't enabled regardless of what
            // they're abou.
            //
            if( !$notify->enabled()) continue;

            // Skip unrelated notifications
            //
            $event_type = $notify->event_type();

            if( !(( $event_type->scope() == 'PROJECT' ) && ( $event_type->name() == $event_name ))) continue;

            // Skip project managers who aren't supposed to be interested in someone
            // else's projects.
            //
            $recipient_uid  = $notify->uid();
            $recipient_type = $event_type->recipient();

            if(( $recipient_type == 'PROJMANAGER' ) && ( $project->owner() != $recipient_uid )) continue;

            array_push(
                $recipients,
                array(
                    'recipient_uid' => $recipient_uid,
                    'event_type'    => $event_type
                )
            );
        }

        // Evaluate each recipient to see if an instant event notification
        // has to be sent now, or it has to be placed into the wait queue.
        //
        $schedule = $this->notify_schedule();

        foreach( $recipients as $entry ) {
            $recipient_uid = $entry['recipient_uid'];
            $event_type    = $entry['event_type'];
            switch($schedule[$event_type->recipient()]) {
                case 'INSTANT': $this->notify4project_instant( $recipient_uid, $event_type->description(), $project, $originator_uid, $event_time ); break;
                case 'DELAYED': $this->notify4project_delayed( $recipient_uid, $event_type->id(),          $project, $originator_uid, $event_time ); break;
            }
        }
    }
    private function notify4project_instant (
        $recipient_uid,
        $event_type_description,
        $project,
        $originator_uid,
        $event_time ) {

        $this->notify4project (
            $recipient_uid,
            $event_type_description,
            $project->id(), $project->title(), $project->owner(), $project->due_time()->toStringDay(),
            $originator_uid,
            $event_time );
    }
    private function notify4project_delayed (
        $recipient_uid,
        $event_type_id,
        $project,
        $originator_uid,
        $event_time ) {

        $recipient_uid_escaped  = $this->connection->escape_string(trim($recipient_uid));
        $originator_uid_escaped = $this->connection->escape_string(trim($originator_uid));

        /* TODO: Store this in the database table.
         */
        $project_title_escaped = $this->connection->escape_string( trim( $project->title()));
        $project_owner_escaped = $this->connection->escape_string( trim( $project->owner()));

        $this->connection->query(
            "INSERT INTO {$this->connection->database}.notify_queue".
            " VALUES(NULL,{$event_type_id},{$event_time->to64()},'{$originator_uid_escaped}','{$recipient_uid_escaped}')"
        );
        $entry = $this->find_notify_queue_entry_by_('id IN (SELECT LAST_INSERT_ID())');
        $this->connection->query(
            "INSERT INTO {$this->connection->database}.notify_queue_project".
            " VALUES({$entry->id()},{$project->id()},'{$project_title_escaped}','{$project_owner_escaped}',{$project->due_time()->to64()})"
        );
    }
    private function notify4project (
        $recipient_uid,
        $event_type_description,
        $project_id, $project_title, $project_owner, $project_due_time,
        $originator_uid,
        $event_time ) {

        $project_owner_name = '';
        $user = $this->find_user_by_uid($project_owner);
        if( !is_null($user)) $project_owner_name = "({$user->name()})";

        $originator_name = '';
        $user = $this->find_user_by_uid($originator_uid);
        if( !is_null($user)) $originator_name = "({$user->name()})";

        $address = "{$recipient_uid}@slac.stanford.edu";
        $subject = 'Project Event Notification';
        $body =<<<HERE
This is an automated notification message on the following event:
        
  '{$event_type_description}'

Project title: {$project_title}
Project owner: {$project_owner} {$project_owner_name}
Project due:   {$project_due_time}

Project URL (if the project is still available):

  https://pswww.slac.stanford.edu/apps-dev/neocaptar/?app=projects:search&project_id={$project_id}

The change was made by user:       {$originator_uid} {$originator_name}
The time when the change was made: {$event_time->toStringShort()}


    **********************************************************************************
    * The message was sent by the automated notification system because your account *
    * was subscribed to recieve notifications on certain cable management events.    *
    * You can manage your subscriptions using the following URL:                     *
    *                                                                                *
    *   https://pswww.slac.stanford.edu/apps-dev/neocaptar/?app=admin:notifications  *
    *                                                                                *
    **********************************************************************************

HERE;
        NeoCaptar::notify( $address, $subject, $body );
    }

    public function add_notification_event4cable($event_name, $cable, $cable_info) {

        $event_name = trim($event_name);
        $event_time = LusiTime::now();

        $originator_uid = trim(AuthDB::instance()->authName());

        // Look for who might be interested in the event
        //
        $recipients = array();
        foreach( $this->notifications() as $notify ) {

            // Skip notifications which aren't enabled regardless of what
            // they're abou.
            //
            if( !$notify->enabled()) continue;

            // Skip unrelated notifications
            //
            $event_type = $notify->event_type();

            if( !(( $event_type->scope() == 'CABLE' ) && ( $event_type->name() == $event_name ))) continue;

            // Skip project managers who aren't supposed to be interested in someone
            // else's projects.
            //
            $recipient_uid  = $notify->uid();
            $recipient_type = $event_type->recipient();

            if(( $recipient_type == 'PROJMANAGER' ) && ( $cable->project()->owner() != $recipient_uid )) continue;

            array_push(
                $recipients,
                array(
                    'recipient_uid' => $recipient_uid,
                    'event_type'    => $event_type
                )
            );
        }

        // Evaluate each recipient to see if an instant event notification
        // has to be sent now, or it has to be placed into the wait queue.
        //
        $schedule = $this->notify_schedule();

        foreach( $recipients as $entry ) {
            $recipient_uid = $entry['recipient_uid'];
            $event_type    = $entry['event_type'];
            switch($schedule[$event_type->recipient()]) {
                case 'INSTANT': $this->notify4cable_instant( $recipient_uid, $event_type->description(), $cable, $cable_info, $originator_uid, $event_time ); break;
                case 'DELAYED': $this->notify4cable_delayed( $recipient_uid, $event_type->id(),          $cable, $cable_info, $originator_uid, $event_time ); break;
            }
        }
    }
    private function notify4cable_instant (
        $recipient_uid,
        $event_type_description,
        $cable, $cable_info,
        $originator_uid,
        $event_time ) {

        $this->notify4cable (
            $recipient_uid,
            $event_type_description,
            $cable->id(), $cable_info,
            $cable->project()->title(), $cable->project()->owner(),
            $originator_uid,
            $event_time );
    }
    private function notify4cable_delayed (
        $recipient_uid,
        $event_type_id,
        $cable, $cable_info,
        $originator_uid,
        $event_time ) {

        $recipient_uid_escaped  = $this->connection->escape_string(trim($recipient_uid));
        $originator_uid_escaped = $this->connection->escape_string(trim($originator_uid));

        /* TODO: Store this in the database table.
         */
        $cable_info_escaped    = $this->connection->escape_string( trim( $cable_info));
        $project_title_escaped = $this->connection->escape_string( trim( $cable->project()->title()));
        $project_owner_escaped = $this->connection->escape_string( trim( $cable->project()->owner()));

        $this->connection->query(
            "INSERT INTO {$this->connection->database}.notify_queue".
            " VALUES(NULL,{$event_type_id},{$event_time->to64()},'{$originator_uid_escaped}','{$recipient_uid_escaped}')"
        );
        $entry = $this->find_notify_queue_entry_by_('id IN (SELECT LAST_INSERT_ID())');
        $this->connection->query(
            "INSERT INTO {$this->connection->database}.notify_queue_cable".
            " VALUES({$entry->id()},{$cable->id()},'{$cable_info_escaped}','{$project_title_escaped}','{$project_owner_escaped}')"
        );
    }
    private function notify4cable (
        $recipient_uid,
        $event_type_description,
        $cable_id, $cable_info,
        $project_title, $project_owner,
        $originator_uid,
        $event_time ) {

        $project_owner_name = '';
        $user = $this->find_user_by_uid($project_owner);
        if( !is_null($user)) $project_owner_name = "({$user->name()})";

        $originator_name = '';
        $user = $this->find_user_by_uid($originator_uid);
        if( !is_null($user)) $originator_name = "({$user->name()})";

        $address = "{$recipient_uid}@slac.stanford.edu";
        $subject = 'Cable Event Notification';
        $body =<<<HERE
This is an automated notification message on the following event:
        
  '{$event_type_description}'
  
Project title: {$project_title}
Project owner: {$project_owner} {$project_owner_name}

{$cable_info}
Cable URL:     https://pswww.slac.stanford.edu/apps-dev/neocaptar/?app=search:cables&cable_id={$cable_id}

The change was made by user:       {$originator_uid} {$originator_name}
The time when the change was made: {$event_time->toStringShort()}


    **********************************************************************************
    * The message was sent by the automated notification system because your account *
    * was subscribed to recieve notifications on certain cable management events.    *
    * You can manage your subscriptions using the following URL:                     *
    *                                                                                *
    *   https://pswww.slac.stanford.edu/apps-dev/neocaptar/?app=admin:notifications  *
    *                                                                                *
    **********************************************************************************

HERE;
        NeoCaptar::notify( $address, $subject, $body );
    }
}

?>
