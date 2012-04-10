<?php

namespace DataPortal;

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal.inc.php' );
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
        $extra_condition = '';
        if(!is_null($title)) {
            $title_escaped = $this->connection->escape_string(trim($title));
            if($extra_condition == '') $extra_condition .= ' WHERE ';
            else                       $extra_condition .= ' AND ';
            if($title_escaped   != '') $extra_condition .= " title LIKE '%{$title_escaped}%'";
        }
        if(!is_null($owner)) {
            $owner_escaped = $this->connection->escape_string(trim($owner));
            if($extra_condition == '') $extra_condition .= ' WHERE ';
            else                       $extra_condition .= ' AND ';
            if( $owner_escaped != '' ) $extra_condition .= " owner='{$owner_escaped}'";
        }
        if(!is_null($begin)) {
            if($extra_condition == '') $extra_condition .= ' WHERE ';
            else                       $extra_condition .= ' AND ';
            $extra_condition .= " created_time >= {$begin->to64()}";
        }
        if(!is_null($end)) {
            if($extra_condition == '') $extra_condition .= ' WHERE ';
            else                       $extra_condition .= ' AND ';
            $extra_condition .= " created_time < {$end->to64()}";
        }
        $list = array();
        $sql = "SELECT * FROM {$this->connection->database}.project {$extra_condition} ORDER BY title";
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
    private function find_project_by_( $condition ) {
        $sql = "SELECT * FROM {$this->connection->database}.project WHERE {$condition} ORDER BY created_time";
        $result = $this->connection->query ( $sql );
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
        	throw new NeoCaptarException (
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
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
        $condition = null;
        $operator = array_key_exists('partial_or',$partial_search_params) ? ' OR ' : ' AND ';
        foreach( $partial_search_params as $p => $v ) {
            $p = trim($p);
            if( $p == 'partial_or') continue;
            $v = $this->connection->escape_string(trim($v));
            if(($p == '') || ($v == '')) continue;
            if( !is_null($condition)) $condition .= $operator;
            $condition .= " {$p} LIKE '%{$v}%' COLLATE latin1_general_ci";  ## case-insensitive search
        }
        if( is_null($condition))
            $condition = 'id is NOT NULL';
            /*
        	throw new NeoCaptarException (
        		__METHOD__, "no search parameters provided" );
             */
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
        		__METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
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
    public function find_cables_by_jobnumber( $jobnumber ) {
    	$jobnumber_escaped = $this->connection->escape_string(trim($jobnumber));
        return $this->find_cables_by_(
            "job='{$jobnumber_escaped}'");
    }
    public function find_cables_by_dict_cable_id($dict_cable_id) {
        $dict_cable = $this->find_dict_cable_by_id($dict_cable_id);
        if( is_null($dict_cable)) return $list;
        $cable_type_escaped = $this->connection->escape_string(trim($dict_cable->name()));
        return $this->find_cables_by_(
            "cable_type='{$cable_type_escaped}'");
    }
    public function find_cables_by_dict_connector_id($dict_connector_id) {
        $dict_connector = $this->find_dict_connector_by_id($dict_connector_id);
        if( is_null($dict_connector)) return $list;
        $cable_type_escaped     = $this->connection->escape_string(trim($dict_connector->cable()->name()));
        $connector_type_escaped = $this->connection->escape_string(trim($dict_connector->name()));
        return $this->find_cables_by_(
            "cable_type='{$cable_type_escaped}' AND ".
            "(origin_conntype='{$connector_type_escaped}' OR destination_conntype='{$connector_type_escaped}')");

    }
    public function find_cables_by_dict_pinlist_id($dict_pinlist_id) {
        $dict_pinlist = $this->find_dict_pinlist_by_id($dict_pinlist_id);
        if( is_null($dict_pinlist)) return $list;
        $cable_type_escaped     = $this->connection->escape_string(trim($dict_pinlist->connector()->cable()->name()));
        $connector_type_escaped = $this->connection->escape_string(trim($dict_pinlist->connector()->name()));
        $pinlist_escaped        = $this->connection->escape_string(trim($dict_pinlist->name()));
        return $this->find_cables_by_(
            "cable_type='{$cable_type_escaped}' AND ".
            "((origin_conntype='{$connector_type_escaped}' AND origin_pinlist='{$pinlist_escaped}') OR ".
            " (destination_conntype='{$connector_type_escaped}' AND destination_pinlist='{$pinlist_escaped}')".
            ")");
    }
    public function find_cables_by_dict_location_id($dict_location_id) {
        $dict_location = $this->find_dict_location_by_id($dict_location_id);
        if( is_null($dict_location)) return $list;
        $location_escaped = $this->connection->escape_string(trim($dict_location->name()));
        return $this->find_cables_by_(
            "origin_loc='{$location_escaped}' OR destination_loc='{$location_escaped}'");
    }
    public function find_cables_by_dict_rack_id($dict_rack_id) {
        $dict_rack = $this->find_dict_rack_by_id($dict_rack_id);
        if( is_null($dict_rack)) return $list;
        $location_escaped = $this->connection->escape_string(trim($dict_rack->location()->name()));
        $rack_escaped     = $this->connection->escape_string(trim($dict_rack->name()));
        return $this->find_cables_by_(
            "(origin_loc='{$location_escaped}' AND origin_rack='{$rack_escaped}') OR ".
            "(destination_loc='{$location_escaped}' AND destination_rack='{$rack_escaped}')");
    }
    public function find_cables_by_dict_routing_id($dict_routing_id) {
        $dict_routing = $this->find_dict_routing_by_id($dict_routing_id);
        if( is_null($dict_routing)) return $list;
        $routing_escaped = $this->connection->escape_string(trim($dict_routing->name()));
        return $this->find_cables_by_(
            "routing='{$routing_escaped}'");
    }
    public function find_cables_by_dict_instr_id($dict_instr_id) {
        $dict_instr = $this->find_dict_instr_by_id($dict_instr_id);
        if( is_null($dict_instr)) return $list;
        $instr_escaped = $this->connection->escape_string(trim($dict_instr->name()));
        return $this->find_cables_by_(
            "origin_instr='{$instr_escaped}' OR destination_instr='{$instr_escaped}'");
    }
    public function cablenumber_allocations() {
        $list = array();
        $this->update_cablenumber_allocations();
        $sql = "SELECT * FROM {$this->connection->database}.cablenumber ORDER BY location";
        $result = $this->connection->query ( $sql );
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push (
                $list,
                new NeoCaptarCableNumberAlloc (
                    $this->connection,
                    $this,
                    $attr,
                    $this->cablenumber_total_allocated($attr['location']),
                    $this->cablenumber_last_allocation($attr['location'])));
        }
        return $list;
    }

    /**
     * Look for cable numbers within  allowed range for the specified location only.
     *
     * @param string $location
     * @return int 
     */
    public function cablenumber_total_allocated($location) {
        $location_escaped = $this->connection->escape_string(trim($location));
        if( $location_escaped == '' )
            throw new NeoCaptarException (
                __METHOD__, "empty string passed where a valid location name was expected." );
        $table_c  = "{$this->connection->database}.cablenumber `c`";
        $table_ca = "{$this->connection->database}.cablenumber_allocation `ca`";
        $sql      = "SELECT COUNT(*) AS `total` FROM {$table_c}, {$table_ca} WHERE c.location='{$location_escaped}' AND ca.cablenumber_id=c.id AND c.first <= ca.cablenumber AND ca.cablenumber <= c.last";
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
     * Look for cable numbers within allowed range for the specified location only.
     *
     * @param string $location
     * @return array (keys: 'cablenumber','allocated_time','allocated_by_uid') 
     */
    private function cablenumber_last_allocation($location) {
        $location_escaped = $this->connection->escape_string(trim($location));
        if( $location_escaped == '' )
            throw new NeoCaptarException (
                __METHOD__, "empty string passed where a valid location name was expected." );
        $table_c  = "{$this->connection->database}.cablenumber `c`";
        $table_ca = "{$this->connection->database}.cablenumber_allocation `ca`";
        $sql      = "SELECT ca.cablenumber,ca.allocated_time,ca.allocated_by_uid FROM {$table_c}, {$table_ca} WHERE c.location='{$location_escaped}' AND ca.cablenumber_id=c.id AND c.first <= ca.cablenumber AND ca.cablenumber <= c.last ORDER BY cablenumber DESC LIMIT 1";
        $result   = $this->connection->query($sql);
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        return mysql_fetch_array( $result, MYSQL_ASSOC );
    }
    public function find_cablenumber_allocation_by_id($id) {
        if( intval($id) == 0 ) throw new NeoCaptarException(__METHOD__,"illegal identifier");
        return $this->find_cablenumber_allocation_by_("id={$id}");

    }
    public function find_cablenumber_allocation_by_location($location) {
    	$location_escaped = $this->connection->escape_string( trim($location));
    	if( $location_escaped == '' ) throw new NeoCaptarException(__METHOD__,"illegal location name");
        return $this->find_cablenumber_allocation_by_("location='{$location_escaped}'");
    }
    private function find_cablenumber_allocation_by_($condition) {
        $result = $this->connection->query("SELECT * FROM {$this->connection->database}.cablenumber WHERE {$condition}");
        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new NeoCaptarException (
                __METHOD__, "inconsistent result returned by the query. Database may be corrupt." );
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return new NeoCaptarCableNumberAlloc (
            $this->connection,
            $this,
            $attr,
            $this->cablenumber_total_allocated($attr['location']),
            $this->cablenumber_last_allocation($attr['location']));
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
    public function is_administrator() {
        $user = $this->current_user();
        return !is_null($user) && $user->is_administrator();
    }
    public function can_manage_projects() {
        $user = $this->current_user();
        return !is_null($user) && ($user->is_administrator() || $user->is_projmanager());
    }

    /**
     * Scan cables to get all known origin locations and create a cable numbers
     * allocation for each location not registered in the database
     * table.
     */
    public function update_cablenumber_allocations() {
        foreach( array_diff( $this->cable_origin_locations(), $this->cablenumber_locations()) as $location )
            $this->add_cablenumber_allocation($location, '', 0, 0);
    }

    /**
     * Scan projects to get all known owner names and create a job numbers
     * allocation for each owner not registered in the database.
     * table.
     */
    public function update_jobnumber_allocations() {
        foreach( array_diff( $this->project_owners(), $this->jobnumber_owners()) as $owner )
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
	    	$this->connection->query (
    			"INSERT INTO {$this->connection->database}.project VALUES(NULL,'{$owner_escaped}','{$title_escaped}','{$description_escaped}',{$created_time->to64()},{$due_time->to64()},{$modified_time->to64()})"
    		);
    	} catch( NeoCaptarException $e ) {
    		if( !is_null( $e->errno ) && ( $e->errno == NeoCaptarConnection::$ER_DUP_ENTRY )) return null;
    		throw $e;
    	}
        $project = $this->find_project_by_( "id IN (SELECT LAST_INSERT_ID())" );

        $owner_gecos = $this->find_user_by_uid($project->owner())->name();
        $due_str     = $project->due_time()->toStringDay();
        $project_url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['SERVER_NAME'].'/apps-dev/portal/neocaptar?app=projects:search&project_id='.$project->id();

        // Notify administrators on the new project
        //
        $body = <<<HERE
Title:  "{$project->title()}"
Owner:  {$project->owner()} ({$owner_gecos})
Due by: {$due_str}
URL:    {$project_url}

{$project->description()}

_________________________________________________________
The message was sent by the automated notification system
because your accont was registered as an administrator in
the PCDS Cable Management Sefvice (Neo-CAPTAR). Please,
contact PCDS Computing Management directly if you think
this was done my mistake.
HERE;
        foreach( $this->users() as $user)
            if( $user->is_administrator())
                NeoCaptar::notify( "{$user->uid()}@slac.stanford.edu", "New Project Registered in Neo-CAPTAR", $body );

        return $project;

    }
    public function  update_project($id, $owner, $title, $description, $due_time) {
        $owner_escaped = $this->connection->escape_string( trim( $owner ));
        if($owner_escaped == '') throw new NeoCaptarException( __METHOD__, "illegal owner. A valid non-empty string is expected." );

        $title_escaped = $this->connection->escape_string( trim( $title ));
        if($title_escaped == '') throw new NeoCaptarException( __METHOD__, "illegal title. A valid non-empty string is expected." );

        $description_escaped = $this->connection->escape_string( trim( $description ));

        $sql = "UPDATE {$this->connection->database}.project SET owner='{$owner_escaped}',title='{$title_escaped}',description='{$description_escaped}',due_time={$due_time->to64()} WHERE id={$id}";
        $this->connection->query($sql);

        return $this->find_project_by_id($id);
    }
    public function delete_project_by_id( $id ) {
    	$this->delete_project_by_( "id={$id}" );
    }
    private function delete_project_by_( $condition ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.project WHERE {$condition}" );
    }

    public function delete_cable_by_id( $id ) {
    	$this->connection->query ( "DELETE FROM {$this->connection->database}.cable WHERE id={$id}" );
    }

    public function add_cablenumber_allocation($location, $prefix, $first, $last) {
        $location_escaped = $this->connection->escape_string( trim( $location ));
        if($location_escaped == '')
            throw new NeoCaptarException( __METHOD__, "illegal location. A valid non-empty string is expected." );

        if(strlen($prefix) > 2)
            throw new NeoCaptarException( __METHOD__, "illegal prefix. The string length must not exit exactly 2 symbols" );

        if( intval($first) > intval($last))
            throw new NeoCaptarException( __METHOD__, "invalid range of numbers. The last number must be strictly greater than the first one."); 

        $this->connection->query("INSERT INTO {$this->connection->database}.cablenumber VALUES(NULL,'{$location_escaped}','{$prefix}',{$first},{$last})");
        return $this->find_cablenumber_allocation_by_('id IN (SELECT LAST_INSERT_ID())');
    }
    public function update_cablenumber($c,$first,$last,$prefix) {
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
        if( $what2update == '') return $c;
        $this->connection->query("UPDATE {$this->connection->database}.cablenumber SET {$what2update} WHERE id={$c->id()}");
        return $this->find_cablenumber_allocation_by_id($c->id());
    }
    public function allocate_cablenumber($location) {
        $ca = $this->find_cablenumber_allocation_by_location($location);
        if(is_null($ca))
            throw new NeoCaptarException(__METHOD__,"cable location '{$location}' is either empty or not configured to be associated with cable numbers");
        if(strlen($ca->prefix()) != 2)
            throw new NeoCaptarException(__METHOD__,"cable number prefix is not set for location: {$location}");

        $cablenumber = $ca->next_available();
        if(!$cablenumber) return null;

        $allocation_time_64       = LusiTime::now()->to64();
        $allocated_by_uid_escaped = $this->connection->escape_string( trim( AuthDB::instance()->authName()));

        $this->connection->query("INSERT {$this->connection->database}.cablenumber_allocation VALUES ({$ca->id()},{$cablenumber},{$allocation_time_64},'{$allocated_by_uid_escaped}')");

        return sprintf("%2s%05d",$ca->prefix(),$cablenumber);
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
    public function allocate_jobnumber($cable) {
        $owner      = $cable->project()->owner();
        $project_id = $cable->project()->id();
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

    private function change_cable_status($cable,$new_status) {
        if(is_null($cable)) throw new NeoCaptarException( __METHOD__, "invalid cable passed into the method." );
        $this->connection->query("UPDATE {$this->connection->database}.cable SET status='{$new_status}' WHERE id={$cable->id()}");
        $new_cable = $this->find_cable_by_id($cable->id());
        if( is_null($new_cable)) return null;
        $this->add_cable_event($new_cable,$new_cable->status());
        return $new_cable;

    }
    public function register_cable($cable) {
        if(is_null($cable))
            throw new NeoCaptarException( __METHOD__, "invalid cable passed into the method." );

        $cablenumber = $this->allocate_cablenumber($cable->origin_loc());
        if( is_null($cablenumber))
            throw new NeoCaptarException(__METHOD__, "failed to allocate a cable number");

        $jobnumber = $this->allocate_jobnumber($cable);

        $this->connection->query("UPDATE {$this->connection->database}.cable SET status='{$state}',cable='{$cablenumber}',job='{$jobnumber}' WHERE id={$cable->id()}");

        return $this->change_cable_status($cable,'Registered');
    }
    public function label_cable     ($cable) { return $this->change_cable_status($cable,'Labeled');      }
    public function fabricate_cable ($cable) { return $this->change_cable_status($cable,'Fabrication');  }
    public function ready_cable     ($cable) { return $this->change_cable_status($cable,'Ready');        }
    public function install_cable   ($cable) { return $this->change_cable_status($cable,'Installed');    }
    public function commission_cable($cable) { return $this->change_cable_status($cable,'Commissioned'); }
    public function damage_cable    ($cable) { return $this->change_cable_status($cable,'Damaged');      }
    public function retire_cable    ($cable) { return $this->change_cable_status($cable,'Retired');      }

    private function add_event($scope,$scope_id,$event) {
        $table      = "{$scope}_history";
        $event_time = LusiTime::now()->to64();
        $event_uid  = $this->connection->escape_string(trim( AuthDB::instance()->authName()));
        $event      = $this->connection->escape_string($event);
        $sql        = "INSERT INTO {$this->connection->database}.{$table} VALUES({$scope_id},{$event_time},'{$event_uid}','{$event}')";
        $this->connection->query($sql);
    }
    public function add_cable_event($cable,$event) { $this->add_event('cable',$cable->id(),$event); }

    public function add_user($uid, $name, $role) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
        $role_uc = strtoupper(trim($role));
        if(!(($role_uc == 'ADMINISTRATOR') || ($role_uc == 'PROJMANAGER')))
            throw new NeoCaptarException(__METHOD__, "no such role: {$role}");
        $name_escaped = $this->connection->escape_string(trim($name));
        $added_time = LusiTime::now()->to64();
        $added_uid_escaped = $this->connection->escape_string(trim( AuthDB::instance()->authName()));
        $sql = "INSERT INTO {$this->connection->database}.user VALUES('{$uid_escaped}','{$role_uc}','{$name_escaped}',{$added_time},'{$added_uid_escaped}',NULL)";
        $this->connection->query($sql);
        return $this->find_user_by_uid($uid);
    }
    public function delete_user($uid) {
        $uid_escaped = $this->connection->escape_string(trim($uid));
        $sql = "DELETE FROM {$this->connection->database}.user WHERE uid='{$uid_escaped}'";
        $this->connection->query($sql);
    }
    public function update_current_user_activity() {
        $user = $this->current_user();
        if(is_null($user)) return;
        $uid_escaped = $this->connection->escape_string(trim($user->uid()));
        $current_time = LusiTime::now()->to64();
        $sql = "UPDATE {$this->connection->database}.user SET last_active_time={$current_time} WHERE uid='{$uid_escaped}'";
        $this->connection->query($sql);
    }
}

?>
