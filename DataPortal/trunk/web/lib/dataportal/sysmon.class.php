<?php

namespace DataPortal;

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;

use LusiTime\LusiTime;

/**
 * Class SysMon encapsulates operations with the PCDS systems monitoring database
 */
class SysMon {

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
        if( is_null( SysMon::$instance )) SysMon::$instance =
        	new SysMon (
        		SYSMON_DEFAULT_HOST,
				SYSMON_DEFAULT_USER,
				SYSMON_DEFAULT_PASSWORD,
				SYSMON_DEFAULT_DATABASE );
        return SysMon::$instance;
    }
	
    /* Parameters of the object
     */
    private $host;
    private $user;
    private $password;
    public  $database;

    /* Current state of the object
     */
    private $link;
    private $in_transaction = false;

    /**
     * Constructor
     *
     * The constructor won't make any actual connection attempts. This will be deffered
     * to operations dealing with queries, transactions, etc.
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database 
     */
    public function __construct ( $host, $user, $password, $database ) {
        $this->host      = $host;
        $this->user      = $user;
        $this->password  = $password;
        $this->database  = $database;
    }

    /**
     * Destructor
     */
    public function __destruct () {

    	// Do not close this connection from here because it might be shared
    	// with other instances of the class (also from other APIs).
    	//
        //if( isset( $this->link )) mysql_close( $this->link );
    }

    /* ================================================================
     *   METHODS FOR RETREIVING VARIOUS INFORMATION FROM THE DATABASE
     * ================================================================
     */

    public function beamtime_config() {
	    $config = array();
   		$this->connect();
    	$result = $this->query( "SELECT * FROM beamtime_config" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
        	$row   = mysql_fetch_array( $result, MYSQL_ASSOC );
            $param = trim($row['param']);
            $value = trim($row['value']);
            switch($param) {
                case 'min_gap_width_sec':              $config[$param] = intval($value); break;
                case 'last_run_begin_time': if($value) $config[$param] = LusiTime::from64($value); break;
                default:                               $config[$param] = $value; break;
            }        	
        }
        return $config;
    }

    public function beamline_runs( $begin_time=null, $end_time=null ) {
        $list = array();
   		$this->connect();
        $interval = '';
        if( !is_null($begin_time)) $interval .= "WHERE begin_time >= {$begin_time->to64()}";
        if( !is_null($end_time  )) $interval .= ($interval == '' ? " WHERE " : " AND ")." begin_time < {$end_time->to64()}";
    	$result = $this->query( "SELECT * FROM beamtime_runs {$interval} ORDER BY begin_time" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
        	array_push(
                $list,
                new SysMonBeamTimeRun(
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }

    public function beamline_gaps( $begin_time=null, $end_time=null ) {
        $list = array();
   		$this->connect();
        $interval = '';
        if( !is_null($begin_time)) $interval .= "WHERE begin_time >= {$begin_time->to64()}";
        if( !is_null($end_time  )) $interval .= ($interval == '' ? " WHERE " : " AND ")." begin_time < {$end_time->to64()}";
    	$result = $this->query( "SELECT * FROM beamtime_gaps {$interval} ORDER BY begin_time" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
        	array_push(
                $list,
                new SysMonBeamTimeGap(
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function beamline_comments( $begin_time=null, $end_time=null ) {
        $list = array();
   		$this->connect();
        $interval = '';
        if( !is_null($begin_time)) $interval .= "WHERE gap_begin_time >= {$begin_time->to64()}";
        if( !is_null($end_time  )) $interval .= ($interval == '' ? " WHERE " : " AND ")." gap_begin_time < {$end_time->to64()}";
    	$result = $this->query( "SELECT * FROM beamtime_comments {$interval} ORDER BY gap_begin_time" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
        	array_push(
                $list,
                new SysMonBeamTimeComment(
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }
    public function beamline_systems() {
        $list = array();
   		$this->connect();
    	$result = $this->query( "SELECT DISTINCT system FROM beamtime_comments WHERE system != ''" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        	array_push(
                $list,
                $attr['system']);
        }
        return $list;
    }
    public function beamtime_comment_at($begin_time) {
        $this->connect();
        $result = $this->query( "SELECT * FROM beamtime_comments WHERE gap_begin_time={$begin_time->to64()}" );
    	$nrows = mysql_numrows( $result );
        if( !$nrows ) return null;
        if( $nrows != 1 )
            throw new DataPortalException (
                __METHOD__,
                "unexpected result set returned by the query" );
        return new SysMonBeamTimeComment(
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }
    
    /* =================================
     *   DATABASE MODIFICATION METHODS
     * =================================
     */

    public function update_beamtime_config($config) {
        $this->connect();
        foreach( $config as $param => $value ) {

            $param_trimmed = strtolower(trim($param));
            $param_escaped = $this->escape_string($param_trimmed);

            $value_escaped = '';

            switch($param_trimmed) {

                case 'min_gap_width_sec':
                    if(!$value)
                        throw new DataPortalException (
                            __METHOD__,
                            "configuration parameter {$param_trimmed} must have a non-empty value");
                    $value_escaped = intval($value);
                    break;

                case 'last_run_begin_time':
                    if(!$value)
                        throw new DataPortalException (
                            __METHOD__,
                            "configuration parameter {$param_trimmed} must have a non-empty value");
                    $value_escaped = $value->to64();
                    break;

                default:
                    $value_escaped = is_null($value) ? "NULL" : "'".$this->escape_string("{$value}")."'";
                    break;
            }
            $old_config = $this->beamtime_config();
            $this->query(
                array_key_exists($param_trimmed, $old_config) ?
                    "UPDATE beamtime_config SET value={$value_escaped} WHERE param='{$param_escaped}'" :
                    "INSERT INTO beamtime_config VALUES('{$param_escaped}',{$value_escaped})"
            );
        }
    }

    public function beamtime_clear_all() {
        $this->connect();
        $this->query('DELETE FROM beamtime_gaps');
        $this->query('DELETE FROM beamtime_runs');
    }
    public function beamtime_clear_from($begin_time) {
        $this->connect();
        $this->query("DELETE FROM beamtime_gaps WHERE begin_time >= {$begin_time->to64()}");
        $this->query("DELETE FROM beamtime_runs WHERE begin_time >= {$begin_time->to64()}");
    }

    public function add_beamtime_run($begin, $end, $exper_id, $runnum, $exper_name, $instr_name) {
        $this->connect();
        $exper_name_escaped = $this->escape_string(trim($exper_name));
        $instr_name_escaped = $this->escape_string(trim($instr_name));
        $this->query(
            "INSERT INTO beamtime_runs VALUES({$begin->to64()},{$end->to64()},{$exper_id},{$runnum},'{$exper_name_escaped}','{$instr_name_escaped}')"
        );
    }
    public function add_beamtime_gap($begin, $end) {
        $this->connect();
        $this->query(
            "INSERT INTO beamtime_gaps VALUES({$begin->to64()},{$end->to64()})"
        );
    }
    public function beamtime_set_gap_comment($gap_begin_time, $comment, $system, $post_time, $posted_by_uid) {
        $this->connect();
        $posted_by_uid_escaped = $this->escape_string(trim($posted_by_uid));
        $comment_escaped       = $this->escape_string(trim($comment));
        $system_escaped        = $this->escape_string(trim($system));
        $this->query(is_null( $this->beamtime_comment_at( $gap_begin_time )) ?
            "INSERT INTO beamtime_comments VALUES({$gap_begin_time->to64()},{$post_time->to64()},'{$posted_by_uid_escaped}','{$comment_escaped}','{$system_escaped}')" :
            "UPDATE beamtime_comments SET post_time={$post_time->to64()}, posted_by_uid='{$posted_by_uid_escaped}', comment='{$comment_escaped}', system='{$system_escaped}' WHERE gap_begin_time={$gap_begin_time->to64()}"
        );
    }
    public function beamtime_clear_gap_comment($gap_begin_time) {
        $this->connect();
        $this->query("DELETE FROM beamtime_comments WHERE gap_begin_time={$gap_begin_time->to64()}");
    }
    /*
     * ================================
     *   MySQL TRANSACTION MANAGEMENT
     * ================================
     */
    public function begin () {
        $this->connect();
        if( $this->in_transaction ) return;
        $this->transaction( 'BEGIN' );
        $this->in_transaction = true;
    }

    public function commit () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'COMMIT' );
        $this->in_transaction = false;
    }

    public function rollback () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'ROLLBACK' );
        $this->in_transaction = false;
    }

    private function transaction ( $transition ) {
        if( !mysql_query( $transition, $this->link ))
            throw new DataPortalException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$transition );
    }

    /* =================
     *   MySQL QUERIES
     * =================
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new DataPortalException (
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql, $this->link );
        if( !$result )
            throw new DataPortalException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
        return $result;
    }

    /* ==========================================================
     *   MISC. OPERATIONS REQUIREING DATABASE SERVER CONNECTION
     * ==========================================================
     */
    public function escape_string( $text ) {
        return mysql_real_escape_string( $text, $this->link );  }
 
    /**
     * Make a connection if this hasn't been done yet.
     */
    private function connect () {
        if( !isset( $this->link )) {

            /* Connect to MySQL server and register the shutdown function
             * to clean up after self by doing 'ROLLBACK'. This would allow
             * to use so called 'persistent' MySQL connections.
             * 
             * NOTE: using the 'persistent' connection. This connection won't be
             * closed by 'mysql_close()'.
             */
        	$new_link = false; // true;
            $this->link = mysql_pconnect( $this->host, $this->user, $this->password, $new_link );
            if( !$this->link )
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_connect" );

            if( !mysql_select_db( $this->database, $this->link ))
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql, $this->link ))
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql, $this->link ))
                throw new DataPortalException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            register_shutdown_function( array( $this, "rollback" ));
        }
    }
}

/* =======================
 * UNIT TEST FOR THE CLASS
 * =======================
 *
try {
	$sysmon  = SysMon::instance();
	$sysmon->begin();


	$sysmon->commit();

} catch ( Exception           $e ) { print $e; }
  catch ( DataPortalException $e ) { print $e->toHtml(); }
*/
?>
