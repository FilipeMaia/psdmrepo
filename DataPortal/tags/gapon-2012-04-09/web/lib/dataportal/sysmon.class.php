<?php

namespace DataPortal;

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;

use LogBook\LogBook;

use LusiTime\LusiTime;
use LusiTime\LusiIntervalItr;

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
	/* Static members of the class
     */
    private static $instrument_names = array(
        'AMO',
        'SXR',
        'XPP',
        'XCS',
        'CXI',
        'MEC'
    );
    private static $beam_destination_masks = array(
        'AMO' =>  2,
        'SXR' =>  4,
        'XPP' =>  8,
        'XCS' =>  32,
        'CXI' =>  64,
        'MEC' => 128
    );

    /* Object members
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

    /**
     * Return an array of properly ordered instrument names. The names
     * are ordered according to a physical location of instrument hatches
     * at PCDS.
     * 
     * @return type array
     */
    public static function instrument_names() { return SysMon::$instrument_names; }

    /**
     * Return a bitmask for an instruments as it's defined for
     * EPICS PV 'XRAY_DESTINATIONS'. An exception is thrown if a non-valid
     * experiment name is passed into the method for which no mask is defined.
     *
     * @param type $instr_name
     * @return type integer mask
     */
    public static function beam_destination_mask($instr_name) {
        $instr_name_trimmed = strtoupper(trim($instr_name));
        if( array_key_exists($instr_name_trimmed, SysMon::$beam_destination_masks ))
            return SysMon::$beam_destination_masks[$instr_name_trimmed];
        throw new DataPortalException (
            __METHOD__,
            "unexpected result set returned by the query" );
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
                case 'min_gap_width_sec':   $config[$param] =                  intval($value);  break;
                case 'last_run_begin_time': $config[$param] = LusiTime::from64(intval($value)); break;
                default:                    $config[$param] =                         $value;   break;
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
    public function beamtime_comment_at($begin_time, $instr) {
        $this->connect();
        $instr_escaped = $this->escape_string(trim($instr));
        $result = $this->query( "SELECT * FROM beamtime_comments WHERE gap_begin_time={$begin_time->to64()} AND instr_name='{$instr_escaped}'" );
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

    /**
     * Return an array of contigous intervals representing beam status.
     * Each interval is an associative array of three elements:
     *
     *   'begin_time' => LusiTime
     *   'end_time'   => LusiTime
     *   'status'     => a value of the specified PV (always integer number)
     *
     * @param type $pvname
     * @param type $begin_time
     * @param type $end_time 
     */
    public function beamtime_beam_status($pvname,$begin_time,$end_time) {

        $leftmost = null;
        {
            $sql = "select timestamp,value from pv,pv_val WHERE pv.id=pv_val.pv_id AND name='{$pvname}' AND timestamp IN (select MAX(timestamp) AS timestamp from pv,pv_val WHERE pv.id=pv_val.pv_id AND name='{$pvname}' AND timestamp <= {$begin_time->to64()} ) ORDER BY timestamp";
            $result = $this->query($sql);
            $nrows = mysql_numrows( $result );
            if( $nrows ) {
                if( $nrows != 1 )
                    throw new DataPortalException (
                        __METHOD__,
                        "unexpected result set returned by the query" );
                $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
                if( $attr['timestamp'] )
                    $leftmost = array(
                        'timestamp' => LusiTime::from64($attr['timestamp']),
                        'value'     => floor(           $attr['value']));
            }
        }
        $within = array();
        {
            $result = $this->query("select timestamp,value from pv,pv_val WHERE pv.id=pv_val.pv_id AND name='{$pvname}' AND ( {$begin_time->to64()} < timestamp ) AND ( timestamp < {$end_time->to64()} ) ORDER BY timestamp");
            for( $nrows = mysql_numrows( $result ), $i = 0; $i < $nrows; $i++ ) {
                $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
                array_push(
                    $within,
                    array(
                        'timestamp' => LusiTime::from64($attr['timestamp']),
                        'value'     => floor(           $attr['value'])
                    ));
            }
        }
        $value = is_null($leftmost) ? 0 : $leftmost['value'];
        $begin = $begin_time;
        $list  = array();
        foreach( $within as $atchange ) {
            $new_value = $atchange['value'];
            if( $new_value == $value ) continue;
            array_push(
                $list,
                array(
                    'begin_time' => $begin,
                    'end_time'   => $atchange['timestamp'],
                    'status'     => $value ));
            $value = $new_value;
            $begin = $atchange['timestamp'];
        }
        array_push(
            $list,
            array(
                'begin_time' => $begin,
                'end_time'   => $end_time,
                'status'     => $value ));
        return $list;
    }

    /* =================================
     *   DATABASE MODIFICATION METHODS
     * =================================
     */

    public function update_beamtime_config($config) {
        $this->connect();
        $old_config = $this->beamtime_config();
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
    public function add_beamtime_gap($begin, $end, $instr_name) {
        $instr_name_escaped = $this->escape_string(trim($instr_name));
        $this->connect();
        $this->query(
            "INSERT INTO beamtime_gaps VALUES({$begin->to64()},{$end->to64()},'{$instr_name_escaped}')"
        );
    }
    public function beamtime_set_gap_comment($gap_begin_time, $instr_name, $comment, $system, $post_time, $posted_by_uid) {
        $this->connect();
        $instr_name_escaped    = $this->escape_string(trim($instr_name));
        $posted_by_uid_escaped = $this->escape_string(trim($posted_by_uid));
        $comment_escaped       = $this->escape_string(trim($comment));
        $system_escaped        = $this->escape_string(trim($system));
        $this->query(is_null( $this->beamtime_comment_at($gap_begin_time, $instr_name)) ?
            "INSERT INTO beamtime_comments VALUES({$gap_begin_time->to64()},'{$instr_name_escaped}',{$post_time->to64()},'{$posted_by_uid_escaped}','{$comment_escaped}','{$system_escaped}')" :
            "UPDATE beamtime_comments SET post_time={$post_time->to64()}, instr_name='{$instr_name_escaped}', posted_by_uid='{$posted_by_uid_escaped}', comment='{$comment_escaped}', system='{$system_escaped}' WHERE gap_begin_time={$gap_begin_time->to64()}"
        );
    }
    public function beamtime_clear_gap_comment($gap_begin_time, $instr_name) {
        $this->connect();
        $instr_name_escaped = $this->escape_string(trim($instr_name));
        $this->query("DELETE FROM beamtime_comments WHERE gap_begin_time={$gap_begin_time->to64()} AND instr_name='{$instr_name_escaped}'");
    }

    private function intersect_candidate_gap_with_beam_status($beam_status, $instr_name, $begin_time_64, $end_time_64, $min_gap_width_64 ) {
        $gaps = array();
        array_push(
            $gaps,
            array(
                'begin'      => LusiTime::from64($begin_time_64),
                'end'        => LusiTime::from64($end_time_64),
                'instr_name' => $instr_name
            )
        );
        return $gaps;
    }
    public function populate($pvname, $min_gap_width_sec=null) {

        // ATTENTION: This will increase the default value for the maximum
        //            execution time limit from 30 seconds to 300 seconds.
        //            The later is the connection timeout for IIS and Apache.
        //            So, it makes no sense to increase it further w/o redesigning
        //            this algorithm.
        //
        set_time_limit( 300 );

        $config = $this->beamtime_config();

        // Cross-check values of optional parameters of the script versus
        // configuration parameters stored at the previous (if any) invocation
        // of the script and decide ha far back in time the script should go.
        //
        // The default starting point is before the first run taken at LCLS
        // (technically speaking - recorded in the E-Log database):
        // 
        //     '2009-09-01 00:00:00'
        //
        // And the default value of the minimum gap is 30 minutes.
        //
        $last_run_begin_time = LusiTime::parse('2009-09-01 00:00:00');
        $min_gap_width_64 = 1800 * ( 1000 * 1000 * 1000 );

        if( array_key_exists( 'last_run_begin_time', $config ) &&
            array_key_exists( 'min_gap_width_sec',   $config ) &&
          ( !isset($min_gap_width_sec) || ( $min_gap_width_sec == $config['min_gap_width_sec'] ))) {

            // This is the only scenario when we can offord optimizing
            // the current operation.
            //
            $last_run_begin_time = LusiTime::parse( $config['last_run_begin_time']->toStringDay().' 00:00:00' );
            $min_gap_width_64 = $config['min_gap_width_sec'] * ( 1000 * 1000 * 1000 );

        } else if( isset($min_gap_width_sec)) {

            $min_gap_width_64 = $min_gap_width_sec * ( 1000 * 1000 * 1000 );
        }
        $this->beamtime_clear_from($last_run_begin_time);

        // Find all instrument names and all experiments. This is just
        // an optimization step needed to prevent unneccesary database
        // operations.
        //
        $logbook = new LogBook();
        $logbook->begin();

        $instrument_names = array();
        $experiments = array();
        foreach( $logbook->regdb()->instruments() as $instrument ) {
            if( !$instrument->is_location()) {
                array_push( $instrument_names, $instrument->name());
                $experiments[$instrument->name()] = array();
                foreach( $logbook->experiments_for_instrument($instrument->name()) as $experiment ) {
                    array_push( $experiments[$instrument->name()], $experiment);
                }
            }
        }
        sort( $instrument_names );

        $itr = new LusiIntervalItr($last_run_begin_time, LusiTime::now());
        while( $itr->next_day()) {

            $start_64 = $itr->start()->to64();
            $stop_64  = $itr->stop()->to64();

            $start_minus_12hrs = $start_64 - 12 * 3600 * 1000 * 1000 * 1000;

            // Get the beam status in all hatches for the duration of the
            // current interval.
            //
            $beam_status = $this->beamtime_beam_status($pvname, $itr->start(), $itr->stop());

            // Find all runs intersecting the current day
            //
            foreach( $instrument_names as $instr_name ) {

                $runs = array();
                foreach( $experiments[$instr_name] as $experiment ) {

                    $exper_name = $experiment->name();
                    $exper_id   = $experiment->id();

                    foreach( $experiment->runs_intersecting_interval( $itr->start(), $itr->stop()) as $run ) {

                        // Ignore runs which lasted for longer than 12 hours. Those are
                        // most likelly runaway. Also ignore runs which have no ending.
                        //
                        $begin_time_64 = $run->begin_time()->to64();
                        if( $begin_time_64 < $start_minus_12hrs ) continue;

                        if( is_null($run->end_time())) continue;
                        $end_time_64 = $run->end_time()->to64();

                        // Trancate run duration to the shift boundaries. And also keep
                        // the non-trancated begin time as it's going to be used as a unique
                        // key in the index of timestamps.
                        //
                        $begin_time_nontrancated_64 = $begin_time_64;
                        if($begin_time_64 < $start_64) $begin_time_64 = $start_64;
                        if(  $end_time_64 >  $stop_64)   $end_time_64 =  $stop_64;

                        $runs[$begin_time_nontrancated_64] = array(
                            'begin_time_64' => $begin_time_64,
                            'end_time_64'   => $end_time_64,
                            'instr_name'    => $instr_name,
                            'exper_name'    => $exper_name,
                            'exper_id'      => $exper_id
                        );
                        $this->add_beamtime_run(
                            LusiTime::from64($begin_time_64),
                            LusiTime::from64($end_time_64),
                            $exper_id,
                            $run->num(),
                            $exper_name,
                            $instr_name);
                    }
                }
                usort($runs,"DataPortal\SysMon::cmp_runs_by_begin_time");

                // Find gaps between runs using the following algorithm:
                //
                // 1. find candidate gaps (intervals of no data taking between two consequtive
                // runs where the amount of time between the end of the previous run and the begin
                // time of the next one exceeds the minimum gap width
                // 
                // 2. evaluate (and truncate if needed) each such candidate against the beam
                // status intervals available in the instrument hatch. If a gap needs to be
                // truncated because of lesser availabilty of the beam then re-evaluate
                // that (truncated) gap against the previously stated condition of being 
                // longer than he minimum gap width
                // 
                // 3. accept gaps which pass both filters
                //
                $prev_end_run_64 = $start_64;
                foreach( $runs as $t => $run ) {

                    $begin_time_64 = $run['begin_time_64'];
                    $end_time_64   = $run['end_time_64'];

                    // Find a gap accross all instruments. Consider gaps which are
                    // longer than the specified width only.
                    //
                    if(( $begin_time_64 > $prev_end_run_64 ) && ( $begin_time_64 - $prev_end_run_64 > $min_gap_width_64 )) {
                        foreach(
                            SysMon::intersect_candidate_gap_with_beam_status(
                                $beam_status,
                                $instr_name,
                                $prev_end_run_64,
                                $begin_time_64,
                                $min_gap_width_64 ) as $gap ) {
                            $this->add_beamtime_gap( $gap['begin'], $gap['end'], $instr_name);
                        }
                    }
                    $prev_end_run_64 = $end_time_64;

                    // Update the global configuration parameter which will be stored
                    // in the database, and which will determinne a checkpoint from where
                    // the next invocation of teh script will run.
                    //
                    $last_run_begin_time_64 = $last_run_begin_time->to64();
                    if( $begin_time_64 > $last_run_begin_time_64 )
                        $last_run_begin_time = LusiTime::from64( $begin_time_64 );
                }

                // Generate the last gap (if any)
                //
                if(( $stop_64 > $prev_end_run_64 ) && ( $stop_64 - $prev_end_run_64 > $min_gap_width_64 )) {
                    $this->add_beamtime_gap( LusiTime::from64( $prev_end_run_64), LusiTime::from64( $stop_64 ), $instr_name);
                }
            }
        }

        // Save updated configuration parameters in the database
        //
        $config['last_run_begin_time'] = $last_run_begin_time;
        $config['min_gap_width_sec']   = $min_gap_width_64 / (1000 * 1000 * 1000);

        $this->update_beamtime_config($config);
    }

    private static function cmp_runs_by_begin_time($a, $b) {
        if($a == $b) return 0;
        return ($a < $b) ? -1 : 1;
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

    $begin_time = LusiTime::from64(1332545544195338010);
    $end_time   = LusiTime::from64(1332545888130285024);
    foreach( $sysmon->beamtime_beam_status('XRAY_DESTINATIONS',$begin_time,$end_time) as $ival ) {
        $ival_begin_time = $ival['begin_time'];
        $ival_end_time   = $ival['end_time'];
        $status          = $ival['status'];
        print "<br>{$ival_begin_time->to64()} - {$ival_end_time->to64()} : {$status}";
    }
    $sysmon->commit();

} catch ( Exception           $e ) { print $e; }
  catch ( DataPortalException $e ) { print $e->toHtml(); }
*/
?>
