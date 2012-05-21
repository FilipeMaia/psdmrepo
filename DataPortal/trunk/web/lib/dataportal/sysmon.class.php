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
    private static $beam_destinations = array(
        'FEE',
        'AMO',
        'SXR',
        'XPP',
        'XRT',
        'XCS',
        'CXI',
        'MEC'
    );
    private static $beam_destination_masks = array(
        'FEE' =>   1,
        'AMO' =>   2,
        'SXR' =>   4,
        'XPP' =>   8,
        'XRT' =>  16,
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
     * Check if the specified name is a name of an intrument rather than just
     * an X-Ray destination. Return True if so. Return False otherwise. Throw
     * an exception if the passed name doesn't correspond to any known X-Ray
     * destinations.
     *
     * @param type $name
     * @return type boolean
     */
    public static function is_instrument_name($name) {
        $name_trimmed = strtoupper(trim($name));
        if( false === array_search($name_trimmed, SysMon::$beam_destinations))
            throw new DataPortalException (
                __METHOD__,
                "illegal parameter value '{$name_trimmed}', beam destination name was expected" );
        if( false === array_search($name_trimmed, SysMon::$instrument_names)) return false;
        return true;
    }

    /**
     * Return an array of properly ordered X-Ray beam destination names.
     * This array is a subset of instruments names since it also includes
     * beam locations in between or before or after instrument hatches.
     * The names are ordered according to a physical location of beams
     * positions.
     *
     * @return type array
     */
    public static function beam_destinations() { return SysMon::$beam_destinations; }

    /**
     * Return a bitmask for an instrument or location (between instruments) as 
     * t's defined for EPICS PV 'XRAY_DESTINATIONS'. An exception is thrown if
     * a non-valid name is passed into the method for which no mask is defined.
     *
     * @param type $name
     * @return type integer mask
     */
    public static function beam_destination_mask($name) {
        $name_trimmed = strtoupper(trim($name));
        if( array_key_exists($name_trimmed, SysMon::$beam_destination_masks ))
            return SysMon::$beam_destination_masks[$name_trimmed];
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

    public function beamtime_runs( $begin_time=null, $end_time=null ) {
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

    public function beamtime_gaps( $begin_time=null, $end_time=null ) {
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
    public function beamtime_comments( $begin_time=null, $end_time=null ) {
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
    public function beamtime_systems() {
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

    private function intersect_candidate_gap_with_beam_status_unless(
        $no_beam_correction4gaps,
        $beam_status,
        $instr_name,
        $begin_time_64,
        $end_time_64,
        $min_gap_width_64 ) {

        $gaps = array();

        if( $no_beam_correction4gaps ) {
            array_push(
                    $gaps,
                    array(
                        'begin'      => LusiTime::from64($begin_time_64),
                        'end'        => LusiTime::from64($end_time_64),
                        'instr_name' => $instr_name
                    )
                );
        } else {

            // Consider only those beam time intervals where the beam time status
            // was corresponding to the requested detector.
            //
            $instr_mask = SysMon::beam_destination_mask($instr_name);

            foreach( $beam_status as $ival ) {

                // Skip beam time intervals ended before the candidate begins and
                // began after the candidate ends.
                //
                $ival_end_64 = $ival['end_time']->to64();
                if( $ival_end_64 <= $begin_time_64 ) continue;

                $ival_begin_64 = $ival['begin_time']->to64();
                if( $ival_begin_64 >= $end_time_64 ) break;

                // Skip beam time intervals which are not relevant
                // to the requested instrument.
                //
                if( $ival['status'] != $instr_mask ) continue;

                // Finally, we seem to have hit the rigth beam time interval.
                // Find its intersection with the candidate gap and evaluate the duration
                // of the intersection to see if it qualifies as the real gap (the one
                // which would have required minimum duration).
                //
                $gap_begin_64 = $ival_begin_64 < $begin_time_64 ? $begin_time_64 : $ival_begin_64;
                $gap_end_64   = $ival_end_64   > $end_time_64   ? $end_time_64   : $ival_end_64;

                if(( $gap_end_64 - $gap_begin_64 ) >= $min_gap_width_64 ) {
                    array_push(
                        $gaps,
                        array(
                            'begin'      => LusiTime::from64($gap_begin_64),
                            'end'        => LusiTime::from64($gap_end_64),
                            'instr_name' => $instr_name
                        )
                    );
                }
            }
        }
        return $gaps;
    }
    /**
     * Update/populate the database content if needed to add more runs or
     * if forced.
     *
     * @param type $pvname
     * @param type $min_gap_width_sec
     * @param type $no_beam_correction4gaps
     * @param type $force 
     */
    public function populate($pvname, $min_gap_width_sec=null, $no_beam_correction4gaps=false, $force=false) {

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
        // NOTE: No optimization in the 'force' mode - rebuild derived
        //       data from the very first run known at the database.
        //       Always give an explicitly specified gap width the highest
        //       priority. Use the gaps width value (if any exists) stored
        //       in the database if no explicit gap is provided. And assume
        //       the default one as a final resort if no gap width found
        //       in the database.
        //
        $last_run_begin_time = LusiTime::parse('2009-09-01 00:00:00');
        $min_gap_width_64 = 1800 * ( 1000 * 1000 * 1000 );

        if( $force ) {
            if( is_null($min_gap_width_sec)) {
                if( array_key_exists( 'min_gap_width_sec', $config ))
                    $min_gap_width_64 = $config['min_gap_width_sec'] * ( 1000 * 1000 * 1000 );
            } else
                $min_gap_width_64 = intval($min_gap_width_sec) * ( 1000 * 1000 * 1000 );
        } else {
            if( array_key_exists( 'last_run_begin_time', $config ) &&
                array_key_exists( 'min_gap_width_sec',   $config ) &&
               ( is_null($min_gap_width_sec) || ( $min_gap_width_sec == $config['min_gap_width_sec'] ))) {

                // This is the only scenario when we can afford optimizing
                // the current operation by limiting a scope of runs by those
                // taken since the last run recorded as a configuration parameter.
                //
                $last_run_begin_time = LusiTime::parse( $config['last_run_begin_time']->toStringDay().' 00:00:00' );
                $min_gap_width_64 = $config['min_gap_width_sec'] * ( 1000 * 1000 * 1000 );

            } else if( !is_null($min_gap_width_sec)) {

                $min_gap_width_64 = $min_gap_width_sec * ( 1000 * 1000 * 1000 );
            }
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
                            SysMon::intersect_candidate_gap_with_beam_status_unless(
                                $no_beam_correction4gaps,
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
                    foreach(
                        SysMon::intersect_candidate_gap_with_beam_status_unless(
                            $no_beam_correction4gaps,
                            $beam_status,
                            $instr_name,
                            $prev_end_run_64,
                            $stop_64,
                            $min_gap_width_64 ) as $gap ) {
                        $this->add_beamtime_gap( $gap['begin'], $gap['end'], $instr_name);
                    }
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

    /* =================================================
     *   METHODS FOR MANAGING NOTIFICATION SUBSCRIPTIONS
     * =================================================
     */

    /* Subscribe the current user for e-mail notifications on downtime explanations
     * if the flag value is set to TRUE. Unsubscribe otherwise.
     */
    public function subscribe4explanations_if ( $subscribe, $subscriber, $address ) {

    	$this->connect();

    	$authdb = AuthDB::instance();
    	$authdb->begin();

    	$subscriber_str = $this->escape_string( trim( $subscriber ));
    	$address_str    = $this->escape_string( trim( $address ));
    	$by             = $this->escape_string( trim( $authdb->authName()));
    	$now            = LusiTime::now();
    	$host_str       = $this->escape_string( trim( $authdb->authRemoteAddr()));

    	$this->query(
			$subscribe ?
    		"INSERT INTO beamtime_subscriber VALUES (NULL,'{$subscriber_str}','{$address_str}','{$by}',{$now->to64()},'{$host_str}')" :
    		"DELETE FROM beamtime_subscriber WHERE subscriber='{$subscriber_str}' AND address='{$address_str}'"
		);

		$url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['SERVER_NAME'].'/apps-dev/portal/experiment_time';

        if( $subscribe )                
        	$this->do_notify(
                'LCLS Data Taking Monitor',
        		$address,
        		"*** SUBSCRIBED ***",
				<<<HERE
                             ** ATTENTION **

The message was sent by the automated notification system because this e-mail
has been just registered to recieve alerts on downtime explanations posted
for long gaps between PCDS DAQ runs.

The registration has been requested by:

  '{$authdb->authName()}' @ {$authdb->authRemoteAddr()} [ {$now->toStringShort()} ]

To unsubscribe from this service, please use the LCLS Data Taking Time Monitor app:

  {$url}

HERE
       		);
        else
        	$this->do_notify(
                'LCLS Data Taking Monitor',
        		$address,
        		"*** UNSUBSCRIBED ***",
				<<<HERE
                             ** ATTENTION **

The message was sent by the automated notification system because this e-mail
has been just unregistered from recieving alerts on downtime explanations posted
for long gaps between PCDS DAQ runs.

The change has been requested by:

  '{$authdb->authName()}' @ {$authdb->authRemoteAddr()} [ {$now->toStringShort()} ]

To subscribe back to this service, please use the LCLS Data Taking Time Monitor app:

  {$url}

HERE
			);
    }

    /* Check if the current user is subscribed for e-mail notifications on downtime
     * explanations, and if so return an array with the details of the subscription.
     * Return null otherwise.
     *
     *   Key             | Type            | Description
     *   ----------------+-----------------+------------------------------------------------------------------------------
     *   id              | unsigned number | unique identifier of the record in the database
     *   subscriber      | string          | user account of a person subscribed
     *   address         | string          | e-mail address of the subscriber
     *   subscribed_by   | LusiTime        | user account of a person who requested the subscription 
     *   subscribed_time | LusiTime        | time when the subscription was made 
     *   subscribed_host | string          | host (IP address or DNS name) name from which the operation was requested
     *
     */
    public function check_if_subscribed4explanations ( $subscriber, $address ) {

        $this->connect();

    	$subscriber_str = $this->escape_string( trim( $subscriber ));
    	$address_str    = $this->escape_string( trim( $address ));
    	$result = $this->query(
    		"SELECT * FROM beamtime_subscriber WHERE subscriber='{$subscriber_str}' AND address='{$address_str}'"
    	);
    	$nrows = mysql_numrows( $result );
    	if( !$nrows ) return null;
    	if( $nrows != 1 )
			throw new DataPortalException (
				__METHOD__,
				"duplicate entries for downtime explanations subscriber: {$subscriber} ({$address}) in database. Database can be corrupted." );
		$row = mysql_fetch_array( $result, MYSQL_ASSOC );
       	$row['subscribed_time'] = LusiTime::from64($row['subscribed_time']);
       	return $row;
    }

    /* Get all known subscribers for downtime explanation notifications.
     * 
     * The method will return an array (a list) of entries similar to
     * the ones reported by the previous method.
     */
    public function get_all_subscribed4explanations () {
	    $list = array();
   		$this->connect();
    	$result = $this->query( "SELECT * FROM beamtime_subscriber" );
    	$nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
        	$row = mysql_fetch_array( $result, MYSQL_ASSOC );
        	$row['subscribed_time'] = LusiTime::from64($row['subscribed_time']);
        	array_push ( $list, $row );
        }
        return $list;
    }

    public function notify_allsubscribed4explanations ($instr_name, $gap_begin_time) {
        $this->connect();
		$url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['SERVER_NAME'].'/apps-dev/portal/experiment_time';
        foreach( $this->get_all_subscribed4explanations() as $subscriber ) {
            $address = $subscriber['address'];
            $this->do_notify(
                'LCLS Data Taking Monitor',
        		$address,
        		"*** DOWNTIME EXPLANATION POSTED *** [ {$instr_name} ] {$gap_begin_time->toStringShort()}",
				<<<HERE
                             ** ATTENTION **

The message was sent by the automated notification system because this e-mail
was found registered to recieve alerts on downtime explanations posted
for long gaps between PCDS DAQ runs.

To subscribe back to this service, please use the LCLS Data Taking Time Monitor app:

  {$url}

HERE
            );
        }
    }
    public function do_notify( $application, $address, $subject, $body ) {
        $tmpfname = tempnam("/tmp", "webportal");
        $handle = fopen( $tmpfname, "w" );
        fwrite( $handle, $body );
        fclose( $handle );

        shell_exec( "cat {$tmpfname} | mail -s '{$subject}' {$address} -- -F '{$application}'" );

        // Delete the file only after piping its contents to the mailer command.
        // Otherwise its contents will be lost before we use it.
        //
        unlink( $tmpfname );
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
