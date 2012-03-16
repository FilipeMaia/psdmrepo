<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\SysMon;
use DataPortal\DataPortalException;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

/**
 * This script will populate the system monitoring database with data
 * representing LCLS beam time usage for all known runs.
 *
 */
function report_error($msg) {
	echo $msg;
    exit;
}

function cmp_runs_by_begin_time($a, $b) {
    if($a == $b) return 0;
    return ($a < $b) ? -1 : 1;
}

/* Parse optional parameter to the script:
 *
 *   min_gap_width_sec
 * 
 *     - a positive integer value is expected. The value will be compared
 *       with what's stored in the database, and if the database has a different
 *       value then the full history will be regenerated. Note, that this will
 *       also cover a case when nothing was stored for that parameter in the database. 
 */
if( isset($_GET['min_gap_width_sec'])) {
    $min_gap_width_sec = intval(trim($_GET['min_gap_width_sec']));
    if(!$min_gap_width_sec)
        report_error('please, provide a positive number for parameter: min_gap_width_sec');
}
try {
    $authdb = AuthDB::instance();
	$authdb->begin();

    if( !$authdb->hasRole($authdb->authName(),null,'BeamTimeMonitor','Editor')) report_error('not authorized to use this service');

    $logbook = new LogBook();
	$logbook->begin();

	$regdb = new RegDB();
	$regdb->begin();

	$sysmon = SysMon::instance();
	$sysmon->begin();

    // Load configuration parameters stored at the last invocation
    // of the script. The parameters are going to drive how far this script
    // should go back in history. The value of the parameters can also be adjusted
    // 
    //
    $config = $sysmon->beamtime_config();
    print <<<HERE
<h3>Configuration loaded from the database:</h3>
<div style="padding-left:10;">
  <table><thead>
HERE;
    foreach($config as $param => $value) {
        print <<<HERE
    <tr>
      <td align=left><b>{$param}</b></td>
      <td> : {$value}</td>
    </tr>
HERE;
    }
    print <<<HERE
  </thead><table>
</div>
HERE;

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
    $sysmon->beamtime_clear_from($last_run_begin_time);

    // Find all instrument names and all experiments. This is just
    // an optimization step needed to prevent unneccesary database
    // operations.
    //
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

    class IntervalItr {
        private $from  = null;
        private $to    = null;
        private $start = null;
        private $stop  = null;
        public function __construct($from,$to) {
            $this->from = $from;
            $this->to   = $to;
        }
        public function next_day() {
            // Next interval begins either from the 'from' limit passed to the constructor,
            // or from the previously calculated 'stop' time.
            //
            $this->start = is_null($this->start) ? $this->from : $this->stop;
            if( $this->start->to64() >= $this->to->to64()) return false;

            // And it ends either in +24 hours after the begin time of the interval,
            // or at the 'to' time passed to the constructor, whichever comes first.
            //
            $start_midnight = LusiTime::parse($this->start->toStringDay().' 00:00:00');
            $this->stop = new LusiTime($start_midnight->sec + 24 * 3600,  $start_midnight->nsec);
            //
            // Note that we also need to catch the Day Time Saving shift in +/-1 hr. In
            // that scenario adding exactly 24 hours would bring us either 1 hr short
            // or 1 hr beyond the desired day.
            //
            $stop_midnight = LusiTime::parse($this->stop->toStringDay().' 00:00:00');
            if($stop_midnight->to64() == $start_midnight->to64()) {

                // We're 1 hr short. Let's correct this by adding 25 hours.
                //
                $this->stop = new LusiTime($start_midnight->sec + 25 * 3600,  $start_midnight->nsec);

            } else {

                // this will automatically correct for 1 hr beyon the midnight
                // (if) added in an opposite direction.
                //
                $this->stop = $stop_midnight;
            }
            if( $this->to->to64() < $this->stop->to64()) $this->stop = $this->to;

            return true;
        }
        public function start() { return $this->start; }
        public function stop () { return $this->stop; }
    }
 
    $itr = new IntervalItr($last_run_begin_time, LusiTime::now());

    while( $itr->next_day()) {

        $start_64 = $itr->start()->to64();
        $stop_64  = $itr->stop()->to64();

        print "<br>day loop: {$start_64} [{$itr->start()->toStringDay()}] - {$stop_64} [{$itr->stop()->toStringDay()}]";

        $start_minus_12hrs = $start_64 - 12 * 3600 * 1000 * 1000 * 1000;

        // Find all runs intersecting the current day
        //
        $runs = array();
        foreach( $instrument_names as $instr_name ) {

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
                    $sysmon->add_beamtime_run(
                        LusiTime::from64($begin_time_64),
                        LusiTime::from64($end_time_64),
                        $exper_id,
                        $run->num(),
                        $exper_name,
                        $instr_name);
                }
            }
        }
        usort($runs,"cmp_runs_by_begin_time");

        // Find gaps between runs
        //
        $gaps = array();
        $prev_end_run_64 = $start_64;
        foreach( $runs as $t => $run ) {

            $begin_time_64 = $run['begin_time_64'];
            $end_time_64   = $run['end_time_64'];

            // Find a gap accross all instruments. Consider gaps which are
            // longer than the specified width only.
            //
            if(( $begin_time_64 > $prev_end_run_64 ) && ( $begin_time_64 - $prev_end_run_64 > $min_gap_width_64 )) {
                $sysmon->add_beamtime_gap( LusiTime::from64( $prev_end_run_64), LusiTime::from64( $begin_time_64 ));
            }
            $prev_end_run_64 = $end_time_64;

            // Update the global configuration parameter which will be stored
            // in the database, and which will determinne a checkpoint from where
            // the next invocation of teh script will run.
            //
            $last_run_begin_time = LusiTime::from64( $begin_time_64 );
        }

        // Generate the last gap (if any)
        //
        if(( $stop_64 > $prev_end_run_64 ) && ( $stop_64 - $prev_end_run_64 > $min_gap_width_64 )) {
            $sysmon->add_beamtime_gap( LusiTime::from64( $prev_end_run_64), LusiTime::from64( $stop_64 ));
        }
    }

    // Save updated configuration parameters in the database
    //
    $config['last_run_begin_time'] = $last_run_begin_time;
    $config['min_gap_width_sec']   = $min_gap_width_64 / (1000 * 1000 * 1000);

    $sysmon->update_beamtime_config($config);

	$authdb->commit();
    $logbook->commit();
    $regdb->commit();
	$sysmon->commit();

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( DataPortalException $e ) { report_error( $e->toHtml()); }
  catch( LogBookException    $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( RegDBException      $e ) { report_error( $e->toHtml()); }
  catch( Exception           $e ) { report_error( "{$e}" );      }
  
?>
