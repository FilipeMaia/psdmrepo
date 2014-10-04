<?php

/*
 * The script will report th etotal amount of data (measured in GB)
 * residing in the SHORT-TERM storage in the specified scope. The scope
 * is deremined by a presense of the 'experiment_name' parameter. If a non-empty
 * value of the parameter is provided the scrpt will return statistics for that
 * experiment. Otherwise the instrument-wide statistics is harvested and returned.
 */
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use DataPortal\Config;
use FileMgr\FileMgrIrodsDb;
use LogBook\LogBook;
use LusiTime\LusiTime;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error ($msg) {
    $result = array(
        'status' => 'error',
        'message' => ''.$msg
    );
    print json_encode($result);
    exit;
}
function report_result ($result) {
    $result['status'] = 'success';
    print json_encode($result);
    exit;
}

if (!isset($_GET['instr_name'])) report_error('no instrument name found among parameters of the script');
$instr_name = trim($_GET['instr_name']);

$exper_name = isset($_GET['exper_name']) ? trim($_GET['exper_name']) : null;
if ($exper_name === '') $exper_name = null;

define( 'BYTES_IN_GB', 1024 * 1024 * 1024 );
define( 'BYTES_IN_TB', 1024 * BYTES_IN_GB );

function expiration_time ($ctime, $retention, $deadline_time=null) {
    $ctime_time = LusiTime::parse($ctime);
    if (is_null($ctime_time)) return '';
    $expiration_time = new LusiTime($ctime_time->sec + 31 * 24 * 3600 * intval($retention));
    if (is_null($expiration_time)) return '';
    $expiration_time_str = $expiration_time->toStringDay();
    if ($deadline_time && $expiration_time->less($deadline_time))
        $expiration_time_str = '<span style="color:red;">'.$expiration_time_str.'</span>';
    return $expiration_time_str;
}

function is_expired ($ctime_time, $retention_months, $deadline_time) {
    $expiration_time = new LusiTime($ctime_time->sec + 31 * 24 * 3600 * $retention_months);
    return !is_null($expiration_time) && $expiration_time->less($deadline_time);
}

try {
    $result = array();

    Config::instance()->begin();
    LogBook::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $default_short_ctime_time_str  =        Config::instance()->get_policy_param('SHORT-TERM',  'CTIME');
    $default_short_retention       = intval(Config::instance()->get_policy_param('SHORT-TERM',  'RETENTION'));
    $default_medium_ctime_time_str =        Config::instance()->get_policy_param('MEDIUM-TERM', 'CTIME');
    $default_medium_retention      = intval(Config::instance()->get_policy_param('MEDIUM-TERM', 'RETENTION'));
    $default_medium_quota          = intval(Config::instance()->get_policy_param('MEDIUM-TERM', 'QUOTA'));

    $now = LusiTime::now();
    $year = $now->year();
    $month = $now->month();

//    $month = $now->month() + 1;
//    if ($month > 12) {
//        $year++;
//        $month = $month - 12;
//    }
    $deadline_time = LusiTime::parse(sprintf("%04d-%02d-02", $year, $month));

    if (is_null($exper_name)) {

        $short_term_files    = 0;
        $short_term_size_tb  = 0;

        $medium_term_files   = 0;
        $medium_term_size_tb = 0;

        $medium_quota_tb     = 0;

        $short_expired_tb = array();

        $expiration_deadlines = array();
        foreach (
            array(
                '2014-06-02',
                '2014-07-02',
                '2014-08-02',
                '2014-09-02',
                '2014-10-02',
                '2014-11-02',
                '2014-12-02') as $day) {
            array_push(
                $expiration_deadlines,
                array(
                    'day'         => $day,
                    'time_object' => LusiTime::parse($day)
                )
            );
            $short_expired_tb[$day] = 0.0;
        }
        foreach (LogBook::instance()->experiments_for_instrument ($instr_name) as $experiment) {
            if ($experiment->is_facility()) continue;

            $all_files = array();
            foreach (array('xtc','hdf5') as $type) {
                foreach (FileMgrIrodsDb::instance()->runs ($instr_name, $experiment->name(), $type) as $r) {
                    foreach ($r->files as $f) {
                        $short_term_files++;
                        if ($f->resource == 'lustre-resc') {
                            $short_term_size_tb += $f->size / BYTES_IN_TB;
                            $all_files[$f->path] = $f;
                        }
                    }
                }
            }
            foreach (Config::instance()->medium_store_files($experiment->id()) as $f) {
                $medium_term_files   += 1;
                $medium_term_size_tb += $f['irods_size'] / BYTES_IN_TB;
                
                // Remove this file from all files as it belongs to the MEDIUM-TERM storage
                //
                unset( $all_files[$f['irods_filepath']]);
            }
            $quota_gb = $default_medium_quota;
            $medium_quota = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA' );
            if (!is_null( $medium_quota )) {
                $val = intval($medium_quota->value());
                if ($val > 0) $quota_gb = $val;
            }
            $medium_quota_tb += $quota_gb / 1024;

            /*
             * TODO: REimplement this algorithm to use CTIME of each file as well, not
             * just default and experiment-specific overrides. Otherwise the algorithm
             * will produce skewed statistics for future experiments.
             */
            $short_quota_ctime           = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-CTIME' );
            $short_quota_ctime_time_str  = is_null( $short_quota_ctime ) ? '' : $short_quota_ctime->value();
            $short_quota_ctime_time      = LusiTime::parse($short_quota_ctime_time_str ? $short_quota_ctime_time_str : $default_short_ctime_time_str);

            $short_retention             = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-RETENTION' );
            $short_retention_months      = is_null( $short_retention ) ? 0 : intval($short_retention->value());
            $short_retention_months      = $short_retention_months ? $short_retention_months : $default_short_retention;
            
            foreach ($all_files as $irods_filepath => $f) {
                $ctime_time = new LusiTime(intval($f->ctime));
                if (!is_null($short_quota_ctime_time) && $ctime_time->less($short_quota_ctime_time)) {
                    $ctime_time = $short_quota_ctime_time;
                }
                foreach ($expiration_deadlines as $deadline)
                    if (is_expired (
                        $ctime_time,
                        $short_retention_months,
                        $deadline['time_object']))
                        $short_expired_tb[$deadline['day']] += $f->size / BYTES_IN_TB;
            }
        }
        foreach ($short_expired_tb as $day => $tb)
            $short_expired_tb[$day] = intval($tb);

        $result['short_term_files'    ] =        $short_term_files    -        $medium_term_files;
        $result['short_term_size_tb'  ] = intval($short_term_size_tb) - intval($medium_term_size_tb);

        $result['medium_term_files'   ] =        $medium_term_files;
        $result['medium_term_size_tb' ] = intval($medium_term_size_tb);
        $result['medium_term_quota_tb'] = intval($medium_quota_tb);

        $result['short_expired_tb'] = $short_expired_tb;


    } else {

        $experiment = LogBook::instance()->find_experiment_by_name($exper_name);
        if (is_null($experiment) || ($experiment->instrument()->name() !== $instr_name))
            report_error("no experiment '{$exper_name}' found at instrument '{$instr_name}'");
            
        $num_runs = intval($experiment->num_runs());
 
        $size_gb = 0;
 
        foreach (array('xtc','hdf5') as $type)
            foreach (FileMgrIrodsDb::instance()->runs ($instr_name, $exper_name, $type) as $r)
                foreach ($r->files as $f)
                    if ($f->resource == 'lustre-resc') $size_gb += $f->size / BYTES_IN_GB;

        $short_quota_ctime           = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-CTIME' );
        $short_quota_ctime_time_str  = is_null( $short_quota_ctime ) ? '' : $short_quota_ctime->value();

        $short_retention             = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-RETENTION' );
        $short_retention_months      = is_null( $short_retention ) ? 0 : intval($short_retention->value());
        $short_retention_str         = $short_retention_months ? $short_retention_months : '';
 
        $short_expiration_str = $num_runs ? expiration_time (
            $short_quota_ctime_time_str ? $short_quota_ctime_time_str : $default_short_ctime_time_str,
            $short_retention_str        ? $short_retention_str        : $default_short_retention,
            $deadline_time) : 
            '';

        $medium_quota_ctime          = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA-CTIME' );
        $medium_quota_ctime_time_str = is_null( $medium_quota_ctime ) ? null : $medium_quota_ctime->value();

        $medium_retention            = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA-RETENTION' );
        $medium_retention_months     = is_null( $medium_retention ) ? 0 : intval($medium_retention->value());
        $medium_retention_str        = $medium_retention_months ? $medium_retention_months : '';

        $medium_expiration_str = $num_runs ?  expiration_time (
            $medium_quota_ctime_time_str ? $medium_quota_ctime_time_str : $default_medium_ctime_time_str,
            $medium_retention_str        ? $medium_retention_str        : $default_medium_retention,
                null) : 
            '';

        $medium_usage_files = 0;
        $medium_usage_gb    = 0;
        foreach( Config::instance()->medium_store_files($experiment->id()) as $file ) {
            $medium_usage_files += 1;
            $medium_usage_gb += $file['irods_size'] / BYTES_IN_GB;
        }

        $result['num_runs']               = $num_runs;
        $result['short_term_size_gb']     = intval($size_gb) - intval($medium_usage_gb);
        $result['short_term_expiration']  = $short_expiration_str;
        $result['medium_term_expiration'] = $medium_expiration_str;
        $result['medium_usage_files']     = $medium_usage_files;
        $result['medium_usage_gb']        = intval($medium_usage_gb);
    }
    FileMgrIrodsDb::instance()->commit();
    LogBook::instance()->commit();
    Config::instance()->commit();

    report_result ($result);

} catch (Exception $e) { report_error($e); }

?>
