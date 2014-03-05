<?php

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

require_once( 'authdb/authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use FileMgr\FileMgrIrodsDb;
use LusiTime\LusiTime;
use RegDB\RegDB;

define( 'BYTES_IN_MB', 1000 * 1000 );
define( 'BYTES_IN_GB', 1000 * BYTES_IN_MB );
define( 'BYTES_IN_TB', 1000 * BYTES_IN_GB );

function report_error_end_exit ($msg) {
    print "<h2 style=\"color:red;\">Error: {$msg}</h2>";
    exit;
}

if (!isset($_GET['exper_id'])) report_error_end_exit ('Please, provide an experiment identifier!') ;
$exper_id = intval($_GET['exper_id']) ;

if (!isset($_GET['first_run'])) report_error_end_exit ('Please, provide the first run!') ;
$first_run = intval($_GET['first_run']) ;

if (!isset($_GET['last_run'])) report_error_end_exit ('Please, provide the last run!') ;
$last_run = intval($_GET['last_run']) ;


try {

    AuthDB::instance()->begin();
    RegDB::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id($exper_id) ;
    if (!$experiment) report_error_end_exit('No such experiment exists') ;

    $file_type = 'XTC';

    $files_daq2offline = array();
    foreach($experiment->data_migration_files() as $file) {
        if(strtoupper($file->type()) != $file_type) continue;
        $files_daq2offline[$file->name()] = $file;
    }
    $files_offline2nersc = array();
    foreach($experiment->data_migration2nersc_files() as $file) {
        if(strtoupper($file->type()) != $file_type) continue;
        $files_offline2nersc[$file->name()] = $file;
    }



    $files = array();
    
    foreach( $experiment->files( null, true ) as $file ) {

        $runnum = $file->run();
        if ($runnum < $first_run) continue;
        if ($runnum > $last_run) continue;
        
        $file_name = sprintf("e%d-r%04d-s%02d-c%02d.xtc",
            $experiment->id(),
            $runnum,
            $file->stream(),
            $file->chunk());

        $daq_open_time = $file->open_time();

        $size_bytes = 0;

        foreach (FileMgrIrodsDb::instance()->find_file ($experiment->instrument()->name(),$experiment->name(), $file_type, $file_name) as $r) {
            if ($r->run == $runnum) {
                foreach ($r->files as $f) {
                    $size_bytes = $f->size;
                    break;
                }
            }
        }
        if (array_key_exists($file_name, $files_daq2offline)) {
            array_push (
                $files,
                array (
                    'time'    => $daq_open_time->toStringShort(),
                    'latency' => $files_daq2offline  [$file_name]->stop_time()->sec - $daq_open_time->sec,
                    'stage'   => 'daq2offline'
                )
            );
        }
        if (array_key_exists($file_name, $files_offline2nersc)) {
            array_push (
                $files,
                array (
                    'time'    => $daq_open_time->toStringShort(),
                    'latency' => $files_offline2nersc[$file_name]->stop_time()->sec - $daq_open_time->sec,
                    'stage'   => 'offline2nersc'
                )
           );
        }
    }
    print json_encode($files);

    AuthDB::instance()->commit();
    RegDB::instance()->commit();
    FileMgrIrodsDb::instance()->commit();

} catch( Exception $e ) { report_error_end_exit($e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'); }

?>