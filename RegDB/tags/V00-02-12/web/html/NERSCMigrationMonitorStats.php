<!DOCTYPE html">
<html>
<head>

<title>Monitor File Migration to NERSC (Statistics)</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>

<style type="text/css">

div#header {
  margin-bottom: 20px;        
}

.document_title,
.document_subtitle {
  font-family: "Times", serif;
  font-size: 32px;
  font-weight: bold;
  text-align: left;
}
.document_subtitle {
  color: #0071bc;
}
a, a.link {
  text-decoration: none;
  font-weight: bold;
  color: #0071bc;
}
a:hover, a.link:hover {
  color: red;
}

td.table_hdr {
  background-color:#d0d0d0;
  padding: 2px 8px 2px 8px;
  border: solid 1px #000000;
  border-top: none;
  border-left: none;
  font-family: Arial, sans-serif;
  font-weight: bold;
  font-size: 12px;
  white-space: nowrap;
}
td.table_cell {
  border:solid 1px #d0d0d0;
  border-top: none;
  border-left: none;
  padding: 2px 8px 2px 8px;
  font-family: Arial, sans-serif;
  font-size: 12px;
  white-space: nowrap;
}
tr.table_active_row:hover {
  cursor: pointer;
}
td.table_cell_left {
  font-weight: bold;
}
td.table_cell_right {
  border-right: none;
}
td.table_cell_bottom {
  border-bottom: none;
}
td.table_cell_within_group {
  border-bottom: none;
}

td.table_cell_highlight {
    background-color: rgba(200,200,200,0.5);
}

</style>


<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

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

?>


</head>

<body>
  <div style="padding:10px; padding-left:20px;">

<?php

try {

    AuthDB::instance()->begin();
    RegDB::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id($exper_id) ;
    if (!$experiment) report_error_end_exit('NO such experiment exists') ;

    print <<<HERE
<div id="header" >
  <span class="document_title">File Migration to NERSC Statistics:&nbsp;</span>
  <span class="document_subtitle">
    <a class="link" title="Open a new tab to the Web Portal of the experiment" href="../portal/?exper_id={$exper_id}" target="_blank" >
      {$experiment->instrument()->name()} / {$experiment->name()}
    </a>
  </span>
</div>

HERE;

    $shifts = array (
        1 => array (
            'end_time'              => LusiTime::parse('2013-03-01 09:00:00'),
            'to_nersc_in_shift'     => 0,
            'to_nersc_after_shift'  => 0
        ),
        2 => array (
            'end_time'              => LusiTime::parse('2013-03-05 09:00:00'),
            'to_nersc_in_shift'     => 0,
            'to_nersc_after_shift'  => 0
        ),
        3 => array (
            'end_time'              => LusiTime::parse('2013-03-05 09:00:00'),
            'to_nersc_in_shift'     => 0,
            'to_nersc_after_shift'  => 0
        ),
        4 => array (
            'end_time'              => LusiTime::parse('2013-03-05 09:00:00'),
            'to_nersc_in_shift'     => 0,
            'to_nersc_after_shift'  => 0
        ),
        5 => array (
            'end_time'              => LusiTime::parse('2013-03-05 09:00:00'),
            'to_nersc_in_shift'     => 0,
            'to_nersc_after_shift'  => 0
        )
    );

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


    foreach( $experiment->files( null, true ) as $file ) {

        $runnum = $file->run();

        $file_name = sprintf("e%d-r%04d-s%02d-c%02d.xtc",
            $experiment->id(),
            $runnum,
            $file->stream(),
            $file->chunk());

        $daq_open_time = $file->open_time();

        $daq2offline_start_time  = $files_daq2offline[$file_name]->start_time();
        $daq2offline_finish_time = $files_daq2offline[$file_name]->stop_time();

        $offline2nersc_start_time  = $files_offline2nersc[$file_name]->start_time();
        $offline2nersc_finish_time = $files_offline2nersc[$file_name]->stop_time();

        $size_bytes = 0;

        foreach (FileMgrIrodsDb::instance()->find_file ($experiment->instrument()->name(),$experiment->name(), $file_type, $file_name) as $r) {
            if ($r->run == $runnum) {
                foreach ($r->files as $f) {
                    $size_bytes = $f->size;
                    break;
                }
            }
        }
        $size_tb = $size_bytes / BYTES_IN_TB;
        
        $shift = 1;
        if     ($runnum <=  46) { $shift = 1; }
        elseif ($runnum <=  89) { $shift = 2; }
        elseif ($runnum <= 122) { $shift = 3; }
        elseif ($runnum <= 190) { $shift = 4; }
        else                    { $shift = 5; }
        
        if ($shifts[$shift]['end_time']->less($offline2nersc_finish_time)) {
            $shifts[$shift]['to_nersc_after_shift'] += $size_tb;
        } else {
            $shifts[$shift]['to_nersc_in_shift'] += $size_tb;
        }
    }

    for ($shift = 1; $shift <= 5; $shift++) {
        print '<br>Shift: '.$shift.'  In-shift: '.$shifts[$shift]['to_nersc_in_shift'].'  Delayed: '.$shifts[$shift]['to_nersc_after_shift'];
    }
    AuthDB::instance()->commit();
    RegDB::instance()->commit();
    FileMgrIrodsDb::instance()->commit();

} catch( Exception $e ) { report_error_end_exit($e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'); }

?>
    </tbody></table>
  </div>
</body>
</html>