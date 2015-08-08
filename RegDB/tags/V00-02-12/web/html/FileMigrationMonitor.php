<!DOCTYPE html">
<html>
<head>

<title>Monitor File Migration (DDD, FFB, ANA)</title>
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

input {
  padding-left: 2px;
  padding-right: 2px;
}

.highlighted {
  font-weight:bold;
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

define( 'BYTES_IN_MB', 1024 * 1024 );
define( 'BYTES_IN_GB', 1024 * BYTES_IN_MB );

function report_error_end_exit ($msg) {
    print "<h2 style=\"color:red;\">Error: {$msg}</h2>";
    exit;
}

if (!isset($_GET['exper_id'])) report_error_end_exit ('Please, provide an experiment identifier!') ;
$exper_id = intval($_GET['exper_id']) ;

?>

<script type="text/javascript">

var exper_id = <?php echo $exper_id; ?>;

var prev_runnum = 0;

$(function () {
    $('tr.table_active_row')
        .click(function () {
            var runnum = this.id ;
            var url = '../portal?exper_id='+exper_id+'&app=elog:search&params=run:'+runnum;
            window.open(url);
        })
        .mouseenter(function () {
            var runnum = this.id;
            if (prev_runnum == runnum) return;
            var prev_elem = $('tr.table_active_row.run_'+prev_runnum);
            prev_elem.css('backgroundColor','');
            prev_elem.removeAttr('title');
            var elem = $('tr.table_active_row.run_'+runnum);
            elem.css('backgroundColor','#b9dcf5');
            elem.attr('title','Click to see this run in the Web Portal of the Experiment');
            prev_runnum = runnum;
        });
});

</script>

</head>

<body>
  <div style="padding:10px; padding-left:20px;">

<?php

function format_seconds ($total_seconds) {
    $days    = 0;
    $hours   = 0;
    $minutes = 0;
    $seconds = 0;
    if ($total_seconds < 60) {
        $seconds = $total_seconds;
    } elseif ($total_seconds < 3600) {
        $minutes = intval($total_seconds / 60);
        $seconds = intval($total_seconds % 60);
    } elseif ($total_seconds < 24 * 3600) {
        $hours   = intval( $total_seconds / 3600);
        $minutes = intval(($total_seconds % 3600) / 60);
    } else {
        $days    = intval( $total_seconds / (24 * 3600));
        $hours   = intval(($total_seconds % (24 * 3600)) / 3600);
    }
    $hms = '&nbsp';
    if      ($hours)   $hms = sprintf("<b>%2d</b>h&nbsp;<b>%02d</b>m&nbsp;<b>%02d</b>s",             $hours, $minutes, $seconds);
    else if ($minutes) $hms = sprintf("&nbsp;&nbsp;&nbsp;&nbsp;<b>%2d</b>m&nbsp;<b>%02d</b>s",       $minutes, $seconds);
    else if ($seconds) $hms = sprintf("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>%2d</b>s", $seconds);
    return ($days ? sprintf("<b>%3d</b>d", $days).'&nbsp;&nbsp;' : '').$hms;
}
try {

    AuthDB::instance()->begin();
    RegDB::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id($exper_id) ;
    if (!$experiment) report_error_end_exit('NO such experiment exists') ;

    print <<<HERE
<div id="header" >
  <span class="document_title">File Migration:&nbsp;</span>
  <span class="document_subtitle">
    <a class="link" title="Open a new tab to the Web Portal of the experiment" href="../portal/?exper_id={$exper_id}" target="_blank" >
      {$experiment->instrument()->name()} / {$experiment->name()}
    </a>
  </span>
</div>

HERE;

    $now = LusiTime::now();

    $file_type = 'XTC';

    $files_daq2ffb = array();
    foreach($experiment->data_migration_files() as $file) {
        if(strtoupper($file->type()) != $file_type) continue;
        $files_daq2ffb[$file->name()] = $file;
    }
    $files_ffb2offline = array();
    foreach($experiment->data_migration2ana_files() as $file) {
        if(strtoupper($file->type()) != $file_type) continue;
        $files_ffb2offline[$file->name()] = $file;
    }
?>

    <table><tbody>
      <tr>
        <td class="table_hdr" rowspan="2" align="right"  >Run</td>
        <td class="table_hdr" rowspan="2" align="center" >File</td>
        <td class="table_hdr" rowspan="2" align="center" >Created (DAQ)</td>
        <td class="table_hdr" rowspan="2" align="center" >Size</td>
        <td class="table_hdr" colspan="4" align="center" >DAQ &Rarr; FFB</td>
        <td class="table_hdr" colspan="4" align="center" >FFB &Rarr; OFFLINE</td>
        <td class="table_hdr" rowspan="2"                >Status</td>
      </tr>
      <tr>
        <td class="table_hdr" align="center" >Started</td>
        <td class="table_hdr" align="right"  >Xfer &Delta;t</td>
        <td class="table_hdr" align="right"  >Size / &Delta;t</td>
        <td class="table_hdr" align="right"  >Latency</td>
        <td class="table_hdr" align="center" >Started</td>
        <td class="table_hdr" align="right"  >Xfer &Delta;t</td>
        <td class="table_hdr" align="right"  >Size / &Delta;t</td>
        <td class="table_hdr" align="right"  >Latency</td>
      </tr>

<?php

    $prev_runnum    = null;
    $prev_html_1st  = null;
    $prev_html_rest = null;
    $prev_num_files = null;

    foreach( $experiment->files( null, true ) as $file ) {

        $runnum = $file->run();
        if (is_null($prev_runnum)) {
            $prev_runnum    = $runnum;
            $prev_html_rest = '';
            $prev_num_files = 0;
        } else {
            if ($prev_runnum == $runnum) {
                $prev_num_files++;
            } else {
                $rows = $prev_num_files + 1;
                print <<<HERE
      <tr class="table_active_row run_{$prev_runnum}" id="{$prev_runnum}" >
        <td class="table_cell " valign="top" align="right" rowspan="{$rows}" >{$prev_runnum}</td>
{$prev_html_1st}
      </tr>
{$prev_html_rest}
HERE;
                $prev_runnum    = $runnum;
                $prev_html_1st  = null;
                $prev_html_rest = '';
                $prev_num_files = 0;
            }
        }
        $file_name = sprintf("e%d-r%04d-s%02d-c%02d.xtc",
            $experiment->id(),
            $runnum,
            $file->stream(),
            $file->chunk());

        $daq_open_time     = $file->open_time();
        $daq_open_time_str = is_null($daq_open_time) ? '&nbsp' : $daq_open_time->toStringShort();

        $daq2ffb_status      = 'pending';
        $daq2ffb_start_time  = null;
        $daq2ffb_finish_time = null;

        if (array_key_exists($file_name, $files_daq2ffb)) {
            $daq2ffb_start_time = $files_daq2ffb[$file_name]->start_time();
            if ($daq2ffb_start_time) {
                $finish_time = $files_daq2ffb[$file_name]->stop_time();
                if ($finish_time) {
                    if ($files_daq2ffb[$file_name]->error_msg()) {
                        $daq2ffb_status = 'failed';
                    } else {
                        $daq2ffb_status = 'complete';
                        $daq2ffb_finish_time = $finish_time;
                    }
                } else {
                    $daq2ffb_status = 'in-progress';
                }
            }
        }
        $daq2ffb_start_time_str  = is_null($daq2ffb_start_time)  ? '&nbsp;' : $daq2ffb_start_time->toStringShort();
        $daq2ffb_finish_time_str = is_null($daq2ffb_finish_time) ? '&nbsp;' : format_seconds($daq2ffb_finish_time->sec - $daq2ffb_start_time->sec);

        $ffb2offline_status      = 'pending';
        $ffb2offline_start_time  = null;
        $ffb2offline_finish_time = null;

        if (($daq2ffb_status == 'complete') && array_key_exists($file_name, $files_ffb2offline)) {
            $ffb2offline_start_time = $files_ffb2offline[$file_name]->start_time();
            if ($ffb2offline_start_time) {
                $finish_time = $files_ffb2offline[$file_name]->stop_time();
                if ($finish_time) {
                    if ($files_ffb2offline[$file_name]->error_msg()) {
                        $ffb2offline_status = 'failed';
                    } else {
                        $ffb2offline_status = 'complete';
                        $ffb2offline_finish_time = $finish_time;
                    }
                } else {
                    $ffb2offline_status = 'in-progress';
                }
            }
        }
        $ffb2offline_start_time_str  = is_null($ffb2offline_start_time)  ? '&nbsp;' : $ffb2offline_start_time->toStringShort();
        $ffb2offline_finish_time_str = is_null($ffb2offline_finish_time) ? '&nbsp;' : format_seconds($ffb2offline_finish_time->sec - $ffb2offline_start_time->sec);

        $size_bytes = null;

        foreach (FileMgrIrodsDb::instance()->find_file ($experiment->instrument()->name(),$experiment->name(), $file_type, $file_name) as $r) {
            if ($r->run == $runnum) {
                foreach ($r->files as $f) {
                    $size_bytes = $f->size;
                    break;
                }
            }
        }

        $daq2ffb_service_delay   = format_seconds($now->sec - $daq_open_time->sec);
        $ffb2offline_service_delay = format_seconds($now->sec - $daq_open_time->sec);

        $status = '';

        switch ($daq2ffb_status) {
            case'complete':
                $daq2ffb_service_delay  = format_seconds($daq2ffb_finish_time->sec - $daq_open_time->sec);
                switch ($ffb2offline_status) {
                    case 'complete':
                        $ffb2offline_service_delay = format_seconds($ffb2offline_finish_time->sec - $daq_open_time->sec);
                        $status = '<span style="color:green;">complete</span>';
                        break;
                    case 'in-progress':
                        $status = '<span>migrating from FFB to OFFLINE...</span>';
                        break;
                    case 'pending':
                        $status = '<span style="color:blue;">waiting to migrate from FFB to OFFLINE...</span>';
                        break;
                    case 'failed':
                        $status = '<span style="color:red;">FFB to OFFLINE migration has failed</span>';
                        break;
                }
                break;
            case 'in-progress':
                $status = '<span>pulling from DAQ to FFB...</span>';
                break;
            case 'pending':
                $status = '<span style="color:blue;">waiting to migrate from DAQ to FFB...</span>';
                break;
            case 'failed':
                $status = '<span style="color:red;">migration from DAQ to FFB has failed</span>';
                break;
        }

        $size_str = '&nbsp;';
        $daq2ffb_speed_mbps = '&nbsp;';
        $ffb2offline_speed_mbps = '&nbsp;';

        if ($size_bytes) {
            $size_str = $f->size < BYTES_IN_GB ? '<b>'.(intval($size_bytes / BYTES_IN_MB)).'</b> MB' : sprintf( "<b>%0.1f</b>", $f->size / BYTES_IN_GB).' GB';
            if ($daq2ffb_status == 'complete') {
                $size_mb = intval($size_bytes / BYTES_IN_MB);
                $transfer_time_sec = $daq2ffb_finish_time->sec - $daq2ffb_start_time->sec;
                if ($transfer_time_sec > 0.)
                    $daq2ffb_speed_mbps = sprintf("<b>%.1f</b> MB/s", $size_mb / $transfer_time_sec);
            }
            if ($ffb2offline_status == 'complete') {
                $size_mb = intval($size_bytes / BYTES_IN_MB);
                $transfer_time_sec = $ffb2offline_finish_time->sec - $ffb2offline_start_time->sec;
                if ($transfer_time_sec > 0.)
                    $ffb2offline_speed_mbps = sprintf("<b>%.1f</b> MB/s", $size_mb / $transfer_time_sec);
            }
        }
        $html =<<<HERE
        <td class="table_cell " >{$file_name}</td>
        <td class="table_cell table_cell_highlight"               >{$daq_open_time_str}</td>
        <td class="table_cell "                     align="right" >{$size_str}</td>
        <td class="table_cell table_cell_highlight"               >{$daq2ffb_start_time_str}</td>
        <td class="table_cell "                     align="right" >{$daq2ffb_finish_time_str}</td>
        <td class="table_cell "                     align="right" >{$daq2ffb_speed_mbps}</td>
        <td class="table_cell "                     align="right" >{$daq2ffb_service_delay}</td>
        <td class="table_cell table_cell_highlight"               >{$ffb2offline_start_time_str}</td>
        <td class="table_cell "                     align="right" >{$ffb2offline_finish_time_str}</td>
        <td class="table_cell "                     align="right" >{$ffb2offline_speed_mbps}</td>
        <td class="table_cell "                     align="right" >{$ffb2offline_service_delay}</td>
        <td class="table_cell table_cell_right "                  >{$status}</td>

HERE;
        if (is_null($prev_html_1st)) $prev_html_1st  = $html;
        else                         $prev_html_rest .=<<<HERE
      <tr class="table_active_row run_{$prev_runnum}" id="{$prev_runnum}" >
{$html}
      </tr>

HERE;
    }
    if ($prev_num_files) {
        $rows = $prev_num_files + 1;
        print <<<HERE
      <tr class="table_active_row run_{$prev_runnum}" id="{$prev_runnum}" >
        <td class="table_cell " valign="top" align="right" rowspan="{$rows}" >{$prev_runnum}</td>
{$prev_html_1st}
      </tr>
{$prev_html_rest}

HERE;
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