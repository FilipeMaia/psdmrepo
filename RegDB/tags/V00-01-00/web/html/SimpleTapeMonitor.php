<!DOCTYPE html">
<html>
<head>

<title>Monitor File Restore Requests</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script> 

<style type="text/css">

td.table_hdr {
  background-color:#d0d0d0;
  padding: 2px 8px 2px 8px;
  border: solid 1px #000000;
  border-top: none;
  border-left: none;
  font-family: Arial, sans-serif;
  font-weight: bold;
  font-size: 12px;
}
td.table_cell {
  border:solid 1px #d0d0d0;
  border-top: none;
  border-left: none;
  padding: 2px 8px 2px 8px;
  font-family: Arial, sans-serif;
  font-size: 12px;
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
    background-color:#f0f0f0;
}

input {
  padding-left: 2px;
  padding-right: 2px;
}

.highlighted {
  font-weight:bold;
}

</style>

</head>

<script type="text/javascript">

function file_action(action, exper_id, runnum, file_type, irods_filepath) {

    var button = $(this);

    button.button('disable');

    var params = {
        action:         action,
        exper_id:       exper_id,
        runnum:         runnum,
        storage:        'SHORT-TERM',
        type:           file_type,
        irods_filepath: irods_filepath,
        force:          ''
    };
    var jqXHR = $.get('../portal/ws/RestoreOneFile.php', params, function(data) {
        var result = eval(data);
        if(result.status != 'success') {
            alert(result.message);
            button.button('enable');
            return;
        }
        button.button('disable');
    },
    'JSON').error(function () {
        alert('failed because of: '+jqXHR.statusText);
        button.button('enable');
    });
}

$(function() {
    $('#tabs').tabs();
    $('button.resubmit').button();
    $('button.cancel').button();
});

</script>

<body>

  <div style="padding:10px; padding-left:20px;">

    <h2>File Restore Requests</h2>

    <div id="tabs" style="padding-left:10px; font-size:12px;">
      <ul>
        <li><a href="#history">History</a></li>
      </ul>
      <div id="default" >
        <div style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >

          <table><tbody>
            <tr>
              <td class="table_hdr" rowspan="2"                >Experiment</td>
              <td class="table_hdr" rowspan="2"  align="right" >Run</td>
              <td class="table_hdr" rowspan="2"                >Type</td>
              <td class="table_hdr" rowspan="2"                >File</td>
              <td class="table_hdr" rowspan="2"                >User</td>
              <td class="table_hdr" rowspan="2"                >Request Time</td>
              <td class="table_hdr" rowspan="2"                >Status</td>
              <td class="table_hdr" colspan="4"                >Service Time / Delay</td>
              <td class="table_hdr" colspan="2"                >Performance</td>
              <td class="table_hdr" rowspan="2"                >Actions</td>
            </tr>
            <tr>
              <td class="table_hdr" align="right" >days</td>
              <td class="table_hdr" align="right" >h</td>
              <td class="table_hdr" align="right" >m</td>
              <td class="table_hdr" align="right" >s</td>
              <td class="table_hdr" align="right" >Size</td>
              <td class="table_hdr" align="right" >MB/s</td>
            </tr>
<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use DataPortal\Config;
use FileMgr\FileMgrIrodsDb;
use LusiTime\LusiTime;
use RegDB\RegDB;

define( 'BYTES_IN_MB', 1024 * 1024 );
define( 'BYTES_IN_GB', 1024 * BYTES_IN_MB );

function report_error ($msg) {
    print "<div style=\"color:red;\">Error: {$msg}</div>";
    exit;
}
try {
    AuthDB::instance()->begin();
    RegDB::instance()->begin();
    Config::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $can_read = AuthDB::instance()->hasPrivilege(AuthDB::instance()->authName(), null, 'StoragePolicyMgr', 'read');
    if (!$can_read) {
        header('Location: access_denied.html');
        exit;
    }
    $can_edit = AuthDB::instance()->hasPrivilege(AuthDB::instance()->authName(), null, 'StoragePolicyMgr', 'edit');

    $now = LusiTime::now();

    foreach (Config::instance()->file_restore_requests() as $request) {

        $exper_id   = $request['exper_id'];
        $experiment = RegDB::instance()->find_experiment_by_id($exper_id);
        if (is_null($experiment))
            report_error ("invalid experiment identifier {$exper_id} found in the database");
        $experiment_link = "<a href=\"../portal/index.php?exper_id={$experiment->id()}&app=datafiles:files\" target=\"_blank\">{$experiment->name()}</a>";

        $runnum    = $request['runnum'];
        $file_type = strtoupper($request['file_type']);

        $irods_filepath = $request['irods_filepath'];
        $irods_filepath_split = explode('/', $irods_filepath);
        $file_name = $irods_filepath_split[count($irods_filepath_split)-1];

        $uid = $request['requested_uid'];

        $requested_time_64 = $request['requested_time'];
        $requested_time    = LusiTime::parse($requested_time_64);
        if (is_null($requested_time)) report_error("invalid request timestamp {$requested_time_64} found in the database");

        $disk_ctime = null;
        $size_bytes = null;
        foreach (FileMgrIrodsDb::instance()->find_file ($experiment->instrument()->name(),$experiment->name(), $file_type, $file_name) as $r) {
            if ($r->run == $runnum) {
                foreach ($r->files as $f) {
                    if ($f->resource == 'lustre-resc') {
                        $disk_ctime = $f->ctime;
                        $size_bytes = $f->size;
                        break;
                    }
                }
                if (!is_null($disk_ctime)) break;
            }
        }
        $status =  '';
        $service_delay_sec = 0;
        if (is_null($disk_ctime)) {
            $status = '<span style="color:red;">pending...</span>';
            $service_delay_sec = $now->sec - $requested_time->sec;
        } else {
            $status = '<span style="color:green;">completed</span>';
            $ctime_time = new LusiTime(intval($disk_ctime));
            $service_delay_sec = $ctime_time->sec - $requested_time->sec;
        }
        $days    = 0;
        $hours   = 0;
        $minutes = 0;
        $seconds = 0;
        if ($service_delay_sec < 60) {
            $seconds = $service_delay_sec;
        } elseif ($service_delay_sec < 3600) {
            $minutes = intval($service_delay_sec / 60);
            $seconds = intval($service_delay_sec % 60);
        } elseif ($service_delay_sec < 24 * 3600) {
            $hours   = intval( $service_delay_sec / 3600);
            $minutes = intval(($service_delay_sec % 3600) / 60);
        } else {
            $days    = intval( $service_delay_sec / (24 * 3600));
            $hours   = intval(($service_delay_sec % (24 * 3600)) / 3600);
        }
        $size       = '';
        $speed_mbps = '';
        if ($size_bytes) {
            $size = $f->size < BYTES_IN_GB ? intval($size_bytes / BYTES_IN_MB).' MB' : sprintf( "%0.1f", $f->size / BYTES_IN_GB).' GB';
            $size_mb    = intval($size_bytes / BYTES_IN_MB);
            $speed_mbps = $service_delay_sec ? sprintf("%.1f", $size_mb / $service_delay_sec) : '';
        }
        $resubmit_action = "file_action('resubmit',{$exper_id}, {$runnum}, '{$file_type}', '{$irods_filepath}')";
        $cancel_action   = "file_action('cancel',  {$exper_id}, {$runnum}, '{$file_type}', '{$irods_filepath}')";
?>
            <tr>
              <td class="table_cell "                                ><?php echo $experiment_link; ?></td>
              <td class="table_cell " align="right"                  ><?php echo $runnum; ?></td>
              <td class="table_cell "                                ><?php echo $file_type; ?></td>
              <td class="table_cell "                                ><?php echo $file_name; ?></td>
              <td class="table_cell "                                ><?php echo $uid; ?></td>
              <td class="table_cell " style="white-space: nowrap;"   ><?php echo $requested_time->toStringShort(); ?></td>
              <td class="table_cell "                                ><?php echo $status; ?></td>
              <td class="table_cell table_cell_right " align="right" style="background-color:#f0f0f0;" ><?php echo $days    ? $days    : '&nbsp;'; ?></td>
              <td class="table_cell table_cell_right " align="right" ><?php echo $hours   ? $hours   : '&nbsp;'; ?></td>
              <td class="table_cell table_cell_right " align="right" ><?php echo $minutes ? $minutes : '&nbsp;'; ?></td>
              <td class="table_cell "                  align="right" ><?php echo $seconds ? $seconds : '&nbsp;'; ?></td>
              <td class="table_cell "                  align="right" ><?php echo $size       ? $size       : '&nbsp;'; ?></td>
              <td class="table_cell "                  align="right" ><?php echo $speed_mbps ? $speed_mbps : '&nbsp;'; ?></td>
              <td class="table_cell table_cell_right " style="white-space: nowrap;" >
<?php   if (is_null($disk_ctime)) { ?>
                <button class="resubmit" title="Re-submit the request" onclick="<?php echo $resubmit_action; ?>">Re-submit</button>                
                <button class="cancel"   title="Cancel the request"    onclick="<?php echo $cancel_action;   ?>">Cancel</button>                
<?php   } else { ?>
                &nbsp;
<?php   } ?>
              </td>
            </tr>
<?php
    }
    AuthDB::instance()->commit();
    RegDB::instance()->commit();
    Config::instance()->commit();
    FileMgrIrodsDb::instance()->commit();

} catch( Exception $e ) { report_error($e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'); }

?>
          </tbody></table>

        </div>
      </div>
    </div>
  </div>
</body>
</html>