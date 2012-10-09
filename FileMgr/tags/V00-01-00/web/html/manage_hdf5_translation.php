<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=encoding">
<title>View/Manage HDF5 Translation</title>

<style type="text/css">
  body {
    /*
    margin: 0;
    padding: 0;
    background-color:#e0e0e0;
    */
  }
  .lb_label {
    text-align: left;
    /*
    color:#0071bc;
    */
    font-weight: bold;
  }
  a,.lb_link {
    text-decoration: none;
    font-weight: bold;
    color: #0071bc;
  }
  a:hover,a.lb_link:hover {
    color: red;
  }
  .first_col_hdr {
    padding-left:5px;
    /*
    padding-bottom:5px;
    padding-right:5px;
    */
    font-weight:bold;
    text-align:righ;
    background-color:#c0c0c0;
  }
  .col_hdr {
    padding-left:5px;
  /*
    padding-bottom:5px;
    padding-left:15px;
    padding-right:5px;
    */
    font-weight:bold;
    text-align:left;
    background-color:#c0c0c0;
  }
  .col_hdr_right {
  /*
    padding-bottom:5px;
    padding-left:15px;
    padding-right:5px;
    */
    font-weight:bold;
    text-align:right;
  }
  .first_separator {
    margin-right:5px;
    height:1px;
    border-top:solid 1px #c0c0c0;
  }
  .separator {
    height:1px;
    margin-right:5px;
    padding-left:10px;
    border-top:solid 1px #c0c0c0;
  }
</style>

<script type="text/javascript" src="js/Loader.js"></script>

</head>

<body>

<div style="padding:10;">

<h1>Controllers</h1>

<div style="margin-left:20; padding:10;">
  <table><thead>
    <tr>
      <td style="width: 50px;" class="first_col_hdr"><b>Id</b></td>
      <td style="width:270px;" class="col_hdr"><b>Host</b></td>
      <td style="width: 50px;" class="col_hdr"><b>PID</b></td>
      <td style="width: 75px;" class="col_hdr"><b>Status</b></td>
      <td style="width:160px;" class="col_hdr"><b>Started</b></td>
      <td style="width:160px;" class="col_hdr"><b>Stopped</b></td>
      <td style="width: 75px;" class="col_hdr"><b>Log File</b></td>
    </tr>
    <!-- 
    <tr>
      <td><div class="first_separator"></div></td>
      <td><div class="separator"></div></td>
      <td><div class="separator"></div></td>
      <td><div class="separator"></div></td>
      <td><div class="separator"></div></td>
      <td><div class="separator"></div></td>
      <td><div class="separator"></div></td>
    </tr>
    -->
  </thead></table>
  <div style="width:950; height:85; overflow: auto;">

<?php

require_once( 'regdb/regdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use RegDB\RegDBHtml;

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

function pre( $str, $width=null ) {
    if( is_null( $width )) return '<pre>'.$str.'</pre>';
    return '<pre>'.sprintf( "%{$width}s", $str ).'</pre>';
}

try {

    $systems = FileMgrIfaceCtrlWs::systems();

    $num_rows = 20 * count( $systems );

    $con = new RegDBHtml( 0, 0, 900, $num_rows );

    $row = 0;
    foreach( $systems as $s ) {

        $log_url = '';
        if( $s->log_url != '' )
            $log_url = pre( '<a href="'.$s->log_url.'" target="_blank" title="click to open the log file in a separate tab or window">View</a>' );

        $color = null; // '#c0c0c0';

        $con->value(   5, $row, pre( $s->id      ), $color );
        $con->value(  65, $row, pre( $s->host    ), $color );
        $con->value( 345, $row, pre( $s->pid     ), $color );
        $con->value( 405, $row, pre( $s->status  ), $color );
        $con->value( 485, $row, pre( $s->started ), $color );
        $con->value( 655, $row, pre( $s->stopped ), $color );
        $con->value( 825, $row, $log_url,           $color );

        $row += 20;
    }
    print $con->html();
?>

  </div>
</div>

<h1>Translation Requests</h1>

<div style="margin-left:20px; width:920px;">
  <div style="float:left;">
    Experiment
    <select name="experiment" id="experiment" onchange="select_experiment()">
      <option value=""></option>
<?php
    $experiments = FileMgrIfaceCtrlWs::experiments();
    $experiments2instruments = array();
    foreach( $experiments as $e ) {
    	$experiments2instruments[$e->experiment] = $e->instrument;
    }
    foreach( array_keys( $experiments2instruments ) as $e ) {
    	echo <<<HERE
      <option value="$e">$e</option>

HERE;
    }
?>
    </select>
  </div>
  <div style="float:right; padding:10px; border:solid 1px #c0c0c0;">
    <table><thead>
      <tr>
        <td>&nbsp;</td>
        <td align="center"></td>
        <td>&nbsp;</td>
        <td align="center">Begin</td>
        <td align="center">End</td>
      </tr>
      <tr>
        <td>Runs</td>
        <td>
          <input id="filter_begin_run" size="4" name="begin_run" type="text" value="" style="padding:1px;" title="the smallest run number" disabled="disabled" /> -
          <input id="filter_end_run"   size="4" name="end_run"   type="text" value="" style="padding:1px;" title="the largest run number" disabled="disabled" /></td>
        <td style="padding-left:10px;">Created</td>
        <td><input id="filter_begin_created" name="begin_created" type="text" value="" style="padding:1px;" title="when the requests began being created" disabled="disabled" /></td>
        <td><input id="filter_end_created"   name="end_created"   type="text" value="" style="padding:1px;" title="when the requests ended up being created" disabled="disabled" /></td>
      </tr>
      <tr>
        <td>Status</td>
        <td>
          <select id="filter_status" name="filter_status" onchange="apply_filter()" disabled="disabled">
            <option value=""></option>
            <option value="Initial_Entry ">Initial_Entry</option>
            <option value="Waiting_Translation ">Waiting_Translation</option>
            <option value="Empty_Fileset ">Empty_Fileset</option>
            <option value="H5Dir_Error ">H5Dir_Error</option>
            <option value="Being_Translated ">Being_Translated</option>
            <option value="Translation_Error ">Translation_Error</option>
            <option value="Archive_Error ">Archive_Error</option>
            <option value="Complete ">Complete</option>
          </select></td>
        <td style="padding-left:10px;">Started</td>
        <td><input id="filter_begin_started" name="begin_started" type="text" value="" style="padding:1px;" title="the oldest time the requests were started" disabled="disabled" /></td>
        <td><input id="filter_end_started"   name="end_started"   type="text" value="" style="padding:1px;" title="the newest time the requests were started" disabled="disabled" /></td>
      </tr>
      <tr>
        <td>&nbsp;</td>
        <td>&nbsp;</td>
        <td style="padding-left:10px;">Stopped</td>
        <td><input id="filter_begin_stopped" type="text" name="begin_stopped" value="" style="padding:1px;" title="the oldest time the requests were stopped" disabled="disabled" /></td>
        <td><input id="filter_end_stopped"   type="text" name="end_stopped"   value="" style="padding:1px;" title="the newest time the requests were stopped" disabled="disabled" /></td>
      </tr>
    </thead></table>
    <div style="margin-top:20px;">
      <center>
        <button id="apply_filter_button" onclick="apply_filter()" disabled="disabled">Search</button>
        <button id="reset_filter_button" onclick="reset_filter_and_search()" disabled="disabled">Reset Filter</button>
      </center>
    </div>
  </div>
  <div style="clear:both;"></div>

<script language="javascript">
var experiments2instruments = new Array();
<?php
    foreach( array_keys( $experiments2instruments ) as $e ) {
    	$i = $experiments2instruments[$e];
    	echo <<<HERE
experiments2instruments['$e']='$i';

HERE;
    }
?>

var experiment_name = '';
var instrument_name = '';

function apply_filter() {
    if( experiment_name == '' ) {
        document.getElementById('experiment_requests').innerHTML='';
    } else {

        var filter_begin_run_value     = document.getElementById('filter_begin_run'    ).value;
        var filter_end_run_value       = document.getElementById('filter_end_run'      ).value;
        var filter_status_value        = document.getElementById('filter_status').options[ document.getElementById('filter_status').selectedIndex ].value;
        var filter_begin_created_value = document.getElementById('filter_begin_created').value;
        var filter_end_created_value   = document.getElementById('filter_end_created'  ).value;
        var filter_begin_started_value = document.getElementById('filter_begin_started').value;
        var filter_end_started_value   = document.getElementById('filter_end_started'  ).value;
        var filter_begin_stopped_value = document.getElementById('filter_begin_stopped').value;
        var filter_end_stopped_value   = document.getElementById('filter_end_stopped'  ).value;

        var filter = '';
    	if( filter_begin_run_value     != '' ) filter += '&begin_run='    +encodeURIComponent( filter_begin_run_value );
    	if( filter_end_run_value       != '' ) filter += '&end_run='      +encodeURIComponent( filter_end_run_value );
    	if( filter_status_value        != '' ) filter += '&status='       +encodeURIComponent( filter_status_value );
    	if( filter_begin_created_value != '' ) filter += '&begin_created='+encodeURIComponent( filter_begin_created_value );
    	if( filter_end_created_value   != '' ) filter += '&end_created='  +encodeURIComponent( filter_end_created_value );
    	if( filter_begin_started_value != '' ) filter += '&begin_started='+encodeURIComponent( filter_begin_started_value );
    	if( filter_end_started_value   != '' ) filter += '&end_started='  +encodeURIComponent( filter_end_started_value );
    	if( filter_begin_stopped_value != '' ) filter += '&begin_stopped='+encodeURIComponent( filter_begin_stopped_value );
    	if( filter_end_stopped_value   != '' ) filter += '&end_stopped='  +encodeURIComponent( filter_end_stopped_value );

        document.getElementById('experiment_requests').innerHTML='<p>Loading...<p>';
        load( '../explorer/ws/DisplayExperimentRequests.php?instr='+instrument_name+'&exp='+experiment_name+filter, 'experiment_requests' );
    }
}

function disable_filter_if( yesOrNo ) {
    document.getElementById('apply_filter_button' ).disabled = yesOrNo;
    document.getElementById('reset_filter_button' ).disabled = yesOrNo;
    document.getElementById('filter_begin_run'    ).disabled = yesOrNo;
    document.getElementById('filter_end_run'      ).disabled = yesOrNo;
    document.getElementById('filter_status'       ).disabled = yesOrNo;
    document.getElementById('filter_begin_created').disabled = yesOrNo;
    document.getElementById('filter_end_created'  ).disabled = yesOrNo;
    document.getElementById('filter_begin_started').disabled = yesOrNo;
    document.getElementById('filter_end_started'  ).disabled = yesOrNo;
    document.getElementById('filter_begin_stopped').disabled = yesOrNo;
    document.getElementById('filter_end_stopped'  ).disabled = yesOrNo;
}
function enable_filter() { disable_filter_if( false ); }
function disable_filter() { disable_filter_if( true ); }

function set_filter( filter_params ) {
    document.getElementById('filter_begin_run'    ).value = filter_params.begin_run;
    document.getElementById('filter_end_run'      ).value = filter_params.end_run;
    document.getElementById('filter_status'       ).selectedIndex = 0;
    document.getElementById('filter_begin_created').value = filter_params.begin_created;
    document.getElementById('filter_end_created'  ).value = filter_params.end_created;
    document.getElementById('filter_begin_started').value = filter_params.begin_started;
    document.getElementById('filter_end_started'  ).value = filter_params.end_started;
    document.getElementById('filter_begin_stopped').value = filter_params.begin_stopped;
    document.getElementById('filter_end_stopped'  ).value = filter_params.end_stopped;
}

function reset_filter() {
	var url = '../explorer/ws/LimitsOfExperimentRequests.php?instr='+instrument_name+'&exp='+experiment_name;
	load_then_call (
		url,
		function( filter_params ) {
			set_filter( filter_params.ResultSet.Result[0] );
		},
		function() {
			alert('failed in URL'+url);
		}
	);
}

function reset_filter_and_search() {
	set_filter(
		{	'begin_run'     : '',
			'end_run'       : '',
			'begin_created' : '',
			'end_created'   : '',
			'begin_started' : '',
			'end_started'   : '',
			'begin_stopped' : '',
			'end_stopped'   : ''
		}
	);
	apply_filter();
	reset_filter();
}

function select_experiment() {
	set_filter(
		{	'begin_run'     : '',
			'end_run'       : '',
			'begin_created' : '',
			'end_created'   : '',
			'begin_started' : '',
			'end_started'   : '',
			'begin_stopped' : '',
			'end_stopped'   : ''
		}
	);
	var index = document.getElementById('experiment').selectedIndex;
	if( index == 0 ) {
		disable_filter();
		experiment_name = '';
		instrument_name = '';
	} else {
        enable_filter();
        experiment_name = document.getElementById('experiment').options[index].value;
        instrument_name = experiments2instruments[experiment_name];
        apply_filter();
    	reset_filter();
	}
}
</script>

  <div style="margin-top:20px;">
    <table><thead>
      <tr>
        <td style="width: 50px;" class="first_col_hdr"><b>Id</b></td>
        <td style="width: 50px;" class="col_hdr"><b>Run</b></td>
        <td style="width:150px;" class="col_hdr"><b>Status</b></td>
        <td style="width: 50px;" class="col_hdr"><b>Priority</b></td>
        <td style="width:160px;" class="col_hdr"><b>Created</b></td>
        <td style="width:160px;" class="col_hdr"><b>Started</b></td>
        <td style="width:160px;" class="col_hdr"><b>Stopped</b></td>
        <td style="width: 75px;" class="col_hdr"><b>Log File</b></td>
      </tr>
      <!-- 
      <tr>
        <td><div class="first_separator"></div></td>
        <td><div class="separator"></div></td>
        <td><div class="separator"></div></td>
        <td><div class="separator"></div></td>
        <td><div class="separator"></div></td>
        <td><div class="separator"></div></td>
        <td><div class="separator"></div></td>
        <td><div class="separator"></div></td>
      </tr>
      -->
    </thead></table>
    <div id="experiment_requests" style="width:920px; height:360px; overflow: auto;"></div>
  </div>

</div>
	
<?php
} catch( FileMgrException $e ) {
	echo $e->toHtml();
}
?>

</div>

</body>

</html>