<?php
require_once('AuthDB/AuthDB.inc.php');
require_once('RegDB/RegDB.inc.php');
require_once('FileMgr/FileMgr.inc.php');

/* Make sure the script is run with proper experiment identifier provided.
 * Such experiment must exist in the experiments Registry database.
 * 
 * TODO: Another possibility would be to select some default experiment
 * based on user account's membership, or on the information saved from
 * some previousl session.
 */
if( !isset( $_GET['exper_id'] )) {
	// Redirect to the experiment selection dialog
}
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' ) die( 'no valid experiment identifier provided to the script' );

try {

	$regdb = new RegDB();
	$regdb->begin();

	$experiment = $regdb->find_experiment_by_id( $exper_id );
	if( is_null( $experiment )) die( 'invalid experiment identifier provided to the script' );
?>

<!--
  ==================================================
  Data Portal framework for various Web applications
  ==================================================
  -->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"> 
<html> 
<head> 
<title>Data Portal of Experiment:</title> 
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 


<link type="text/css" href="/jquery/jquery-ui-1.8.5/themes/base/jquery-ui.css" rel="Stylesheet" />
<link type="text/css" href="/jquery/jquery-ui-themes-1.8.5/themes/redmond/jquery-ui.css" rel="Stylesheet" />	


<style type="text/css"> 

  body {
    margin:0px;
    padding:20px;

    /*
    font-family:"Times", serif;
    font-size:14px;
    */
    overflow: hidden;	/* Remove scroll bars on browser window */	
    font-size: 75%;
  }

  #header {
    padding-left:10px;
  }
  #title {
    font-family:"Times", serif;
    font-size:42px;
    font-weight:bold;
    text-align:left;
  }
  #experiment {
    font-family:"Times", serif;
    font-size:42px;
    color:#0071bc;
  }
  #login {
    font-size:14px;
  }

  a {
    text-decoration: none;
    font-weight: bold;
    color: #0071bc;
  }
  a:hover {
    color: red;
  }
  .first_col_hdr, .col_hdr {
    background-color: #d0d0d0;
    padding: 0.5em;
    font-weight: bold;
  }

</style>

<script type="text/javascript" src="/jquery/jquery-ui-1.8.5/jquery-1.4.2.js"></script>
<script type="text/javascript" src="/jquery/jquery-ui-1.8.5/ui/jquery-ui.js"></script>

<script type="text/javascript" src="Utilities.js"></script>
<script type="text/javascript" src="Loader.js"></script>

<script type="text/javascript">

<?php

    $auth_svc = AuthDB::instance();
    echo <<<HERE

/* ----------------------------------------
 * Authentication and authorization context
 * ----------------------------------------
 */
var auth_is_authenticated="{$auth_svc->isAuthenticated()}";
var auth_type="{$auth_svc->authType()}";
var auth_remote_user="{$auth_svc->authName()}";

var auth_webauth_token_creation="{$_SERVER['WEBAUTH_TOKEN_CREATION']}";
var auth_webauth_token_expiration="{$_SERVER['WEBAUTH_TOKEN_EXPIRATION']}";

function refresh_page() {
    window.location = "{$_SERVER['REQUEST_URI']}";
}

HERE;

?>

/*
 * Session expiration timer for WebAuth authentication.
 */
var auth_timer = null;
function auth_timer_restart() {
    if( auth_is_authenticated && ( auth_type == 'WebAuth' ))
        auth_timer = window.setTimeout( 'auth_timer_event()', 1000 );
}
var auth_last_secs = null;
function auth_timer_event() {

    var auth_expiration_info = document.getElementById( "auth_expiration_info" );
    var now = mktime();
    var seconds = auth_webauth_token_expiration - now;
    if( seconds <= 0 ) {
        $('#popupdialogs').html(
        	'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
        	'Your WebAuth session has expired. Press <b>Ok</b> or use <b>Refresh</b> button'+
        	'of the browser to renew your credentials.</p>'
        );
        $('#popupdialogs').dialog({
        	resizable: false,
        	modal: true,
        	buttons: {
        		"Ok": function() {
        			$( this ).dialog( "close" );
        			refresh_page();
        		}
        	},
        	title: 'Session Expiration Notification'
        });
        return;
    }
    var hours_left   = Math.floor(seconds / 3600);
    var minutes_left = Math.floor((seconds % 3600) / 60);
    var seconds_left = Math.floor((seconds % 3600) % 60);

    var hours_left_str = hours_left;
    if( hours_left < 10 ) hours_left_str = '0'+hours_left_str;
    var minutes_left_str = minutes_left;
    if( minutes_left < 10 ) minutes_left_str = '0'+minutes_left_str;
    var seconds_left_str = seconds_left;
    if( seconds_left < 10 ) seconds_left_str = '0'+seconds_left_str;

    auth_expiration_info.innerHTML=
        '<b>'+hours_left_str+':'+minutes_left_str+'.'+seconds_left_str+'</b>';

    auth_timer_restart();
}

function logout() {
	$('#popupdialogs').html(
		'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
    	'This will log yout out from the current WebAuth session. Are you sure?</p>'
	 );
	$('#popupdialogs').dialog( {
		resizable: false,
		modal: true,
		buttons: {
			"Yes": function() {
				$( this ).dialog( "close" );
	            document.cookie = 'webauth_wpt_krb5=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
	            document.cookie = 'webauth_at=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
	            refresh_page();
			},
			Cancel: function() {
				$( this ).dialog( "close" );
			}
		},
		title: 'Session Logout Warning'
	} );
}

var exper_id = '<?php echo $exper_id ?>';
var experiment_name = '<?php echo $experiment->name()?>';
var instrument_name = '<?php echo $experiment->instrument()->name()?>';

function apply_filter() {


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
        load( '../explorer/DisplayExperimentRequests.php?instr='+instrument_name+'&exp='+experiment_name+filter, 'experiment_requests' );
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
	var url = '../explorer/LimitsOfExperimentRequests.php?instr='+instrument_name+'&exp='+experiment_name;
	load_then_call (
		url,
		function( filter_params ) {
			set_filter( filter_params.ResultSet.Result[0] );
			enable_filter();
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

function create_buttons() {
	$( '#apply_filter_button' ).button();
	$( '#apply_filter_button' ).click(
		function() {
			apply_filter();
			return false;
		}
	);
	$( '#reset_filter_button' ).button();
	$( '#reset_filter_button' ).click(
		function() {
			reset_filter_and_search();
			return false;
		}
	);
}

/* --------------------------------------------------- 
 * The starting point where the JavaScript code starts
 * ---------------------------------------------------
 */
$(document).ready(
    function() {
        auth_timer_restart();
        $('#date').datepicker();
        $('#tabs').tabs();
        $('#tabs-hdf5').tabs();
        reset_filter_and_search();
    	create_buttons();
		$.get(
			'../logbook/DisplayExperiment.php',
			{ id : exper_id },
			function( data ) {
				$('#experiment_status').html( data );
			}
		);
    }
);

</script>

</head>
<body>

  <div id="header">
    <div style="float:left;">
      <span id="title">Data Portal of Experiment:</span>
      <span id="experiment"><a href="select_experiment.php" title="Switch to another experiment"><?php echo $experiment->instrument()->name(); ?>&nbsp;/&nbsp;<?php echo $experiment->name(); ?></a></span>
    </div>
    <div id="login" style="float:right;">
      <table><tbody>
        <tr>
          <td>&nbsp;</td>
          <td>[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]</td>
        </tr>
        <tr>
          <td>Welcome,&nbsp;</td>
          <td><p><b><?php echo $auth_svc->authName() ?></b></p></td>
        </tr>
        <tr>
          <td>Session expires in:&nbsp;</td>
          <td><p id="auth_expiration_info"><b>00:00.00</b></p></td>
        </tr>
      </tbody></table>
    </div>
    <div style="clear:both;"></div>
  </div>

  <!--
    ==========================================================================
    APPLICATIONS: HTML markup for applications within tabs has to be generated
                  by invoking specially designed PHP classes/functions.
    ==========================================================================
    -->

  <div id="tabs">
	<ul>
	  <li><a href="#tabs-1">Experiment</a></li>
	  <li><a href="#tabs-2">e-Log</a></li>
	  <li><a href="#tabs-3">Data Files</a></li>
	  <li><a href="#tabs-4">XTC/HDF5 Translation</a></li>
	  <li><a href="#tabs-5">My Account</a></li>
	</ul>
	<div id="tabs-1">
      <div id="experiment_status"></div>
	</div>
	<div id="tabs-2">
      <p>Electronic LogBook should be seen here. But firt we need to redesign it using JavaScript
      classes to mavoid various sorts of conflicts.</p>
	</div>
	<div id="tabs-3">
        <p>File explorer to show data files for this experiment only.</p>
        <input type="text" name="date" id="date" />
	</div>

 	<div id="tabs-4">
 
      <div id="tabs-hdf5">
	    <ul>
	      <li><a href="#tabs-hdf5-1">Manage</a></li>
	      <li><a href="#tabs-hdf5-2">History of Requests</a></li>
	    </ul>
	    <div id="tabs-hdf5-1">
          <p>Manage translation requests for runs here.</p>
	    </div>
	    <div id="tabs-hdf5-2">
  
      <div style="margin-left:20px; width:920px;">
        <div style="float:left; padding:10px;">
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
              <button id="apply_filter_button" class="ui-button ui-button-text-only ui-widget ui-state-default ui-corner-all">
                <span class="ui-button-text">Search</span>
              </button>
              <button id="reset_filter_button" class="ui-button ui-button-text-only ui-widget ui-state-default ui-corner-all">
                <span class="ui-button-text">Reset Filter</span>
              </button>
            </center>
          </div>
        </div>
        <div style="clear:both;"></div>
 	  </div>
      <div style="margin-top:20px;">
        <table><thead>
          <tr>
            <td style="width: 50px;"><span class="first_col_hdr">Id</span></td>
            <td style="width: 50px;"><span class="col_hdr">Run</span></td>
            <td style="width:150px;"><span class="col_hdr">Status</span></td>
            <td style="width: 50px;"><span class="col_hdr">Priority</span></td>
            <td style="width:160px;"><span class="col_hdr">Created</span></td>
            <td style="width:160px;"><span class="col_hdr">Started</span></td>
            <td style="width:160px;"><span class="col_hdr">Stopped</span></td>
            <td style="width:100px;"><span class="col_hdr">Log File</span></td>
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

      </div>
      </div>
	<div id="tabs-5">
      <p>User account information, privileges, POSIX groups, other experiments participation, subscriptions, etc.</p>
	</div>
  </div>

  <div id="popupdialogs" style="display:none;"></div>

</body>
</html>

<?php

  $auth_svc->commit();

} catch( AuthDBException  $e ) { print $e->toHtml();
} catch( FileMgrException $e ) { print $e->toHtml();
} catch( LogBookException $e ) { print $e->toHtml();
} catch( RegDBException   $e ) { print $e->toHtml();
}

?>
