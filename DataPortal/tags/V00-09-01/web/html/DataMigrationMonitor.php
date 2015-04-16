<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use DataPortal\Config;

use RegDB\RegDB;

$refresh_interval_sec = 5;

try {
    $authdb = AuthDB::instance();
    $authdb->begin();

    $config  = Config::instance();
    $config->begin();

    $regdb = RegDB::instance();
    $regdb->begin();

    $subscriber    = $authdb->authName();
    $address       = $subscriber.'@slac.stanford.edu';
    $is_subscribed = !is_null( $config->check_if_subscribed4migration ( $subscriber, $address ));

?>
<!DOCTYPE html> 
<html>
<head>

<title>Data Migration Monitor</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script> 

<style type="text/css">

body {
  margin: 0;
  padding: 0;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
.comment {
  padding-left:20px;
  max-width: 720px;
  font-size: 13px;
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
  font-size: 75%;
}
td.table_hdr_nodecor {
  padding: 2px 8px 2px 8px;
  border-top: none;
  border-left: none;
  font-family: Arial, sans-serif;
  font-size: 75%;
}
td.table_cell {
  border:solid 1px #d0d0d0;
  border-bottom: none;
  border-left: none;
  padding: 4px 8px 4px 8px;
  font-family: Arial, sans-serif;
  font-size: 75%;
}
td.table_cell_left {
  font-weight: bold;
}
td.table_cell_right {
  border-right: none;
}
td.table_cell_top {
  border-top: none;
}
td.table_cell_within_group {
  border-bottom: none;
}

.visible {
  display: block;
}

.hidden {
  display: none;
}

</style>

<script type="text/javascript">

var refresh_interval = 1000 * <?php echo $refresh_interval_sec; ?>;
var refresh_enabled = false;
var refresh_timer = null;

function request_update() {
    $('#search_submit_button').button('disable');
    var params = {};
    if(  $('#file_filter_form').find('input[name="active"]'   ).attr('checked')) params.active = '';
    if(  $('#file_filter_form').find('input[name="recent"]'   ).attr('checked')) params.recent = '';
    params.instrument = $( '#file_filter_form').find('select[name="instrument"]').val();
    if( !$('#file_filter_form').find('input[name="archived"]' ).attr('checked')) params.skip_non_archived = '';
    if( !$('#file_filter_form').find('input[name="local"]'    ).attr('checked')) params.skip_non_local    = '';
    if( !$('#file_filter_form').find('input[name="migrated"]' ).attr('checked')) params.skip_non_migrated = '';
    switch( $( '#file_filter_form').find('select[name="ignore"]').val()) {
    case '1h': params.ignore_1h = ''; break;
    case '1d': params.ignore_1d = ''; break;
    case '1w': params.ignore_1w = ''; break;
    }
    var jqXHR = $.get(
        '../portal/ws/DataMigrationMonitorImpl.php', params,
        function(data) {
            $('#search_result').html(data);
        },
        'HTML'
    ).error( function () {
        alert('failed because of: '+jqXHR.statusText); }
    ).complete( function() {
        if( refresh_enabled ) {
            refresh_timer = window.setTimeout( 'request_update()', refresh_interval );
            return;
        }
        if( refresh_timer != null ) {
            window.clearTimeout(refresh_timer);
            refresh_timer = null;
        }
        $('#search_submit_button').button('enable'); }
    );
}

$(function() {

    $('#unsubscribe_button').button().click(function() {
        var params = {};
        var jqXHR = $.get(
            '../portal/ws/DataMigrationSubscriptionToggle.php', params,
            function(data) {
                $('#subscribe_area'  ).removeClass('hidden').addClass('visible');
                $('#unsubscribe_area').removeClass('visible').addClass('hidden');
            },
            'HTML'
        ).error( function () {
            alert('failed because of: '+jqXHR.statusText); }
        ).complete( function() {}
        );
    });
    $('#subscribe_button').button().click(function() {
        var params = {};
        var jqXHR = $.get(
            '../portal/ws/DataMigrationSubscriptionToggle.php', params,
            function(data) {
                $('#subscribe_area'  ).removeClass('visible').addClass('hidden');
                $('#unsubscribe_area').removeClass('hidden').addClass('visible');
            },
            'HTML'
        ).error( function () {
            alert('failed because of: '+jqXHR.statusText); }
        ).complete( function() {}
        );
    });
    $('#search_submit_button').button().click(function() {
        request_update();
    });
    $('#automatic_search_checkbox').change(function() {
        if( $(this).attr('checked')) {
            refresh_enabled = true;
            request_update();
        } else {
            refresh_enabled = false;
        }
    });
});

</script>

</head>

<body>

<div style="padding:20px;">

  <span style="font-size:32px; font-weight:bold;">Data Migration Monitor</span>
  <br>
  <br>
  <div style="padding-left:20px; font-family: Arial, sans-serif; font-size:80%;">
    <div style="float:left;">
      <form id="file_filter_form" style="margin-bottom:0px;" action="../portal/ws/DataMigrationMonitorImpl.php" method="get">
        <div style="float:left; margin-right:20px;">
          <b>Select experiments which:</b><br>
          <div style="padding-left:20px; padding-top:5px; padding-bottom:5px;">
            <input type="checkbox" name="active"   value="1" checked="checked" /> are presently <a href="../portal/experiment_switch" target="_blank"
                   title="see Experiment Switch for a list of currently active experiment">active</a>, or<br>
            <input type="checkbox" name="recent"   value="1" checked="checked" /> took last run within 7 days
          </div>
          <b>Of instrument:</b>
          <select name="instrument">
            <option value="" >any</option>
<?php
    foreach( $regdb->instruments() as $instrument )
        if( !$instrument->is_location())
            print <<<HERE
            <option value="{$instrument->name()}" >{$instrument->name()}</option>

HERE;
?>
          </select>
        </div>
        <div style="float:left; margin-right:20px;">
          <b>Report files which are:</b><br>
          <div style="padding-left:10px; padding-top:5px; padding-bottom:5px;">
            <input type="checkbox" name="archived" value="1" checked="checked" /> not archived to tape, or<br>
            <input type="checkbox" name="local"    value="1" checked="checked" /> not available on disk, or<br>
            <input type="checkbox" name="migrated" value="1" checked="checked" /> never migrated from DAQ or deleted<br>
          </div>
          <b>And created within last:</b>
          <select name="ignore">
            <option value="1w" >week</option>
            <option value="1d" >day (24 hours)</option>
            <option value="1h" >hour</option>
            <option value="" >&lt;no restriction&gt;</option>
          </select>
        </div>
        <div style="clear:both;"></div>
      </form>
    </div>
    <div style="float:right; padding:10px; border:1px solid #c0c0c0; background-color:#f0f0f0;">
      <b>Alerts by e-mail:</b><br>
      <div id="subscribe_area" class="<?php echo $is_subscribed ? 'hidden' : 'visible' ?>" style="padding-left:10px; padding-top:5px;">
        Your SLAC account <b><?php echo $subscriber ?></b> is <b>NOT</b> subscribed for<br>
        hourly reports on delayed migrations. Subscribe<br>
        to receive alerts at: <b><?php echo $address ?></b>.<br>
        <button id="subscribe_button" style="font-size:9px; margin-top:10px;">Subscribe</button>
      </div>
      <div id="unsubscribe_area" class="<?php echo $is_subscribed ? 'visible' : 'hidden' ?>" style="padding-left:10px; padding-top:5px;">
        Your SLAC account <b><?php echo $subscriber ?></b> is already subscribed<br>
        for hourly reports on delayed migrations. Alerts are<br>
        sent to: <b><?php echo $address ?></b>.<br>
        <button id="unsubscribe_button" style="font-size:9px; margin-top:10px;">Unsubscribe</button>
      </div>
    </div>
    <div style="clear:both;"></div>
    <div style="margin-top:10px;">
      <input type="submit" id="search_submit_button" value="Search Now" />
      or check
      <input type="checkbox" id="automatic_search_checkbox" name="autoupdate" value="1" />
      to keep updating every <?php echo $refresh_interval_sec; ?> seconds
    </div>
  </div>
  <div id="search_result" style="padding-left:20px; padding-top:20px; width:1100px;"></div>
</div>

</body>
</html>

<?php

    $config->commit();
    $regdb->commit();

} catch( Exception $e ) { print '<pre style="padding:10px; border-top:solid 1px maroon; color:maroon;">'.print_r($e,true).'</pre>'; }

?>
