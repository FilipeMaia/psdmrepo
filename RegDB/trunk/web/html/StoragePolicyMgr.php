<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use DataPortal\Config;
use DataPortal\DataPortal;
use LogBook\LogBook;
use LusiTime\LusiTime;
use RegDB\RegDB;

define( 'BYTES_IN_GB', 1024 * 1024 * 1024 );

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

try {
    AuthDB::instance()->begin();
    $can_read = AuthDB::instance()->hasPrivilege(AuthDB::instance()->authName(), null, 'StoragePolicyMgr', 'read');
    if (!$can_read) {
        header('Location: access_denied.html');
        exit;
    }
    $can_edit = AuthDB::instance()->hasPrivilege(AuthDB::instance()->authName(), null, 'StoragePolicyMgr', 'edit');
    //if (AuthDB::instance()->authName() === 'gapon') $can_edit = false;
    $logbook = LogBook::instance();
    $logbook->begin();

    $regdb = RegDB::instance();
    $regdb->begin();

    $config = Config::instance();
    $config->begin();
    
    $default_short_ctime_time_str  =        $config->get_policy_param('SHORT-TERM',  'CTIME');
    $default_short_retention       = intval($config->get_policy_param('SHORT-TERM',  'RETENTION'));
    $default_medium_ctime_time_str =        $config->get_policy_param('MEDIUM-TERM', 'CTIME');
    $default_medium_retention      = intval($config->get_policy_param('MEDIUM-TERM', 'RETENTION'));
    $default_medium_quota          = intval($config->get_policy_param('MEDIUM-TERM', 'QUOTA'));

    $instruments = array();
    $experiments_by_instrument = array();
    foreach ($logbook->instruments() as $instrument ) {

        if ($instrument->is_location()) continue;
        array_push ($instruments, $instrument->name());

        $experiments_by_instrument[$instrument->name()] = array();
        foreach ($logbook->experiments_for_instrument($instrument->name()) as $experiment) {
            if( $experiment->is_facility()) continue;
            array_push ($experiments_by_instrument[$instrument->name()], $experiment);
        }
    }

?>

<!DOCTYPE html">
<html>
<head>

<title>Report and modify Data Path parameter for known experiments</title>
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
td.table_hdr {
  background-color:#d0d0d0;
  padding: 2px 8px 2px 8px;
  border: solid 1px #000000;
  border-top: none;
  border-left: none;
  /*
  font-family: Arial, sans-serif;
  */
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
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

<script type="text/javascript">

var instruments = [];
var experiments = [];
<?php
    foreach ($instruments as $instrument_name) {
        echo "instruments.push('{$instrument_name}');\n";
        foreach ($experiments_by_instrument[$instrument_name] as $experiment) {
            $contacts = array();
            foreach (DataPortal::experiment_contacts( $experiment ) as $contact) {
                array_push($contacts, $contact);
            }
            $experiment_entry = json_encode(array(
                'instr_name' => $instrument_name,
                'exper_name' => $experiment->name(),
                'exper_id'   => $experiment->id(),
                'contacts'   => $contacts
            ));
            echo "experiments.push({$experiment_entry});\n";
        }
    }
?>

function load_stats_for_experiment (descr) {
    $.ajax({
        type: 'GET',
        url: '../regdb/ws/StoragePolicyStats.php',
        data: {
            instr_name: descr.instr_name,
            exper_name: descr.exper_name
        },
        success: function(data) {
            if( data.status != 'success' ) { alert(data.message); return; }
            $('#num_runs_'              +descr.exper_id).html(data.num_runs           ? data.num_runs           : '');
            $('#short_term_size_gb_'    +descr.exper_id).html(data.short_term_size_gb ? data.short_term_size_gb : '');
            $('#short_term_expiration_' +descr.exper_id).html(data.short_term_expiration);
            $('#medium_term_expiration_'+descr.exper_id).html(data.medium_term_expiration);
            $('#medium_usage_gb_'       +descr.exper_id).html(data.medium_usage_gb    ? data.medium_usage_gb    : '');
            $.ajax({
                type: 'GET',
                url: '../portal/ws/filemgr_files_search.php',
                data: {
                    exper_id: descr.exper_id
                },
                success: function(data) {
                    if( data.Status != 'success' ) { alert(data.Message); return; }
                    var total_size_gb = 0;
                    if( data.overstay['SHORT-TERM'] != undefined) total_size_gb = Math.floor(data.overstay['SHORT-TERM'].total_size_gb);
                    $('#short_term_expired_size_gb_'+descr.exper_id).html(total_size_gb ? total_size_gb : '').css('color','red');
                    if( total_size_gb ) {
                        var email_to      = descr.contacts[0];  // the first contact in the list only
                        var email_cc      = 'perazzo@slac.stanford.edu;gapon@slac.stanford.edu';
                        var email_subject = encodeURIComponent('Cleaning expired data files of LCLS experiment '+descr.exper_name);
                        var email_body    = encodeURIComponent(
'Dear LCLS user,\n'+
'\n'+
'You\'re receiving this communication because you\'re registered as the PI of experiment '+descr.exper_name+'. '+
'We would like to let you know that some of the data files you have at our disk storage system have either expired '+
'or about to expire in accordance with the LCLC Data Retention Policy. The expired files will be removed from '+
'the disks 1 week from the date of this message.\n'+
'\n'+
'The total amount if expired data: '+total_size_gb+' GB\n'+
'\n'+
'Please, note that you can always restore these files from tape using the \'File Manager\' page of the Web Portal '+
'of your experiment. Another possibility which you may consider is to move selected files to the MEDIUM-STORAGE class '+
'using the Web Portal.  This will prolong the disk stay of those files for up to 2 years from the day when the files '+
'were recorded. Be advised that the amount of data which you can keep in the MEDIUM-STORAGE is limited by a quota whose '+
'default value is 10 TB.\n'+
'You can learn about the Data Retention Policy and instructions for moving files between different storage levels at:\n'+
'\n'+
'https://confluence.slac.stanford.edu/display/PCDS/Data+Retention+Policy\n'+
'\n'+
'Please, do not hesitate to contact us directly or via pcds-help@slac.stanford.edu should you have any further '+
'questions regarding the Policy or the status of your files. And we would also appreciate if you spread this news '+
'to the other members of your experiment.\n\n'
                        );
                        var notify_pi = $('a.notify_pi[name="notify_pi_of_exper_id_'+descr.exper_id+'"]');
                        var notify_pi_href = 'mailto:'+email_to+'?cc='+email_cc+'&subject='+email_subject+'&body='+email_body;
                        notify_pi.attr('href',notify_pi_href).button('enable');
                        $('button.delete_expired[name="'+descr.exper_id+'"]').button('enable');
                    }
                },
                error: function() {
                    alert('The request can not go through due a failure to contact the server.');
                },
                dataType: 'json'
            });
        },
        error: function() {
            alert('The request can not go through due a failure to contact the server.');
        },
        dataType: 'json'
    });
}

var total_short_usage_files = 0;
var total_short_usage_tb = 0;
var total_short_expired_tb = {
    '2014-12-02': 0,
    '2015-01-02': 0,
    '2015-02-02': 0,
    '2015-03-02': 0,
    '2015-04-02': 0,
    '2015-05-02': 0,
    '2015-06-02': 0
};
var total_medium_usage_files = 0;
var total_medium_usage_tb = 0;
var total_medium_quota_tb = 0;

function load_stats_for_instrument (name) {
    $.ajax({
        type: 'GET',
        url: '../regdb/ws/StoragePolicyStats.php',
        data: {
            instr_name: name
        },
        success: function(data) {
            if( data.status != 'success' ) { alert(data.message); return; }
            $('#short_usage_files_' +name ).html(data.short_term_files     ? data.short_term_files     : '');
            $('#short_usage_tb_'    +name ).html(data.short_term_size_tb   ? data.short_term_size_tb   : '');
            $('#medium_usage_files_'+name ).html(data.medium_term_files    ? data.medium_term_files    : '');
            $('#medium_usage_tb_'   +name ).html(data.medium_term_size_tb  ? data.medium_term_size_tb  : '');
            $('#medium_quota_tb_'   +name ).html(data.medium_term_quota_tb ? data.medium_term_quota_tb : '');
            for (var day in data.short_expired_tb) {
                var short_expired_tb = data.short_expired_tb[day];
                $('#short_expired_tb_'+day+'_'+name).html(short_expired_tb ? short_expired_tb : '');
                total_short_expired_tb[day] += short_expired_tb;
                $('#total_short_expired_tb_'+day).html(total_short_expired_tb[day] ? total_short_expired_tb[day] : '');
            }
            total_short_usage_files += data.short_term_files;
            total_short_usage_tb    += data.short_term_size_tb;
            $('#total_short_usage_files' ).html(total_short_usage_files   ? total_short_usage_files   : '');
            $('#total_short_usage_tb'    ).html(total_short_usage_tb      ? total_short_usage_tb      : '');
            total_medium_usage_files += data.medium_term_files;
            total_medium_usage_tb    += data.medium_term_size_tb;
            $('#total_medium_usage_files').html(total_medium_usage_files  ? total_medium_usage_files  : '');
            $('#total_medium_usage_tb'   ).html(total_medium_usage_tb     ? total_medium_usage_tb     : '');
            total_medium_quota_tb    += data.medium_term_quota_tb;
            $('#total_medium_quota_tb'   ).html(total_medium_quota_tb     ? total_medium_quota_tb     : '');
            
        },
        error: function() {
            alert('The request can not go through due a failure to contact the server.');
        },
        dataType: 'json'
    });
}

$(function() {

    $('#tabs').tabs();

    $('button.default').button().button('disable').click(function() {
 
        var name = this.name;
        var button = $(this);

        button.button('disable');
        $('#comment_'+name).text('saving...');

        var params = {};
        params[this.name] = $('input[name="'+name+'"]').val();
        var jqXHR = $.get('../regdb/ws/SetStoragePolicy.php', params, function(data) {
            var result = eval(data);
            if(result.Status != 'success') {
                alert(result.Message);
                button.button('enable');
                $('#comment_'+name).text('failed');
                return;
            }
            button.button('disable');
            $('#comment_'+name).text('saved');
        },
        'JSON').error(function () {
            alert('failed because of: '+jqXHR.statusText);
            button.button('enable');
            $('#comment_'+name).text('failed');
        });
    });
    $('input.default').keyup(function() {
        $('button[name="'+this.name+'"]').button('enable');
    });

    $('button.experiment').button().button('disable').click(function() {

        var exper_id = this.name;
        var button = $(this);

        button.button('disable');
        $('#comment_'+exper_id).text('saving...');

        var params = {
            'exper_id'        : exper_id,
            'short_ctime'     : $('input[name="short_ctime_'     +exper_id+'"]').val(),
            'short_retention' : $('input[name="short_retention_' +exper_id+'"]').val(),
            'medium_quota'    : $('input[name="medium_quota_'    +exper_id+'"]').val(),
            'medium_ctime'    : $('input[name="medium_ctime_'    +exper_id+'"]').val(),
            'medium_retention': $('input[name="medium_retention_'+exper_id+'"]').val()
        };
        var jqXHR = $.get('../regdb/ws/SetStoragePolicy.php', params, function(data) {
            var result = eval(data);
            if(result.Status != 'success') {
                alert(result.Message);
                button.button('enable');
                $('#comment_'+exper_id).text('failed');
                return;
            }
            button.button('disable');
            $('#comment_'+exper_id).text('saved');
        },
        'JSON').error(function () {
            alert('failed because of: '+jqXHR.statusText);
            button.button('enable');
            $('#comment_'+exper_id).text('failed');
        });
    });
    $('button.delete_expired').button().button('disable').click(function() {

        var exper_id = this.name;
        var button = $(this);

        button.button('disable');
        $('#comment_'+exper_id).text('deleting...');

        var params = {
            'exper_id': exper_id,
            'storage' : 'SHORT-TERM'
        };
        var jqXHR = $.get('../portal/ws/DeleteExpiredFiles.php', params, function(data) {
            var result = eval(data);
            if(result.status != 'success') {
                alert(result.message);
                button.button('enable');
                $('#comment_'+exper_id).text('failed');
                return;
            }
            button.button('enable');
            $('#comment_'+exper_id).text('deleted');
        },
        'JSON').error(function () {
            alert('failed because of: '+jqXHR.statusText);
            button.button('enable');
            $('#comment_'+exper_id).text('failed');
        });
    });
    $('a.notify_pi').button().button('disable').click(function() {
        $(this).button('disable');
    });
    $('input.experiment').keyup(function() {
        var exper_id = parseInt(this.name.substr(this.name.lastIndexOf('_')+1));
        var button = $('button[name="'+exper_id+'"]');
        button.button('enable');
    });
    for (var i in experiments) {
        var descr = experiments[i];
        load_stats_for_experiment(descr);
    }
    for (var i in instruments) {
        var name = instruments[i];
        load_stats_for_instrument(name);
    }
});

</script>

</head>

<body>

  <div style="padding:10px; padding-left:20px;">

    <h2>View/Modify Experiment Data Retention Policies</h2>

    <div class="comment" >
      <p>This application is meant to view and (if your account has sufficient privileges) to modify
      parameters of the data retention policy. The <b>Default Policies</b> tab allows to set default
      policies affecting all experiments.  These values can be changed for individual experiments
      using experiment-specific tables on other tabs of this application.
      </p>
    </div>
    <div id="tabs" style="padding-left:10px; font-size:12px;">
      <ul>
        <li><a href="#default">Default Policies</a></li>
<?php
    foreach ($instruments as $instrument_name) {
        print <<<HERE
        <li><a href="#{$instrument_name}">{$instrument_name}</a></li>

HERE;
    }
?>
        <li><a href="#usage">Storage Usage</a></li>
      </ul>

      <div id="default" >
        <div style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >

          <table style="width:1024px;"><tbody>

              <tr>
                <td class="table_hdr" >Storage</td>
                <td class="table_hdr" >Parameter</td>
                <td class="table_hdr" >Definition</td>
                <td class="table_hdr" >Units</td>
                <td class="table_hdr" >Value</td>
                <td class="table_hdr" >Default</td>
<?php if ($can_edit) { ?>
                <td class="table_hdr" >Actions</td>
                <td class="table_hdr" >Comments</td>
<?php } ?>
              </tr>

              <tr>
                <td class="table_cell table_cell_left" rowspan="2" valign="top" style="white-space: nowrap;" >SHORT-TERM</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >CTIME override</td>
                <td class="table_cell" valign="top" >
                  The parameter (if set) will tell the File Manager to ignore actual file creation timestamps of files which are older
                  than the specified value of the parameter and to use the value of the parameter when calculating the expiration
                  dates of the files. Note that this won't affect the real timestamp of the file in a file system neither in
                  the Experiment Portal Web interface.
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >yyyy-mm-dd</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >
<?php if ($can_edit) { ?>
                  <input type="text" name="default_short_ctime" class="default" value="<?php echo $default_short_ctime_time_str; ?>" size="11" />
<?php } else { ?>
                  <span><?php echo $default_short_ctime_time_str; ?><span/>
<?php } ?>
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >actual file creation time</td>
<?php if ($can_edit) { ?>
                <td class="table_cell" valign="top" style="white-space: nowrap;" ><button class="default" name="default_short_ctime" >Save</button></td>
                <td class="table_cell table_cell_right" valign="top" ><span id="comment_default_short_ctime"></span></td>
<?php } ?>
              </tr>
              <tr>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >retention</td>
                <td class="table_cell" valign="top" >
                  The parameter determines the maximum duration (retention) of stay for a file in this type of storage.
                  Files will be considered <b>expired</b> on a day which is the specified (or default) number of months
                  after they (the files) are created (unless the CTIME override is used). An empty value of the parameter
                  means that no specific limit is imposed on a duration of time the files can be kept in this type of storage.
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >number of months</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >
<?php if ($can_edit) { ?>
                  <input type="text" name="default_short_retention" class="default" value="<?php echo $default_short_retention; ?>" size="2" />
<?php } else { ?>
                  <span><?php echo $default_short_retention; ?><span/>
<?php } ?>
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >no limit</td>
<?php if ($can_edit) { ?>
                <td class="table_cell" valign="top" style="white-space: nowrap;" ><button class="default"  name="default_short_retention" >Save</button></td>
                <td class="table_cell table_cell_right" valign="top" ><span id="comment_default_short_retention"></span></td>
<?php } ?>
              </tr>

              <tr>
                <td class="table_cell table_cell_left" rowspan="3" valign="top" style="white-space: nowrap;" >MEDIUM-TERM</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >CTIME override</td>
                <td class="table_cell" valign="top" >
                  The parameter (if set) will tell the File Manager to ignore actual file creation timestamps of files which are older
                  than the specified value of the parameter and to use the value of the parameter when calculating the expiration
                  dates of the files. Note that this won't affect the real timestamp of the file in a file system neither in
                  the Experiment Portal Web interface.
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >yyyy-mm-dd</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >
<?php if ($can_edit) { ?>
                  <input type="text" name="default_medium_ctime" class="default" value="<?php echo $default_medium_ctime_time_str; ?>" size="11" />
<?php } else { ?>
                  <span><?php echo $default_medium_ctime_time_str; ?><span/>
<?php } ?>
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >actual file creation time</td>
<?php if ($can_edit) { ?>
                <td class="table_cell" valign="top" style="white-space: nowrap;" ><button class="default"  name="default_medium_ctime" >Save</button></td>
                <td class="table_cell table_cell_right" valign="top" ><span id="comment_default_medium_ctime"></span></td>
<?php } ?>
              </tr>
              <tr>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >retention</td>
                <td class="table_cell" valign="top" >
                  The parameter determines the maximum duration (retention) of stay for a file in this type of storage.
                  Files will be considered <b>expired</b> on a day which is the specified (or default) number of months
                  after they (the files) are created (unless the CTIME override is used). An empty value of the parameter
                  means that no specific limit is imposed on a duration of time the files can be kept in this type of storage.
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >number of months</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >
<?php if ($can_edit) { ?>
                  <input type="text" name="default_medium_retention" class="default" value="<?php echo $default_medium_retention; ?>" size="2" />
<?php } else { ?>
                  <span><?php echo $default_medium_retention; ?><span/>
<?php } ?>
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >no limit</td>
<?php if ($can_edit) { ?>
                <td class="table_cell" valign="top" style="white-space: nowrap;" ><button class="default"  name="default_medium_retention" >Save</button></td>
                <td class="table_cell table_cell_right" valign="top" ><span id="comment_default_medium_retention"></span></td>
<?php } ?>
              </tr>
              <tr>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >quota</td>
                <td class="table_cell" valign="top" >
                  The parameter determines the storage quota allocated for each experiment. An empty value of the parameter means that
                  no specific limit is imposed on the amount of data which can be kept in this type of storage.
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >GB</td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >
<?php if ($can_edit) { ?>
                  <input type="text" name="default_medium_quota" class="default" value="<?php echo $default_medium_quota; ?>" size="4" />
<?php } else { ?>
                  <span><?php echo $default_medium_quota; ?><span/>
<?php } ?>
                </td>
                <td class="table_cell" valign="top" style="white-space: nowrap;" >no limit</td>
<?php if ($can_edit) { ?>
                <td class="table_cell" valign="top" style="white-space: nowrap;" ><button class="default"  name="default_medium_quota" >Save</button></td>
                <td class="table_cell table_cell_right" valign="top" ><span id="comment_default_medium_quota"></span></td>
<?php } ?>
              </tr>
            </tbody></table>
        </div>
      </div>
<?php

    $deadline_time = LusiTime::parse('2012-10-02');
    $usage = array ();

    foreach( $instruments as $instrument_name ) {

        $usage[$instrument_name] = array (

            'SHORT-TERM' => array (
                'usage'    => array ('files' => 0, 'tb' => 0.0),
                'expired'  => array ('files' => 0, 'tb' => 0.0)),

            'MEDIUM-TERM' => array (
                'usage'    => array ('files' => 0, 'tb' => 0.0 ),
                'expired'  => array ('files' => 0, 'tb' => 0.0 ),
                'quota_tb' => 0.0)
        );

        print <<<HERE

      <div id="{$instrument_name}">
        <div style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <table style="margin-left:10px;"><tbody>
            <tr>
              <td class="table_hdr" rowspan=2 >Experiment</td>
              <td class="table_hdr" rowspan=2 >Id</td>
              <td class="table_hdr" rowspan=2 >#runs</td>
              <td class="table_hdr" colspan=5 align="center" >SHORT-TERM</td>
              <td class="table_hdr" colspan=6 align="center" >MEDIUM-TERM</td>

HERE;
        if( $can_edit )
            print <<<HERE
              <td class="table_hdr" rowspan=2 >Actions</td>
              <td class="table_hdr" rowspan=2 >Comments</td>

HERE;
        print <<<HERE
            </tr>
            <tr>
              <td class="table_hdr" align="right" >total [GB]</td>
              <td class="table_hdr" align="right" >expired [GB]</td>
              <td class="table_hdr" >CTIME override</td>
              <td class="table_hdr" align="right" >retention</td>
              <td class="table_hdr" >expiration</td>
              <td class="table_hdr" align="right" >total [GB]</td>
              <td class="table_hdr" align="right" >quota [GB]</td>
              <td class="table_hdr" >used [%]</td>
              <td class="table_hdr" >CTIME override</td>
              <td class="table_hdr" align="right" >retention</td>
              <td class="table_hdr" >expiration</td>
            </tr>

HERE;

        foreach ($experiments_by_instrument[$instrument_name] as $experiment) {

            $short_quota_ctime           = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-CTIME' );
            $short_quota_ctime_time_str  = is_null( $short_quota_ctime ) ? '' : $short_quota_ctime->value();

            $short_retention             = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-RETENTION' );
            $short_retention_months      = is_null( $short_retention ) ? 0 : intval($short_retention->value());
            $short_retention_str         = $short_retention_months ? $short_retention_months : '';

            $medium_quota                = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA' );
            $medium_quota_gb             = is_null( $medium_quota ) ? 0 : intval($medium_quota->value());
            $medium_quota_str            = $medium_quota_gb ? $medium_quota_gb : '';

            $medium_quota_ctime          = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA-CTIME' );
            $medium_quota_ctime_time_str = is_null( $medium_quota_ctime ) ? null : $medium_quota_ctime->value();

            $medium_retention            = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA-RETENTION' );
            $medium_retention_months     = is_null( $medium_retention ) ? 0 : intval($medium_retention->value());
            $medium_retention_str        = $medium_retention_months ? $medium_retention_months : '';

            $medium_usage_files = 0;
            $medium_usage_gb    = 0;
            foreach( $config->medium_store_files($experiment->id()) as $file ) {
                $medium_usage_files += 1;
                $medium_usage_gb += $file['irods_size'];
            }
            $medium_usage_gb          = intval( 1.0 * $medium_usage_gb / BYTES_IN_GB );
            $medium_usage_percent_str = '';
            if( $medium_quota_gb || $default_medium_quota ) {
                $medium_usage_percent     = 100. * $medium_usage_gb / ( $medium_quota_gb ? $medium_quota_gb : $default_medium_quota );
                $medium_usage_percent_str = '<span style="color:'.($medium_usage_percent >= 100.0 ? 'red' : 'black').'">'.sprintf("%4.1f", $medium_usage_percent ).'</span>';
            }
            $medium_usage_gb_str   = $medium_usage_gb ? $medium_usage_gb : '';

            print <<<HERE
            <tr>
              <td class="table_cell"                                       ><a target="_blank" href="../portal/index.php?exper_id={$experiment->id()}&app=datafiles:files" title="open Web Portal of the Experiment in new window/tab">{$experiment->name()}</a></td>
              <td class="table_cell"                      align="right"    >{$experiment->id()}</td>
              <td class="table_cell table_cell_highlight" align="right" id="num_runs_{$experiment->id()}"           >Loading...</td>
              <td class="table_cell table_cell_highlight" align="right" id="short_term_size_gb_{$experiment->id()}" >Loading...</td>
              <td class="table_cell table_cell_highlight" align="right" id="short_term_expired_size_gb_{$experiment->id()}" >Loading...</td>

HERE;
            if ($can_edit) {
                print <<<HERE
              <td class="table_cell"                                       ><input type="text" class="experiment" name="short_ctime_{$experiment->id()}"      value="{$short_quota_ctime_time_str}"  size="11" /></td>
              <td class="table_cell"                                       ><input type="text" class="experiment" name="short_retention_{$experiment->id()}"  value="{$short_retention_str}"         size="2" style="text-align:right" /></td>
              <td class="table_cell"                                    id="short_term_expiration_{$experiment->id()}" >Loading...</td>
              <td class="table_cell table_cell_highlight" align="right" id="medium_usage_gb_{$experiment->id()}" >Loading...</td>
              <td class="table_cell "                                      ><input type="text" class="experiment" name="medium_quota_{$experiment->id()}"     value="{$medium_quota_str}"            size="4" style="text-align:right" /></td>
              <td class="table_cell table_cell_highlight" align="right"    >{$medium_usage_percent_str}</td>
              <td class="table_cell "                                      ><input type="text" class="experiment" name="medium_ctime_{$experiment->id()}"     value="{$medium_quota_ctime_time_str}" size="11" /></td>
              <td class="table_cell "                                      ><input type="text" class="experiment" name="medium_retention_{$experiment->id()}" value="{$medium_retention_str}"        size="2" style="text-align:right" /></td>
              <td class="table_cell"                                    id="medium_term_expiration_{$experiment->id()}" >Loading...</td>
              <td class="table_cell " style="white-space: nowrap;"         ><button class="experiment" name="{$experiment->id()}" >Save</button><button class="delete_expired" name="{$experiment->id()}" >Delete Expired Files</button><a class="notify_pi" name="notify_pi_of_exper_id_{$experiment->id()}" >Notify PI</a></td>
              <td class="table_cell table_cell_right"                      ><span id="comment_{$experiment->id()}"}></span></td>

HERE;
            } else {
                print <<<HERE
              <td class="table_cell"                                       >{$short_quota_ctime_time_str}</td>
              <td class="table_cell"                                       >{$short_retention_str}</td>
              <td class="table_cell"                                    id="short_term_expiration_{$experiment->id()}" >Loading...</td>
              <td class="table_cell table_cell_highlight" align="right"    >{$medium_usage_gb_str}</td>
              <td class="table_cell table_cell_highlight" align="right"    >{$medium_quota_str}</td>
              <td class="table_cell table_cell_highlight" align="right"    >{$medium_usage_percent_str}</td>
              <td class="table_cell table_cell_highlight"                  >{$medium_quota_ctime_time_str}</td>
              <td class="table_cell table_cell_highlight table_cell_right" >{$medium_retention_str}</td>
              <td class="table_cell"                                    id="medium_term_expiration_{$experiment->id()}" >Loading...</td>

HERE;
            }
            print <<<HERE
            </tr>

HERE;
            // Update total stats
            //
            $usage[$instrument_name]['MEDIUM-TERM']['usage']['files'] += $medium_usage_files;
            $usage[$instrument_name]['MEDIUM-TERM']['usage']['tb']    += $medium_usage_gb / 1024.0;
            $usage[$instrument_name]['MEDIUM-TERM']['quota_tb']       += ( $medium_quota_gb ? $medium_quota_gb : $default_medium_quota ) / 1024.0;
        }
        print <<<HERE
          </tbody></table>
        </div>
      </div>

HERE;
    }
?>
      <div id="usage" >
        <div style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <table><tbody>
            <tr>
              <td class="table_hdr" rowspan="3" align="center" >Instrument</td>
              <td class="table_hdr" colspan="9" align="center" >SHORT-TERM</td>
              <td class="table_hdr" colspan="3" align="center" >MEDIUM-TERM</td>
            </tr>
            <tr>
              <td class="table_hdr" colspan="2" align="center" >Usage</td>
              <td class="table_hdr" colspan="7" align="center" >Expiration [TB]</td>
              <td class="table_hdr" colspan="2" align="center" >Usage</td>
              <td class="table_hdr" rowspan="2" align="right"  >Quota [TB]</td>
            </tr>
            <tr>
              <td class="table_hdr"             align="right"  >files</td>
              <td class="table_hdr"             align="right"  >TB</td>
              <td class="table_hdr"             align="right"  >2014-12-01</td>
              <td class="table_hdr"             align="right"  >2015-01-01</td>
              <td class="table_hdr"             align="right"  >2015-02-01</td>
              <td class="table_hdr"             align="right"  >2015-03-01</td>
              <td class="table_hdr"             align="right"  >2015-04-01</td>
              <td class="table_hdr"             align="right"  >2015-05-01</td>
              <td class="table_hdr"             align="right"  >2015-06-01</td>
              <td class="table_hdr"             align="right"  >files</td>
              <td class="table_hdr"             align="right"  >TB</td>
            </tr>
<?php

    $total_short_usage_files    = 0;
    $total_short_usage_tb       = 0.0;

    $total_medium_quota_tb      = 0.0;
    $total_medium_usage_files   = 0;
    $total_medium_usage_tb      = 0.0;


    foreach ($instruments as $instrument_name) {

        $short_usage_files    = $usage[$instrument_name]['SHORT-TERM']['usage'    ]['files'];
        $short_usage_tb       = $usage[$instrument_name]['SHORT-TERM']['usage'    ]['tb'];

        $medium_usage_files   = $usage[$instrument_name]['MEDIUM-TERM']['usage'   ]['files'];
        $medium_usage_tb      = $usage[$instrument_name]['MEDIUM-TERM']['usage'   ]['tb'];
        $medium_quota_tb      = $usage[$instrument_name]['MEDIUM-TERM']['quota_tb'];

        $total_short_usage_files    += $short_usage_files;
        $total_short_usage_tb       += $short_usage_tb;

        $total_medium_usage_files   += $medium_usage_files;
        $total_medium_usage_tb      += $medium_usage_tb;

        $short_usage_files    = $short_usage_files    ?        $short_usage_files    : '';
        $short_usage_tb       = $short_usage_tb       ? intval($short_usage_tb)      : '';

        $medium_usage_files   = $medium_usage_files   ?        $medium_usage_files   : '';
        $medium_usage_tb      = $medium_usage_tb      ? intval($medium_usage_tb)     : '';

        print <<<HERE
            <tr>
              <td class="table_cell table_cell_left"  align="left"  >{$instrument_name}</td>
              <td class="table_cell "                 align="right" id="short_usage_files_{$instrument_name}"           >Loading</td>
              <td class="table_cell highlighted "     align="right" id="short_usage_tb_{$instrument_name}"              >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2014-12-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2015-01-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2015-02-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2015-03-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2015-04-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2015-05-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="short_expired_tb_2015-06-02_{$instrument_name}" >Loading...</td>
              <td class="table_cell "                 align="right" id="medium_usage_files_{$instrument_name}"          >Loading...</td>
              <td class="table_cell highlighted "     align="right" id="medium_usage_tb_{$instrument_name}"             >Loading...</td>
              <td class="table_cell table_cell_right" align="right" id="medium_quota_tb_{$instrument_name}"             >Loading...</td>
            </tr>
HERE;
    }
    print <<<HERE
            <tr style="background-color:#f0f0f0;">
              <td class="table_cell table_cell_left  table_cell_bottom"             align="left"  >Total:</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_usage_files"           >Loading</td>
              <td class="table_cell                  table_cell_bottom highlighted" align="right" id="total_short_usage_tb"              >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-06-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-07-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-08-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-09-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-10-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-11-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_short_expired_tb_2014-12-02" >Loading...</td>
              <td class="table_cell                  table_cell_bottom"             align="right" id="total_medium_usage_files"          >Loading...</td>
              <td class="table_cell                  table_cell_bottom highlighted" align="right" id="total_medium_usage_tb"             >Loading...</td>
              <td class="table_cell table_cell_right table_cell_bottom"             align="right" id="total_medium_quota_tb"             >Loading...</td>
            </tr>
HERE;
?>
          </tbody></table>
        </div>
      </div>
<?php
    $regdb->commit();
    $logbook->commit();
    $config->commit();

} catch( Exception $e ) { print $e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'; }

?>
    </div>
  </div>
</body>
</html>
