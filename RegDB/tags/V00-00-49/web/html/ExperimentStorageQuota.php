<!-- The script for reporting and optionally modifying the MEDIUM-TERM-DISK-QUOTA
     parameter of experiments. -->

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>

<title>Report and modify Data Path parameter for known experiments</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="css/common.css" rel="Stylesheet" />

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
  font-size: 75%;
}
td.table_cell {
  border:solid 1px #d0d0d0;
  border-top: none;
  border-left: none;
  padding: 2px 8px 2px 8px;
  font-family: Arial, sans-serif;
  font-size: 75%;
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

input.quota {
  padding-left: 2px;
  padding-right: 2px;
}

</style>

<script type="text/javascript">

$(function() {

	$('button').button().click(function() {

		var exper_id = this.name;
        var button = $(this);

        button.button('disable');

		$('#save-form input[name="exper_id"]' ).val(exper_id);
		$('#save-form input[name="short_ctime"]'     ).val($('input[name="short_ctime_'     +exper_id+'"]').val());
		$('#save-form input[name="short_retention"]' ).val($('input[name="short_retention_' +exper_id+'"]').val());
		$('#save-form input[name="medium_quota"]'    ).val($('input[name="medium_quota_'    +exper_id+'"]').val());
		$('#save-form input[name="medium_ctime"]'    ).val($('input[name="medium_ctime_'    +exper_id+'"]').val());
		$('#save-form input[name="medium_retention"]').val($('input[name="medium_retention_'+exper_id+'"]').val());

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		$('#save-form').ajaxSubmit({
			success: function(data) {
				if( data.Status != 'success' ) {
					alert( data.Message );
					return;
				}
			},
			complete: function() {
				button.button('disable');
				$('#comment_'+exper_id).text('saved');
			},
			error: function() {
				alert('failed to submit the request');
				button.button('enable');
			},
			dataType: 'json'
		});

	});
	$('input.quota').keyup(function() {
		var exper_id = parseInt(this.name.substr(this.name.lastIndexOf('_')+1));
        var button = $('button[name="'+exper_id+'"]');
		button.button('enable');
	});
});

</script>

</head>

<body>

  <div style="padding-left:20px; padding-right:20px;">

    <h2>View/Modify Experiment Data Retention Policies</h2>
    <div style="width:1024px; padding-left:20px;">

      <p>This application is mean to view and (if your account has sufficient privileges) to modify
         parameters of the data retention policy. Different experiments may get different values:
      </p>

      <table style="margin-left:10px;"><tbody>

        <tr>
          <td class="table_hdr" >Storage</td>
          <td class="table_hdr" style="white-space: nowrap;" >Parameter</td>
          <td class="table_hdr" style="white-space: nowrap;" >Experiment Registry</td>
          <td class="table_hdr" >Units</td>
          <td class="table_hdr"  style="white-space: nowrap;" >Default</td>
          <td class="table_hdr" >Meaning</td>
        </tr>

        <tr>
          <td class="table_cell table_cell_left" rowspan="2" valign="top" style="white-space: nowrap;" >SHORT-TERM</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >CTIME override</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >SHORT-TERM-DISK-QUOTA-CTIME</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >yyyy-mm-dd</td>
          <td class="table_cell" valign="top" >actual file creation time</td>
          <td class="table_cell table_cell_right" valign="top" >
            The parameter allows to override the actual file creation timestamps of files which are older
            than the specified value of the parameter. This won't affect the real timestamp of the file in a file
            system neither in the Experiment Portal Web interface. It's just meant to be used to adjust data retention policy
            for the corresponding experiment.
          </td>
        </tr>
        <tr>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >retention</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >SHORT-TERM-DISK-QUOTA-RETENTION</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >positive number</td>
          <td class="table_cell" valign="top" >3</td>
          <td class="table_cell table_cell_right" valign="top" >
            The parameter determines the maximum duration (retention) of stay for a file in this type of storage.
            The file is supposed to be expired on a day which comes the specified (or default) number of months
            after the file gets created (unless the CTIME override is used).
          </td>
        </tr>

        <tr>
          <td class="table_cell table_cell_left" rowspan="3" valign="top" style="white-space: nowrap;" >MEDIUM-TERM</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >CTIME override</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >MEDIUM-TERM-DISK-QUOTA-CTIME</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >yyyy-mm-dd</td>
          <td class="table_cell" valign="top" >actual file creation time</td>
          <td class="table_cell table_cell_right" valign="top" >
            The parameter allows to override the actual file creation timestamps of files which are older
            than the specified value of the parameter. This won't affect the real timestamp of the file in a file
            system neither in the Experiment Portal Web interface. It's just meant to be used to adjust data retention policy
            for the corresponding experiment.
          </td>
        </tr>
        <tr>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >retention</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >MEDIUM-TERM-DISK-QUOTA-RETENTION</td>
          <td class="table_cell" valign="top" style="white-space: nowrap;" >positive number</td>
          <td class="table_cell" valign="top" >3</td>
          <td class="table_cell table_cell_right" valign="top" >
            The parameter determines the maximum duration (retention) of stay for a file in this type of storage.
            The file is supposed to be expired on a day which comes the specified (or default) number of months
            after the file gets created (unless the CTIME override is used).
          </td>
        </tr>
        <tr>
          <td class="table_cell " valign="top" style="white-space: nowrap;" >quota</td>
          <td class="table_cell " valign="top" style="white-space: nowrap;" >MEDIUM-TERM-DISK-QUOTA</td>
          <td class="table_cell " valign="top" style="white-space: nowrap;" >GB</td>
          <td class="table_cell " valign="top" >no limit</td>
          <td class="table_cell  table_cell_right" valign="top" >
            The parameter determines the storage quota allocated for the experiment. The default value of the parameter means that
            the corresponding experiment is not limited in the amount of data which can be kept in this type of storage.
            duration (retention) of stay for a file in this type of storage.
          </td>
        </tr>
      </tbody></table>

    </div>

    <div style="padding-left:20px;margin-top:20px;">
      <form id="save-form" enctype="multipart/form-data" action="../regdb/SetExperimentStorageQuota.php" method="post">
        <input type="hidden" name="exper_id"         value="" />
        <input type="hidden" name="short_ctime"      value="" />
        <input type="hidden" name="short_retention"  value="" />
        <input type="hidden" name="medium_quota"     value="" />
        <input type="hidden" name="medium_ctime"     value="" />
        <input type="hidden" name="medium_retention" value="" />
      </form>

      <div>
        <p style="width:1024px;">The table shown below will also show some current statistics on the number of runs
        taken by experiments and the current amount of data in the MEDIUM-TERM storage
        allocated for each experiment. Please, reload this page to refresh the statistics.
        </p>
      </div>

<?php
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

use DataPortal\Config;
use DataPortal\DataPortalException;

define( 'BYTES_IN_GB', 1024 * 1024 * 1024 );

try {
	$logbook = LogBook::instance();
	$logbook->begin();

	$regdb = RegDB::instance();
	$regdb->begin();

    $config = Config::instance();
    $config->begin();

	$can_modify = RegDBAuth::instance()->canEdit();

    $first_instrument = true;
	foreach( $logbook->instruments() as $instrument ) {

        $margin = '20px';
        if( $first_instrument ) {
            $first_instrument = false;
            $margin = '0px';
        } 
		if( $instrument->is_location()) continue;

		print <<<HERE
<table style="margin-top:{$margin}; margin-left:10px;"><tbody>
  <tr>
    <td class="table_hdr" rowspan=2 >Instrument</td>
    <td class="table_hdr" rowspan=2 >Experiment</td>
    <td class="table_hdr" rowspan=2 >Id</td>
    <td class="table_hdr" rowspan=2 >#runs</td>
    <td class="table_hdr" colspan=2 align="center" >SHORT-TERM</td>
    <td class="table_hdr" colspan=5 align="center" >MEDIUM-TERM</td>
HERE;
        if( $can_modify )
			print <<<HERE
    <td class="table_hdr" rowspan=2 >Actions</td>
    <td class="table_hdr" rowspan=2 >Comments</td>
HERE;
		print <<<HERE
  </tr>
  <tr>
    <td class="table_hdr" >CTIME override</td>
    <td class="table_hdr" align="right" >retention</td>
    <td class="table_hdr" align="right" >data [GB]</td>
    <td class="table_hdr" align="right" >quota [GB]</td>
    <td class="table_hdr" >used [%]</td>
    <td class="table_hdr" >CTIME override</td>
    <td class="table_hdr" align="right" >retention</td>
  </tr>
HERE;

		foreach( $logbook->experiments_for_instrument( $instrument->name()) as $experiment ) {

			if( $experiment->is_facility()) continue;

			$num_runs = $experiment->num_runs();
			$num_runs_str = 0 == $num_runs ? '' : $num_runs;

            $short_quota_ctime          = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-CTIME' );
            $short_quota_ctime_time     = is_null( $short_quota_ctime ) ? null : LusiTime::parse($short_quota_ctime->value());
			$short_quota_ctime_time_str = is_null( $short_quota_ctime ) ? '' : $short_quota_ctime->value(); // is_null( $short_quota_ctime_time ) ? '' : $short_quota_ctime_time->toStringDay();

            $short_retention        = $experiment->regdb_experiment()->find_param_by_name( 'SHORT-TERM-DISK-QUOTA-RETENTION' );
            $short_retention_months = is_null( $short_retention ) ? 0 : intval($short_retention->value());
			$short_retention_str    = $short_retention_months ? $short_retention_months : '';

			$medium_quota     = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA' );
            $medium_quota_gb  = is_null( $medium_quota ) ? 0 : intval($medium_quota->value());
			$medium_quota_str = $medium_quota_gb ? $medium_quota_gb : '';

            $medium_quota_ctime          = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA-CTIME' );
            $medium_quota_ctime_time     = is_null( $medium_quota_ctime ) ? null : LusiTime::parse($medium_quota_ctime->value());
			$medium_quota_ctime_time_str = is_null( $medium_quota_ctime_time ) ? '' : $medium_quota_ctime_time->toStringShort();

            $medium_retention        = $experiment->regdb_experiment()->find_param_by_name( 'MEDIUM-TERM-DISK-QUOTA-RETENTION' );
            $medium_retention_months = is_null( $medium_retention ) ? 0 : intval($medium_retention->value());
			$medium_retention_str    = $medium_retention_months ? $medium_retention_months : '';

            $medium_usage_gb = 0;
            foreach( $config->medium_store_files($experiment->id()) as $file ) {
                $medium_usage_gb += $file['irods_size'];
            }
            $medium_usage_gb          = intval( 1.0 * $medium_usage_gb / BYTES_IN_GB );
            $medium_usage_percent_str = '';
            if( $medium_quota_gb ) {
                $medium_usage_percent     = 100. * $medium_usage_gb / $medium_quota_gb;
                $medium_usage_percent_str = '<span style="color:'.($medium_usage_percent >= 100.0 ? 'red' : 'black').'">'.sprintf("%4.1f", $medium_usage_percent ).'</span>';
            }
            $medium_usage_gb_str = $medium_usage_gb ? $medium_usage_gb : '';

			print <<<HERE
  <tr>
    <td class="table_cell">{$experiment->instrument()->name()}</td>
    <td class="table_cell"                                    ><a target="_blank" href="../portal/index.php?exper_id={$experiment->id()}&app=datafiles:files" title="open Web Portal of the Experiment in new window/tab">{$experiment->name()}</a></td>
    <td class="table_cell"                      align="right" >{$experiment->id()}</td>
    <td class="table_cell table_cell_highlight" align="right" >{$num_runs_str}</td>
HERE;
			if( $can_modify ) {
				print <<<HERE
    <td class="table_cell"                                    ><input type="text" class="quota" name="short_ctime_{$experiment->id()}"      value="{$short_quota_ctime_time_str}" size="6" /></td>
    <td class="table_cell"                                    ><input type="text" class="quota" name="short_retention_{$experiment->id()}"  value="{$short_retention_str}"        size="1" style="text-align:right" /></td>
	<td class="table_cell table_cell_highlight" align="right" >{$medium_usage_gb_str}</td>
    <td class="table_cell "                                   ><input type="text" class="quota" name="medium_quota_{$experiment->id()}"     value="{$medium_quota_str}"            size="2" style="text-align:right" /></td>
	<td class="table_cell table_cell_highlight" align="right" >{$medium_usage_percent_str}</td>
    <td class="table_cell "                                   ><input type="text" class="quota" name="medium_ctime_{$experiment->id()}"     value="{$medium_quota_ctime_time_str}" size="6" /></td>
    <td class="table_cell "                                   ><input type="text" class="quota" name="medium_retention_{$experiment->id()}" value="{$medium_retention_str}"        size="1" style="text-align:right" /></td>
    <td class="table_cell "                                   ><button name="{$experiment->id()}" disabled="disabled">Save</button></td>
    <td class="table_cell table_cell_right"                   ><span id="comment_{$experiment->id()}"}></span></td>
HERE;
			} else {
				print <<<HERE
	<td class="table_cell"                  >{$short_quota_ctime_time_str}</td>
	<td class="table_cell"                  >{$short_quota_retention_str}</td>
	<td class="table_cell table_cell_highlight" align="right"    >{$medium_usage_gb_str}</td>
	<td class="table_cell table_cell_highlight" align="right"    >{$medium_usage_percent_str}</td>
	<td class="table_cell table_cell_highlight" align="right"    >{$medium_quota_str}</td>
	<td class="table_cell table_cell_highlight"                  >{$medium_quota_ctime_time_str}</td>
	<td class="table_cell table_cell_highlight table_cell_right" >{$medium_quota_retention_str}</td>
HERE;
			}
			print <<<HERE
  </tr>
HERE;
		}
		print <<<HERE
</tbody><table>
HERE;
	}
	$regdb->commit();
	$logbook->commit();
    $config->commit();

} catch( DataPortalException $e ) { print $e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'; }
  catch( LusiTimeException   $e ) { print $e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'; }
  catch( LusiTimeException   $e ) { print $e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'; }
  catch( RegDBException      $e ) { print $e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'; }

?>

      </div>
    </div>
  </body>
</html>