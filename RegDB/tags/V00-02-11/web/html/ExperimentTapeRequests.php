<!-- The script is for viewing and managing outstanding file restore requests.
  -->

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>

<title>is</title>
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

input.data_path {
  padding-left: 2px;
  padding-right: 2px;
}

</style>

<script type="text/javascript">

$(function() {

	$('button').button().click(function() {

		var exper_id = this.id;

		$('#save-form input[name="exper_id"]').val(exper_id);
		$('#save-form input[name="data_path"]').val($('input.data_path[name="'+exper_id+'"]').val());

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
				$('button#'+exper_id).button('disable');
				$('#comment_'+exper_id).text('saved');
			},
			error: function() {
				alert('failed to submit the request');
			},
			dataType: 'json'
		});

	});
	$('input.data_path').keyup(function() {
		var name = this.name;
		if( '' == this.value ) $('button#'+name).button('disable');
		else                   $('button#'+name).button('enable');
	});
});

</script>

</head>

<body>

  <div style="padding-left:20px; padding-right:20px;">

    <h2>View/Modify Experiment Data Path</h2>
    <p>This tool is mean to view and (if your account has sufficient privileges) to modify
    values of the experiments' parameter 'DATA_PATH'. This parameter is normally set
    when registering an experiment in the <a target="_blank" href="../regdb/">Experiment Registry Database</a>.
    To avoid inconcsistencies between the database and the experiments' data file systems it's not advised
    to modify values of the parameter for those experiments which have already taken data.
    </p>
    <div style="padding-left:20px;">
      <form id="save-form" enctype="multipart/form-data" action="../regdb/ws/SetExperimentdataPath.php" method="post">
        <input type="hidden" name="exper_id" value="" />
        <input type="hidden" name="data_path" value="" />
      </form>

<?php
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

try {
	$logbook = LogBook::instance();
	$logbook->begin();

	$regdb = RegDB::instance();
	$regdb->begin();

	$can_modify = RegDBAuth::instance()->canEdit();

	foreach( $logbook->instruments() as $instrument ) {

		if( $instrument->is_location()) continue;

		print <<<HERE
<table><tbody>
  <tr>
    <td class="table_hdr">Instrument</td>
    <td class="table_hdr">Experiment</td>
    <td class="table_hdr">Id</td>
    <td class="table_hdr">#runs</td>
    <td class="table_hdr">Data Path</td>
HERE;
        if( $can_modify )
			print <<<HERE
    <td class="table_hdr">Actions</td>
    <td class="table_hdr">Comments</td>
HERE;
		print <<<HERE
  </tr>
HERE;

		foreach( $logbook->experiments_for_instrument( $instrument->name()) as $experiment ) {

			if( $experiment->is_facility()) continue;

			$num_runs = $experiment->num_runs();
			$num_runs_str = 0 == $num_runs ? '' : $num_runs;

			$data_path = $experiment->regdb_experiment()->find_param_by_name( 'DATA_PATH' );
			$data_path_str = is_null( $data_path ) ? '' : $data_path->value();

			print <<<HERE
  <tr>
    <td class="table_cell">{$experiment->instrument()->name()}</td>
    <td class="table_cell"><a target="_blank" href="../portal/index.php?exper_id={$experiment->id()}" title="open Web Portal of the Experiment in new window/tab">{$experiment->name()}</a></td>
    <td class="table_cell">{$experiment->id()}</td>
    <td class="table_cell">{$num_runs_str}</td>
HERE;
			if( $can_modify ) {
				print <<<HERE
    <td class="table_cell"><input type="text" class="data_path" name="{$experiment->id()}" value="{$data_path_str}" /></td>
    <td class="table_cell"><button id="{$experiment->id()}" disabled="disabled">Save</button></td>
    <td class="table_cell table_cell_right"><span id="comment_{$experiment->id()}"}></span></td>
HERE;
			} else {
				print <<<HERE
	<td class="table_cell table_cell_right">{$data_path_str}</td>
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
	
} catch( LogBookException $e ) { print $e->toHtml(); }
  catch( RegDBException   $e ) { print $e->toHtml(); }

?>

      </div>
    </div>
  </body>
</html>