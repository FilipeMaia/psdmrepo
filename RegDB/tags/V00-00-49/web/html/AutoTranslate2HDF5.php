
<?php
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

$tables_html = '';
$experiments2load = '';

try {
	$logbook = LogBook::instance();
	$logbook->begin();

	$regdb = RegDB::instance();
	$regdb->begin();

	$can_modify = RegDBAuth::instance()->canEdit();

	foreach( $logbook->instruments() as $instrument ) {

		if( $instrument->is_location()) continue;

		$tables_html .= <<<HERE
<table><tbody>
  <tr>
    <td class="table_hdr">Instrument</td>
    <td class="table_hdr">Experiment</td>
    <td class="table_hdr">Id</td>
    <td class="table_hdr">#runs</td>
    <td class="table_hdr">#translated</td>
    <td class="table_hdr">Auto-Translate</td>
HERE;
        if( $can_modify )
			$tables_html .= <<<HERE
    <td class="table_hdr">Actions</td>
    <td class="table_hdr">Comments</td>
HERE;
		$tables_html .= <<<HERE
  </tr>
HERE;

		foreach( $logbook->experiments_for_instrument( $instrument->name()) as $experiment ) {

			if( $experiment->is_facility()) continue;

			$num_runs = $experiment->num_runs();
			$num_runs_str = '';
            $loading_comment = '';
            if($num_runs) {
                $num_runs_str = $num_runs;
                $loading_comment = 'Loading...';
                if($experiments2load == '')
                    $experiments2load = "var experiments2load=[{$experiment->id()}";
                else
                    $experiments2load .= ",{$experiment->id()}";
            }
			$autotranslate2hdf5 = $experiment->regdb_experiment()->find_param_by_name( 'AUTO_TRANSLATE_HDF5' );
			$autotranslate2hdf5_str = $autotranslate2hdf5 ? 'checked="checked"' : '';

			$tables_html .= <<<HERE
  <tr>
    <td class="table_cell">{$experiment->instrument()->name()}</td>
    <td class="table_cell"><a target="_blank" href="../portal/index.php?exper_id={$experiment->id()}&app=hdf:manage" title="open Web Portal of the Experiment in new window/tab">{$experiment->name()}</a></td>
    <td class="table_cell">{$experiment->id()}</td>
    <td class="table_cell">{$num_runs_str}</td>
    <td class="table_cell"><span id="num_translated_{$experiment->id()}"}>{$loading_comment}</td>
HERE;
			if( $can_modify ) {
				$tables_html .= <<<HERE
    <td class="table_cell"><input type="checkbox" class="autotranslate2hdf5" name="{$experiment->id()}" value=1 {$autotranslate2hdf5_str} /></td>
    <td class="table_cell"><button id="{$experiment->id()}" disabled="disabled">Save</button></td>
    <td class="table_cell table_cell_right"><span id="comment_{$experiment->id()}"}></span></td>
HERE;
			} else {
				$tables_html .= <<<HERE
    <td class="table_cell table_cell_right"><input type="checkbox" class="autotranslate2hdf5" name="{$experiment->id()}" value=1 disabled="disabled" {$autotranslate2hdf5_str} /></td>
HERE;
			}
			$tables_html .= <<<HERE
  </tr>
HERE;
		}
		$tables_html .= <<<HERE
</tbody><table>
HERE;
	}
    if($experiments2load == '')
        $experiments2load = "var experiments2load=[];\n";
    else
        $experiments2load .= "];\n";

    $regdb->commit();
    $logbook->commit();

} catch( LogBookException $e ) { print $e->toHtml(); }
  catch( RegDBException   $e ) { print $e->toHtml(); }
?>

<!-- The script for reporting and optionally modifying the auto-translation
     option for HDF5 files of experiments. -->

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>

<title>Report and modify automatic translation (XTC to HDF5) parameter for known experiments </title>
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

input.autotranslate2hdf5 {
  padding-left: 2px;
  padding-right: 2px;
}

</style>

<script type="text/javascript">

<?php echo $experiments2load; ?>

function load_hdf5_files(exper_id) {
    $.ajax({
        type: 'GET',
        url: '../portal/SearchFiles.php',
        data: {
            exper_id: exper_id,
            types: 'hdf5'
        },
        success: function(data) {
            if( data.Status != 'success' ) { alert(data.Message); return; }
            var num_files = 0;
            for( var i in data.runs ) {
                num_files += data.runs[i].files.length;
            }
            $('#num_translated_'+exper_id).html(num_files?num_files:'');
        },
        error: function() {	alrt('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    });
}

$(function() {

	$('button').button().click(function() {

		var exper_id = this.id;
        var is_checked = $('input.autotranslate2hdf5[name="'+exper_id+'"]').is(':checked');

        $('button#'+exper_id).button('disable');
        $('#comment_'+exper_id).text('saving...');

        $.ajax({
            type: 'POST',
            url: '../regdb/SetAutoTranslate2HDF5.php',
            data: {
                exper_id: exper_id,
                autotranslate2hdf5: is_checked ? 1 : 0
            },
			success: function(data) {
				if( data.Status != 'success' ) {
					$('#comment_'+exper_id).text(data.Message);
					return;
				}
				$('#comment_'+exper_id).text('saved');
			},
			error: function() {
                $('button#'+exper_id).button('enable');
				$('#comment_'+exper_id).text('failed to submit the request');
			},
			dataType: 'json'
		});

	});
	$('input.autotranslate2hdf5').change(function() {
		var name = this.name;
		$('button#'+name).button('enable');
	});

    // Begin asynchronious loading of the number of HDF5 files for each
    // experiment which had at least one run taken.
    //
    for(var i in experiments2load) {
       var exper_id = experiments2load[i];
       load_hdf5_files(exper_id);
    }
});

</script>

</head>
  <body>
    <div style="padding-left:20px; padding-right:20px;">

      <h2>View/Modify Auto-Translation Option for HDF5</h2>
      <p>This tool is mean to view and (if your account has sufficient privileges) to modify
      values of the experiments' parameter 'AUTO_TRANSLATE_HDF5'. This parameter can be also set
      when registering an experiment in the <a target="_blank" href="../regdb/">Experiment Registry Database</a>.
      The 'HDF5' tab of Web Portal of each experiment also allows to view or modify a value of the parameter
      for the corresponding experiment.
      </p>
      <div style="padding-left:20px;"><?php echo $tables_html; ?></div>
    </div>
  </body>
</html>
