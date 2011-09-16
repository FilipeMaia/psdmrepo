<?php

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'regdb/regdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use DataPortal\DataPortal;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

use RegDB\RegDB;
use RegDB\RegDBException;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;

use AuthDB\AuthDB;
use AuthDB\AuthDBException;


/* Let a user to select an experiment first if no valid experiment
 * identifier is supplied to the script.
 */
if( !isset( $_GET['exper_id'] )) {
	header("Location: select_experiment.php");
	exit;
}
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' ) die( 'no valid experiment identifier provided to the script' );

if( isset( $_GET['page1'] )) {
	$page1 = trim( $_GET['page1'] );
	if( isset( $_GET['page2'] )) {
		$page2 = trim( $_GET['page2'] );
	}
}

try {

	// Connect to databases
	//
	$auth_svc = AuthDB::instance();
	$auth_svc->begin();

	$regdb = new RegDB();
	$regdb->begin();

	$logbook = new LogBook();
	$logbook->begin();

	$logbook_experiment = $logbook->find_experiment_by_id( $exper_id );
	if( is_null( $logbook_experiment )) die( 'invalid experiment identifier provided to the script' );

	$experiment = $logbook_experiment->regdb_experiment();
    $instrument = $experiment->instrument();

    /* Get stats for e-log
     */
    $min_run = null;
    $max_run = null;
    $logbook_runs = $logbook_experiment->runs();
    foreach( $logbook_runs as $r ) {
  		$run = $r->num();
  		if( is_null( $min_run )) {
  			$min_run = $run;
  			$max_run = $run;
  		} else {
    		if( $run < $min_run ) $min_run = $run;
  			if( $run > $max_run ) $max_run = $run;
  		}
    }
    $logbook_shifts = $logbook_experiment->shifts();
    
    /* Get the stats for data files
     */
    $num_runs = 0;
    $xtc_num_files  = 0;
    $xtc_size       = 0.0;
    $xtc_local_copy = 0;
    $xtc_archived   = 0;

    $hdf5_num_files  = 0;
    $hdf5_size       = 0.0;
    $hdf5_local_copy = 0;
    $hdf5_archived   = 0;

    if( $experiment->begin_time()->greaterOrEqual( LusiTime::now())) {
        ;
    } else {

        $range = FileMgrIrodsWs::max_run_range( $instrument->name(), $experiment->name(), array('xtc','hdf5'));

        $num_runs      = $range['total'];
        $range_of_runs = $range['min'].'-'.$range['max'];

        $xtc_runs = null;
        FileMgrIrodsWs::runs( $xtc_runs, $instrument->name(), $experiment->name(), 'xtc', $range_of_runs );
        foreach( $xtc_runs as $run ) {
            $unique_files = array();  // per this run
            $files = $run->files;
            foreach( $files as $file ) {
                if( !array_key_exists( $file->name, $unique_files )) {
                    $unique_files[$file->name] = $run->run;
                    $xtc_num_files++;
                    $xtc_size += $file->size / (1024.0 * 1024.0 * 1024.0);
                }
                if( $file->resource == 'hpss-resc'   ) $xtc_archived++;
                if( $file->resource == 'lustre-resc' ) $xtc_local_copy++;
            }
        }
        $xtc_size_str = sprintf( "%.0f", $xtc_size );

        $hdf5_runs = null;
        FileMgrIrodsWs::runs( $hdf5_runs, $instrument->name(), $experiment->name(), 'hdf5', $range_of_runs );
        foreach( $hdf5_runs as $run ) {
            $unique_files = array();  // per this run
            $files = $run->files;
            foreach( $files as $file ) {
                if( !array_key_exists( $file->name, $unique_files )) {
                    $unique_files[$file->name] = $run->run;
                    $hdf5_num_files++;
                    $hdf5_size += $file->size / (1024.0 * 1024.0 * 1024.0);
                }
                if( $file->resource == 'hpss-resc'   ) $hdf5_archived++;
                if( $file->resource == 'lustre-resc' ) $hdf5_local_copy++;
            }
        }
        $hdf5_size_str = sprintf( "%.0f", $hdf5_size );
    }

    $range_of_runs = '0-0';
    if( !$experiment->begin_time()->greaterOrEqual( LusiTime::now())) {
        $range = FileMgrIrodsWs::max_run_range( $instrument->name(), $experiment->name(), array( 'xtc', 'hdf5' ));
        $range_of_runs = $range['min'].'-'.$range['max'];
    }

    /* Get the stats for HDF translation
     */
    $latest_only = true;
	$hdf5_requests = FileMgrIfaceCtrlWs::experiment_requests (
		$instrument->name(),
		$experiment->name(),
		$latest_only
	);

    $hdf5_num_runs_complete = 0;
    $hdf5_num_runs_failed = 0;
    $hdf5_num_runs_wait = 0;
    $hdf5_num_runs_translate = 0;
	$hdf5_num_runs_unknown = 0;

   	foreach( $hdf5_requests as $req ) {
		switch( $req->status ) {

			case 'Initial_Entry':
			case 'Waiting_Translation':
				$hdf5_num_runs_wait++;
				break;

			case 'Being_Translated':
				$hdf5_num_runs_translate++;
				break;

			case 'Empty_Fileset':
			case 'H5Dir_Error':
			case 'Translation_Error':
		    case 'Archive_Error':
		    	$hdf5_num_runs_failed++;
		    	break;

		    case 'Complete':
		    	$hdf5_num_runs_complete++;
		    	break;

		    default:
		    	$hdf5_num_runs_unknown++;
		}
   	}
?>




<!------------------- Document Begins Here ------------------------->

<?php
    DataPortal::begin( "Data Portal of Experiment" );
?>



<!------------------- Page-specific Styles ------------------------->

<link type="text/css" href="css/portal.css" rel="Stylesheet" />

<style type="text/css">

  .not4print {
  }

  .elog-message-group-day-toggler,
  .elog-message-toggler,
  .elog-attachment-toggler {
    background-color:#ffffff;
    border:1px solid #c0c0c0;
  }

  .elog-message-group-day,
  .elog-attachment-description {
    font-weight:bold;
    margin-left:10px;
  }

  .elog-message-time,
  .elog-message-author {
    font-weight:bold;
    margin-left:10px;
  }

  .elog-message-subject {
    color:maroon;
    margin-left:10px;
  }

  .elog-attachment {
    padding:5px;
  }

  .elog-message-group-day-header {
    margin-top:5px;
    padding:5px;
    background-color:#e0e0e0;
  }
  .elog-message-group-day-header:hover {
    background-color:#d0d0d0;
    cursor:hand;
  }

  .elog-message-group-day-container {
    padding-top:5px;
    padding-left:20px;
    background-color:#f0f0f0;
  }

  .elog-message-header {
    padding:5px;
  }

  .elog-message-header:hover {
    background-color:#e0e0e0;
    cursor:hand;
  }

  .elog-message-container {
    padding-top:5px;
    padding-left:20px;
  }

  .elog-attachment-container {
    margin-left:10px;
    padding-top:5px;
    padding-left:20px;
  }

  .elog-message-body {
    padding:10px;
    background-color:#ffffff;
    border-bottom:5px solid e8e8e8;
  }

  .elog-message-attachments {
    padding:10px;
    background-color:#ffffff;
    border-bottom:5px solid e8e8e8;
  }

  .elog-attachment-info {
    margin-left:5px;
  }

  .elog-message-tags {
    padding:10px;
    background-color:#ddffff;
  }

  .elog-message-group-day-hidden,
  .elog-message-hidden,
  .elog-attachment-hidden {
    display: none;
  }

  .elog-message-group-day-visible,
  .elog-message-visible,
  .elog-attachment-visible {
    display: block;
  }

  .small-button-container {
    font-size:80%;
  }
</style>

<!----------------------------------------------------------------->



<?php
    DataPortal::scripts( "init" );
?>


<!------------------ Page-specific JavaScript ---------------------->

<script type="text/javascript" src="js/ELog.js"></script>
<script type="text/javascript">


/* -----------------------------------------
 *             GLOBAL VARIABLES
 * -----------------------------------------
 */
elog.author_account  = '<?=$auth_svc->authName()?>';
elog.exper_id        = '<?=$exper_id?>';
elog.experiment_name = '<?=$experiment->name()?>';
elog.instrument_name = '<?=$experiment->instrument()->name()?>';
elog.range_of_runs   = '<?=$range_of_runs?>';
elog.min_run         = <?=(is_null($min_run)?'null':$min_run)?>;
elog.max_run         = <?=(is_null($max_run)?'null':$max_run)?>;
<?php
	foreach( $logbook_runs as $run ) echo "elog.runs[{$run->num()}]={$run->id()};\n";
	foreach( $logbook_shifts as $shift ) echo "elog.shifts['{$shift->begin_time()->toStringShort()}']={$shift->id()};\n";
?>

/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */

function init() {

	$('#tabs').tabs();

	experiment_init();
	elog.init();
	files_init();
	hdf5_init();

	/* Open the initial tab if explicitly requested. Otherwise the first
	 * tab will be shown.
	 */
<?php
	if( isset($page1)) {
		echo "\t$('#tabs').tabs('select', '#tabs-{$page1}');\n";
		if( isset($page2))
			echo "\t$('#tabs-{$page1}-subtabs').tabs('select', '#tabs-{$page1}-{$page2}');\n";
	}
?>
}

/* ----------------------------------------
 *             TAB: EXPERIMENT
 * ----------------------------------------
 */
function experiment_init() {

	$('#button-toggle-group').click(
		function() {
			if( $('#group-members').hasClass   ( 'group-members-hidden' ) ) {
				$('#group-members').removeClass( 'group-members-hidden' )
				                   .addClass   ( 'group-members-visible' );
				$('#button-toggle-group').removeClass( 'ui-icon-triangle-1-e' )
				                         .addClass   ( 'ui-icon-triangle-1-s' );
			} else {
				$('#group-members').removeClass( 'group-members-visible' )
				                   .addClass   ( 'group-members-hidden' );
				$('#button-toggle-group').removeClass( 'ui-icon-triangle-1-s' )
				                         .addClass   ( 'ui-icon-triangle-1-e' );
			}
		}
	);
	$( '#button-select-experiment' ).button();
	$( '#button-select-experiment' ).click(
		function() {
			window.location = 'select_experiment.php';
		}
	);
}

/* --------------------------------------
 *             TAB: FILES
 * --------------------------------------
 */
function files_init() {

	$('#tabs-files-subtabs').tabs();

	$('#button-files-filter-reset').button();
	$('#button-files-filter-reset').click(
		function() {
			reset_files_filter();
			search_files();
		}
	);
	$('#button-files-filter-apply').button();
	$('#button-files-filter-apply').click(
		function() {
			search_files();
		}
	);
	$('#button-files-filter-import').button();
	$('#button-files-filter-import').click(
		function() {
			search_files( true );
		}
	);
	search_files();
}

function reset_files_filter() {

	$('#files-search-filter :input:radio[name=runs]' ).val( ['all'] );
	$('#files-search-filter :input:text[name=runs_range]' ).val( elog.range_of_runs );
	$('#files-search-filter :input:radio[name=archived]' ).val( ['yes_or_no'] );
	$('#files-search-filter :input:radio[name=local]' ).val( ['yes_or_no'] );
	$('#files-search-filter :input:checkbox[name=xtc]' ).val( ['XTC'] );
    $('#files-search-filter :input:checkbox[name=hdf5]' ).val( ['HDF5'] );
}

function search_files( import_format ) {

	var params = { exper_id: elog.exper_id };

	if( $('#files-search-filter :input:radio[name=runs]:checked' ).val() != 'all' ) {
		var runs = $('#files-search-filter :input:text[name=runs_range]' ).val();
		if( runs != elog.range_of_runs )	params.runs = runs;
	}

	var archived = $('#files-search-filter :input:radio[name=archived]:checked' ).val();
	if( archived != 'yes_or_no' ) params.archived = ( archived == 'no' ? 0 : 1 );

	var local = $('#files-search-filter :input:radio[name=local]:checked' ).val();
	if( local != 'yes_or_no' ) params.local = ( local == 'no' ? 0 : 1 );

	var checked_types = [
		$('#files-search-filter :input:checkbox[name=xtc]:checked' ).val(),
	    $('#files-search-filter :input:checkbox[name=hdf5]:checked' ).val()
	];

	var types = null;
	for( idx in checked_types ) {
		if( checked_types[idx] == null ) continue;
		types = ( types == null ? '' : types + ',' );
		types += checked_types[idx];
	}
	if( types != null ) params.types = types;

	if( import_format ) params.import_format = null;

	$( '#files-search-result' ).html( 'Searching...' );
	$.get(
	   	'SearchFiles.php',
	   	params,
	   	function( data ) {
			$( '#files-search-result' ).html( data );
	    }
	);
}

/* --------------------------------------
 *             TAB: HDF5
 * --------------------------------------
 */
 function hdf5_init() {

    $('#tabs-translate-subtabs').tabs();

	$('#button-translate-filter-reset').button();
	$('#button-translate-filter-reset').click(
		function() {
			reset_translate_filter();
			search_translate_requests();
		}
	);
	$('#button-translate-filter-apply').button();
	$('#button-translate-filter-apply').click(
		function() {
			search_translate_requests();
		}
	);
	search_translate_requests();
}

function reset_translate_filter() {

	$('#translate-search-filter :input:radio[name=runs]' ).val( ['all'] );
	$('#translate-search-filter :input:text[name=runs_range]' ).val( elog.range_of_runs );
	$('#translate-search-filter :input:radio[name=translated]' ).val( ['yes_or_no'] );
}

function search_translate_requests() {

	var params = {
		exper_id: elog.exper_id,
		show_files: 1
	};

	if( $('#translate-search-filter :input:radio[name=runs]:checked' ).val() != 'all' ) {
		var runs = $('#translate-search-filter :input:text[name=runs_range]' ).val();
		if( runs != elog.range_of_runs )	params.runs = runs;
	}

	var translated = $('#translate-search-filter :input:radio[name=translated]:checked' ).val();
	if( translated != 'yes_or_no' ) params.translated = ( translated == 'no' ? 0 : 1 );

	$('#translate-search-result').html( 'Searching...' );
	$.get(
	   	'SearchRequests.php',
	   	params,
	   	function( data ) {
			$('#translate-search-result').html( data );
			$('#translate-search-result .translate').button();
			$('#translate-search-result .translate').click(
				function(e) {
					e.preventDefault();
					$.get(
					   	'NewRequest.php',
					   	{ exper_id: elog.exper_id, runnum: $(this).val() },
					   	function( data ) {
						   	if( data.ResultSet.Status == 'success' )
						   		search_translate_requests();
						   	else
						   		alert( 'The request has failed because of: '+data.ResultSet.Reason );
					   	}
				   	);
				}
			);
			$('#translate-search-result .escalate').button();
			$('#translate-search-result .escalate').click(
				function(e) {
					e.preventDefault();
					$.get(
						'EscalateRequestPriority.php',
						{ exper_id: elog.exper_id, id: $(this).val() },
						function( data ) {
						   	if( data.ResultSet.Status == 'success' ) {
						   		$('#translate-search-result #priority_'+data.ResultSet.Result.id).text(data.ResultSet.Result.priority);
						   	} else
								alert( 'The request has failed because of: '+data.ResultSet.Reason );
						}
					);
				}
			);
			$('#translate-search-result .delete').button();
			$('#translate-search-result .delete').click(
				function(e) {
					e.preventDefault();
					$.get(
						'DeleteRequest.php',
						{ id: $(this).val() },
						function( data ) {
						   	if( data.ResultSet.Status == 'success' )
						   		search_translate_requests();
						   	else
								alert( 'The request has failed because of: '+data.ResultSet.Reason );
						}
					);
				}
			);
			/* ----------------------------------
			 * THIS IS HOW IT WILL WORK FOR MS IE
			 * ----------------------------------

			 	function(e) {
					var event = e || window.event;
					var target = event.target || event.srcElement;
					$.get(
						'DeleteRequest.php',
						{ id: e.originalEvent.target.value },
						...
			*/
	    }
	);
}

/* ----------------------------------------------
 *             CONTEXT MANAGEMENT
 * ----------------------------------------------
 */
var current_tab = 'tabs-experiment';

function set_current_tab( tab ) {
	current_tab = tab;
}

/* ----------------------------------------------
 *             UTILITY FUNCTIONS
 * ----------------------------------------------
 */
function show_email( user, addr ) {
	$('#popupdialogs').html( '<p>'+addr+'</p>' );
	$('#popupdialogs').dialog({
		modal:  true,
		title:  'e-mail: '+user
	});
}

function display_path( file ) {
	$('#popupdialogs').html( '<p>'+file+'</p>' );
	$('#popupdialogs').dialog({
		modal:  true,
		title:  'file path'
	});
}

function pdf( context ) {
	if( context == 'translate-manage' ) {
		var url = 'Requests2pdf.php?exper_id='+elog.exper_id+'&show_files';
		var winRef = window.open( url, 'Translation Requests' );
	}
}

function printer_friendly() {
	var el = document.getElementById( current_tab );
	if (el) {
		var html = document.getElementById(current_tab).innerHTML;
		var pfcopy = window.open("about:blank");
		pfcopy.document.write('<html xmlns="http://www.w3.org/1999/xhtml">');
		pfcopy.document.write('<head><meta http-equiv="Content-Type" content="text/html; charset=windows-1252" />');
		pfcopy.document.write('<link rel="stylesheet" type="text/css" href="css/default.css" />');
		pfcopy.document.write('<link type="text/css" href="css/portal.css" rel="Stylesheet" />');
		pfcopy.document.write('<style type="text/css"> .not4print { display:none; }	</style>');
		pfcopy.document.write('<title>Data Portal of Experiment: '+elog.instrument_name+' / '+elog.experiment_name+'</title></head><body><div class="maintext">');
		pfcopy.document.write(html);
		pfcopy.document.write("</div></body></html>");
		pfcopy.document.close();
	}
}

</script>
<!----------------------------------------------------------------->



<?php
    DataPortal::body(
    	'Data Portal of Experiment:',
    	'<a href="select_experiment.php" title="Switch to another experiment">'.$experiment->instrument()->name().'&nbsp;/&nbsp;'.$experiment->name().'</a>',
    	'experiment'
   	);
?>




<!------------------ Page-specific Document Body ------------------->

<?php

	$tabs = array();

	$decorated_experiment_status  = DataPortal::decorated_experiment_status_UP   ( $experiment );
	$decorated_experiment_contact = DataPortal::decorated_experiment_contact_info( $experiment );
	$experiment_group_members     = "<table><tbody>\n";
    foreach( $experiment->group_members() as $m ) {
    	$uid   = $m['uid'];
    	$gecos = $m['gecos'];
        $experiment_group_members .= "<tr><td><b>{$uid}</b></td><td>{$gecos}</td></tr>\n";
	}
	$experiment_group_members .= "</tbody></table>\n";
    $tabs_experiment =<<<HERE
	  <button id="button-select-experiment" class="not4print">Select another experiment</button>
	  <br>
	  <br>
      <table>
        <tbody>
          <tr>
            <td class="table_cell_left">Id</td>
            <td class="table_cell_right">{$experiment->id()}</td>
          </tr>
          <tr>
            <td class="table_cell_left">Status</td>
            <td class="table_cell_right">{$decorated_experiment_status}</td>
          </tr>
          <tr>
            <td class="table_cell_left">Begin</td>
            <td class="table_cell_right">{$experiment->begin_time()->toStringShort()}</td>
          </tr>
          <tr>
            <td class="table_cell_left">End</td>
            <td class="table_cell_right">{$experiment->end_time()->toStringShort()}</td>
          </tr>
          <tr>
            <td class="table_cell_left">Description</td>
            <td class="table_cell_right"><pre style="background-color:#e0e0e0; padding:0.5em;">{$experiment->description()}</pre></td>
          </tr>
          <tr>
            <td class="table_cell_left">Contact</td>
            <td class="table_cell_right">{$decorated_experiment_contact}</td>
          </tr>
          <tr>
            <td class="table_cell_left">Leader</td>
            <td class="table_cell_right">{$experiment->leader_Account()}</td>
          </tr>
          <tr>
            <td class="table_cell_left table_cell_bottom" valign="top">POSIX Group</td>
            <td class="table_cell_right table_cell_bottom">
              <table cellspacing=0 cellpadding=0><tbody>
                <tr>
                  <td valign="top">{$experiment->POSIX_gid()}</td>
                  <td>&nbsp;</td>
                  <td>
                    <span id="button-toggle-group" class="ui-icon ui-icon-triangle-1-e" style="border:1px solid #c0c0c0;" title="click to see/hide the list of members"></span>
                    <div id="group-members" class="group-members-hidden">{$experiment_group_members}</div>
                  </td>
                </tr>
              </tbody></table>
            </td>
          </tr>
        </tbody>
      </table>\n
HERE;
	array_push(
		$tabs,
    	array(
    		'name' => 'Experiment',
    		'id'   => 'tabs-experiment',
    		'html' => $tabs_experiment,
    	    'callback' => 'set_current_tab("tabs-experiment")'
    	)
    );

    /*
     * [ e-log ]
     */
    $tabs_elog_subtabs = array();

    $tabs_elog_recent =<<<HERE
    <div style="padding:20px; padding-top:10px;">
      <div style="float:left;">
        <div style="font-weight:bold;">Last threads to display:</div>
        <div id="elog-live-range-selector" style="margin-top:4px;">
          <input type="radio" id="elog-live-range-20"   name="range" value="20"  checked="checked" /><label for="elog-live-range-20"   >20</label>
          <input type="radio" id="elog-live-range-100"  name="range" value="100"                   /><label for="elog-live-range-100"  >100</label>
          <input type="radio" id="elog-live-range-day"  name="range" value="day"                   /><label for="elog-live-range-day"  >24 hrs</label>
          <input type="radio" id="elog-live-range-week" name="range" value="week"                  /><label for="elog-live-range-week" >7 days</label>
          <input type="radio" id="elog-live-range-all"  name="range" value="all"                   /><label for="elog-live-range-all"  >everything</label>
        </div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="font-weight:bold;">Show runs:</div>
        <div style="margin-top:4px;">
          <div id="elog-live-runs-selector" style="float:left;">
            <input type="radio" id="elog-live-runs-on"  name="runs" value="on"  checked="checked" /><label for="elog-live-runs-on"  >On</label>
            <input type="radio" id="elog-live-runs-off" name="runs" value="off"                   /><label for="elog-live-runs-off" >Off</label>
          </div>
        </div>
      </div>
      <div style="float:right;">
        <div style="font-weight:bold;">Autorefresh:</div>
        <div style="margin-top:4px;">
          <div id="elog-live-refresh-selector" style="float:left;">
            <input type="radio" id="elog-live-refresh-on"  name="refresh" value="on"  checked="checked" /><label for="elog-live-refresh-on"  >On</label>
            <input type="radio" id="elog-live-refresh-off" name="refresh" value="off"                   /><label for="elog-live-refresh-off" >Off</label>
          </div>
          <div style="float:left; margin-left:10px;">
            Interval:
            <select>
              <option>2</option>
              <option>5</option>
              <option>10</option>
            </select>
            seconds
          </div>
          <div style="clear:both;"></div>
        </div>
      </div>
      <div style="clear:both;"></div>
      <div style="margin-top:10px; padding-bottom:20px;">
        <button id="elog-live-expand"     title="Step-by-step expantion of the message tree. Each click will reveal more detail.">Expand++</button>
        <button id="elog-live-collapse"   title="Each click will collapse the tree to the previous level of detail.">Collapse--</button>
        <button id="elog-live-viewattach" title="view attachments of expanded messages">View Attachments</button>
        <button id="elog-live-hideattach" title="hide attachments of expanded messages">Hide Attachments</button>
      </div>
    </div>
    \n
HERE;
	$attachments_info = array();
	array_push(
		$attachments_info,
		array(
			'id'          => 15347,
			'description' => 'Image #1',
			'mime-type'   => 'image/jpeg',
			'size'        => 348953,
			'preview_url' => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/preview/15347',
			'url'         => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/15347/preview'
	));
	array_push(
		$attachments_info,
		array(
			'id'          => 14347,
			'description' => 'file',
			'mime-type'   => 'image/png',
			'size'        => 85653,
			'preview_url' => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/preview/14347',
			'url'         => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/14347/preview'
	));
	array_push(
		$attachments_info,
		array(
			'id'          => 16340,
			'description' => 'Something added later',
			'mime-type'   => 'image/jpeg',
			'size'        => 12345,
			'preview_url' => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/preview/16340',
			'url'         => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/16340/preview'
	));
	array_push(
		$attachments_info,
		array(
			'id'          => 16350,
			'description' => 'Image #2',
			'mime-type'   => 'image/jpeg',
			'size'        => 234565,
			'preview_url' => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/preview/16350',
			'url'         => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/16350/preview'
	));
	array_push(
		$attachments_info,
		array(
			'id'          => 16357,
			'description' => 'Chlorella Virus',
			'mime-type'   => 'image/jpeg',
			'size'        => 3456433,
			'preview_url' => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/preview/16357',
			'url'         => 'https://pswww.slac.stanford.edu/apps-dev/logbook/attachments/16357/preview'
	));
	function day2name( $idx, $day ) {
		switch( $idx ) {
		case 10 : return 'today';
		case  9 : return 'yesterday';
		}
		return $day;
	}
	for( $i = 10; $i >= 1; $i-- ) {
    	$day = sprintf( "%02d-Feb-2011", $i );
    	$day_name = day2name( $i, $day );
    	$messages = '';
    	for( $s=0; $s<$i; $s++ ) {

    		$time = '08:32:'.sprintf("%02d", $s);

    		// TODO: replace with the unique message ID or its high resolution
    	    //       timestamp: <seconds>-<nanoseconds>. Need this because this
    	    //       is going to be used as part of HTML element id.
    		$timestamp = $day.'-0832'.sprintf("%02d", $s);

    		$attachments =<<<HERE
<div style="float:right;" class="small-button-container"><button class="elog-live-attachments-add">add</button></div>
HERE;
    		foreach( $attachments_info as $a ) {

    			$id          = $a['id'];
    			$description = $a['description'];
    			$mime        = $a['mime-type'];
    			$size        = $a['size'];
    			$preview_url = $a['preview_url'];
    			$url         = $a['url'];

    			$timestamp_and_id = $timestamp.'-'.$id;

    			$attachments .=<<< HERE
<div style="float:left;" class="elog-attachment">
  <div style="float:left;"><span class="ui-icon ui-icon-triangle-1-e elog-attachment-toggler" id="elog-attachment-toggler-{$timestamp_and_id}" onclick="elog.live_toggle_attachment('{$timestamp_and_id}');"></span></div>
  <div style="float:left;" class="elog-attachment-description"><a class="link" href="{$preview_url}">{$description}</a></div>
  <div style="float:left; margin-left:10px;" class="elog-attachment-info">( type: {$mime}: {$size} )</div>
  <div style="clear:both;"></div>
  <div class="elog-attachment-container elog-attachment-hidden" id="elog-attachment-container-{$timestamp_and_id}"><a href="{$url}"><img src="{$preview_url}" /></a></div>
</div>\n
HERE;
    		}
   			$attachments .=<<<HERE
<div style="clear:both;"></div>
HERE;

    		$messages .=<<<HERE
<div style="padding-right:10px; padding-bottom:10px;">
  <div style="float:left;" class="elog-message-header" onclick="elog.live_toggle_message('{$timestamp}');">
    <div style="float:left;"><span class="ui-icon ui-icon-triangle-1-e elog-message-toggler" id="elog-message-toggler-{$timestamp}"></span></div>
    <div style="float:left;" class="elog-message-time">{$time}</div>
    <div style="float:left;" class="elog-message-author">gapon</div>
    <div style="float:left; margin-left:10px;" class="elog-message-subject">Here be the message itself, its attachments...</div>
    <div style="clear:both;"></div>
  </div>
  <div style="float:left; margin-left:10px;" class="small-button-container"><button class="elog-live-message-reply">reply</button></div>
  <div style="clear:both;"></div>
  <div style="padding-left:10px; padding-left:30px;" class="elog-message-container elog-message-hidden" id="elog-message-container-{$timestamp}">
    <div class="elog-message-body">
      <div style="float:left;">
        <pre style="font-size:12px;">
# xpppython scan #
# normalization monitor: XPP:IOC:USERPV:SB2:IPM:SUM
# readings (expect for monitor) are normalized
#number of shots per point: 120
#** Scanning  motorpv: XPP:USR:R39:MMS:01
#   number of points: 41
#   number of shots per point: 120
                |      diode3.ch0     |       ipm2.sum      |
point   position|   avg        err    |   avg        err    |
    1  -104.0700|+2.781e-03 +6.700e-04|+8.196e-01 +5.888e-02|
    2  -103.9950|+3.955e-03 +9.049e-04|+8.197e-01 +6.544e-02|
    3  -103.9200|+4.194e-03 +7.955e-04|+9.499e-01 +7.051e-02|
    4  -103.8450|+7.074e-03 +2.119e-03|+6.189e-01 +6.579e-02|
    5  -103.7700|+8.826e-02 +1.533e-02|+8.599e-01 +6.681e-02|
    6  -103.6950|+4.440e-01 +7.184e-02|+8.739e-01 +6.789e-02|
    7  -103.6200|+4.500e-01 +6.318e-02|+9.693e-01 +6.916e-02|
    8  -103.5450|+4.584e-01 +1.354e-01|+5.743e-01 +6.312e-02|
    9  -103.4700|+4.470e-01 +6.583e-02|+9.089e-01 +6.533e-02|
   10  -103.3950|+4.440e-01 +8.266e-02|+8.242e-01 +7.145e-02|
   11  -103.3200|+1.499e-01 +3.052e-02|+7.664e-01 +6.722e-02|
   12  -103.2450|+2.191e-01 +5.351e-02|+6.585e-01 +6.364e-02|
   13  -103.1700|+2.534e-01 +4.786e-02|+8.087e-01 +6.774e-02|
   14  -103.0950|+2.680e-01 +3.786e-02|+9.332e-01 +6.394e-02|
   15  -103.0200|+2.790e-01 +6.232e-02|+6.982e-01 +6.436e-02|
   16  -102.9450|+2.873e-01 +5.580e-02|+7.303e-01 +5.994e-02|
   17  -102.8700|+2.965e-01 +5.610e-02|+7.545e-01 +6.162e-02|
   18  -102.7950|+2.853e-01 +4.115e-02|+9.340e-01 +6.505e-02|
   19  -102.7200|+2.824e-01 +4.259e-02|+9.428e-01 +6.921e-02|
   20  -102.6450|+2.779e-01 +5.505e-02|+7.617e-01 +6.618e-02|
   21  -102.5700|+2.598e-01 +5.989e-02|+6.890e-01 +6.514e-02|
   22  -102.4950|+2.312e-01 +7.259e-02|+5.808e-01 +6.742e-02|
   23  -102.4200|+1.602e-01 +2.945e-02|+7.544e-01 +5.953e-02|
   24  -102.3450|+3.880e-01 +7.955e-02|+6.947e-01 +5.836e-02|
   25  -102.2700|+4.644e-01 +1.647e-01|+4.766e-01 +5.528e-02|
   26  -102.1950|+4.647e-01 +9.423e-02|+7.181e-01 +6.189e-02|
   27  -102.1200|+4.547e-01 +8.850e-02|+7.652e-01 +6.606e-02|
   28  -102.0450|+4.584e-01 +6.523e-02|+8.699e-01 +5.888e-02|
   29  -101.9700|+3.931e-01 +5.324e-02|+9.201e-01 +6.097e-02|
   30  -101.8950|+4.246e-03 +9.852e-04|+7.856e-01 +6.613e-02|
   31  -101.8200|+3.439e-03 +9.452e-04|+6.773e-01 +6.071e-02|
   32  -101.7450|+4.539e-03 +1.270e-03|+6.773e-01 +6.453e-02|
   33  -101.6700|+4.139e-03 +9.677e-04|+7.660e-01 +6.583e-02|
   34  -101.5950|+7.502e-03 +4.604e-03|+3.406e-01 +5.558e-02|
   35  -101.5200|+4.217e-03 +1.130e-03|+7.165e-01 +6.932e-02|
   36  -101.4450|+3.469e-03 +9.766e-04|+6.844e-01 +6.505e-02|
   37  -101.3700|+2.491e-03 +5.506e-04|+9.455e-01 +6.528e-02|
   38  -101.2950|+2.638e-03 +5.864e-04|+9.127e-01 +6.382e-02|
   39  -101.2200|+5.222e-03 +1.758e-03|+5.601e-01 +6.300e-02|
        </pre>
      </div>
      <div style="float:right;" class="small-button-container"><button class="elog-live-message-edit">edit</button></div>
      <div style="clear:both;"></div>
    </div>
    <div class="elog-message-attachments">{$attachments}</div>
    <div class="elog-message-tags">
      <div style="float:left;"><b><i>keywords: </i></b>TTT</div>
      <div style="float:right;" class="small-button-container"><button class="elog-live-tags-add">add</button></div>
      <div style="clear:both;"></div>
    </div>
  </div>
</div>
\n
HERE;
    	}
    	$tabs_elog_recent .=<<<HERE
<div class="elog-message-group-day-header" onclick="elog.live_toggle_group_day('{$day}');">
  <div style="float:left;">
    <span class="ui-icon ui-icon-triangle-1-e elog-message-group-day-toggler" id="elog-message-group-day-toggler-{$day}"></span>
  </div>
  <div style="float:left;" class="elog-message-group-day">{$day_name}</div>
  <div style="float:right; margin-right:10px;"><i>1 shift, 23 runs, 34 messages</i></div>
  <div style="clear:both;"></div>
</div>
<div class="elog-message-group-day-container elog-message-group-day-hidden" id="elog-message-group-day-container-{$day}">{$messages}</div>
\n
HERE;
    }
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Recent (Live)',
    		'id'   => 'tabs-elog-recent',
    		'html' => $tabs_elog_recent,
    	    'callback' => 'set_current_tab("tabs-elog-recent")'
    	)
    );
    $known_tags_html = "<option> - known tags - </option>\n";
    foreach( $logbook_experiment->used_tags() as $tag ) {
    	$known_tags_html .= "<option>{$tag}</option>\n";
    }

    $tags_html = '';

    // TODO: Move this parameter upstream and make it available
    //       to JavaScript which would use th elibrary to fill ouut
    //       tag names.
    //
    $num_tags = 3;
    for( $i=0; $i<$num_tags; $i++) {
    	$tags_html .=<<<HERE
name:  <select id="elog-tags-library-{$i}">{$known_tags_html}</select>
       <input type="text" id="elog-tag-name-{$i}"  name="tag_name_{$i}"  value="" size=16 title="type new tag here or select a known one from the left" />
value: <input type="text" id="elog-tag-value-{$i}" name="tag_value_{$i}" value="" size=16 title="put an optional value here" /><br>
HERE;
    }

    $today = date("Y-m-d");
    $now   = "00:00:00";

    // TODO: load these two parameters from the database. Update them from thedatabase
    //       when the form gets reset.
    //
    $shifts_html = '';
    foreach( $logbook_shifts as $shift )
    	$shifts_html .= "<option>{$shift->begin_time()->toStringShort()}</option>\n";

    $tabs_elog_post_extra = array();
    array_push(
		$tabs_elog_post_extra,
    	array(
    		'name' => 'Attachments',
    		'id'   => 'tabs-elog-post-attachments',
    		'html' => <<<HERE
<div id="elog-post-attachments">
  file: <input type="file" name="file2attach_0" onchange="elog.post_add_attachment(this)" />
  description: <input type="text" name="file2attach_0" value="" title="put an optional file description here" /><br>
</div>
HERE
			,
			'callback' => 'set_current_tab("tabs-elog-post-attachments")'
    	)
    );
    array_push(
		$tabs_elog_post_extra,
    	array(
    		'name' => 'Tags',
    		'id'   => 'tabs-elog-post-tags',
    		'html' => <<<HERE
<div id="elog-tags" style="margin-top:4px;">{$tags_html}</div>
HERE
			,
			'callback' => 'set_current_tab("tabs-elog-post-tags")'
    	)
    );
    array_push(
		$tabs_elog_post_extra,
    	array(
    		'name' => 'Context & Post Time',
    		'id'   => 'tabs-elog-post-context',
    		'html' => <<<HERE
<div>
  <div style="float:left;">
    <div style="font-weight:bold;">Context:</div>
    <div id="elog-post-context-selector" style="margin-top:4px;">
      <input type="radio" id="elog-post-context-experiment" name="scope" value="experiment" checked="checked" /><label for="elog-post-context-experiment">experiment</label>
      <input type="radio" id="elog-post-context-shift"      name="scope" value="shift"                        /><label for="elog-post-context-shift"     >shift</label>
      <input type="radio" id="elog-post-context-run"        name="scope" value="run"                          /><label for="elog-post-context-run"       >run</label>
    </div>
  </div>
  <div style="float:left; margin-left:10px;">
    <div style="font-weight:bold;">Shift:</div>
    <div style="margin-top:4px;">
      <select id="elog-post-shift">{$shifts_html}</select>
    </div>
  </div>
  <div style="float:left; margin-left:10px;">
    <div style="font-weight:bold;">Run:</div>
    <div style="margin-top:4px;">
      <input type="text" id="elog-post-runnum" value="{$max_run}" size=4 />
      <span id="elog-post-runnum-error" style="color:red;"></span>
    </div>
  </div>
  <div style="clear:both;"></div>
</div>
<div style="margin-top:20px;">
  <div style="float:left;">
    <div style="font-weight:bold;">Post time:</div>
    <div id="elog-post-relevance-selector" style="margin-top:4px;">
      <input type="radio" id="elog-post-relevance-now"   name="relevance" value="now"   checked="checked" /><label for="elog-post-relevance-now"   title="it will be the actual posting time"      >now</label>
      <input type="radio" id="elog-post-relevance-past"  name="relevance" value="past"                    /><label for="elog-post-relevance-past"  title="use date and time selector on the right" >past</label>
      <input type="radio" id="elog-post-relevance-shift" name="relevance" value="shift"                   /><label for="elog-post-relevance-shift" title="within specified shift"                  >in shift</label>
      <input type="radio" id="elog-post-relevance-run"   name="relevance" value="run"                     /><label for="elog-post-relevance-run"   title="within specified run"                    >in run</label>
    </div>
  </div>
  <div style="float:left; margin-left:10px;">
    <div style="font-weight:bold;">&nbsp;</div>
    <div style="margin-top:4px;">
      <input type="text" id="elog-post-datepicker" value="{$today}" size=11 />
      <input type="text" id="elog-post-time" value="{$now}"  size=8 />
    </div>
  </div>
  <div style="clear:both"></div>
</div>
HERE
			,
			'callback' => 'set_current_tab("tabs-elog-post-context")'
    	)
    );
    $tabs_elog_post_extra = DataPortal::tabs_html( "tabs-elog-post-subtabs", $tabs_elog_post_extra );
    $tabs_elog_post =<<<HERE
    <div style="float:left; margin-right:20px;">
      <form id="elog-form-post" enctype="multipart/form-data" action="/apps-dev/logbook/NewFFEntry4portal.php" method="post">
        <input type="hidden" name="author_account" value="" />
        <input type="hidden" name="id" value="" />
        <input type="hidden" name="run_id" value="" />
        <input type="hidden" name="shift_id" value="" />
        <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />
        <input type="hidden" name="num_tags" value="{$num_tags}" />
        <input type="hidden" name="onsuccess" value="" />
        <input type="hidden" name="relevance_time" value="" />
        <textarea name="message_text" rows="12" cols="80" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>
        <div style="margin-top:10px;">{$tabs_elog_post_extra}</div>
      </form>
    </div>
    <div style="float:left;">
      <div><button id="elog-submit-and-stay">Submit and stay on this page</button></div>
      <div><button id="elog-submit-and-follow">Submit and follow up</button></div>
      <div><button id="elog-reset">Reset form</button></div>
    </div>
    <div style="clear:both;"></div>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Post',
    		'id'   => 'tabs-elog-post',
    		'html' => $tabs_elog_post,
    	    'callback' => 'set_current_tab("tabs-elog-post")'
    	)
    );

    $tabs_elog_search =<<<HERE
      <p>There will be a dialog to search for messages</p>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Search',
    		'id'   => 'tabs-elog-search',
    		'html' => $tabs_elog_search,
    	    'callback' => 'set_current_tab("tabs-elog-search")'
    	)
    );

    $tabs_elog_browse =<<<HERE
      <p>This has to be an extended version of the live display allowing more
      sophisticated ways for browsing the database.</p>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Browse',
    		'id'   => 'tabs-elog-browse',
    		'html' => $tabs_elog_browse,
    	    'callback' => 'set_current_tab("tabs-elog-browse")'
    	)
    );

    $tabs_elog_runs =<<<HERE
      <p>A list of all runs, a selector for a run and runs summary page.</p>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Runs',
    		'id'   => 'tabs-elog-runs',
    		'html' => $tabs_elog_runs,
    	    'callback' => 'set_current_tab("tabs-elog-runs")'
    	)
    );

    $tabs_elog_shifts =<<<HERE
      <p>A list of all shifts, a selector for a shift and the shift goals page.</p>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Shifts',
    		'id'   => 'tabs-elog-shifts',
    		'html' => $tabs_elog_shifts,
    	    'callback' => 'set_current_tab("tabs-elog-shifts")'
    	)
    );

    $tabs_elog_subscribe =<<<HERE
      <p>A dialog for viewing/managing subscriptions for e-mail notifications.</p>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Subscribe',
    		'id'   => 'tabs-elog-subscribe',
    		'html' => $tabs_elog_subscribe,
    	    'callback' => 'set_current_tab("tabs-elog-subscribe")'
    	)
    );

    array_push(
		$tabs,
    	array(
    		'name' => 'e-Log',
    		'id'   => 'tabs-elog',
    		'html' => DataPortal::tabs_html( "tabs-elog-subtabs", $tabs_elog_subtabs ),
    	    'callback' => 'set_current_tab("tabs-elog")'
    	)
    );

    $tabs_files_subtabs = array();
    $tabs_files_4runs =<<<HERE
	  <div>
        <div id="files-search-summary" style="float: left">
          <table><tbody>
            <tr>
              <td class="grid-sect-hdr-first">R u n s</td>
            </tr>
            <tr>
              <td class="grid-key">Number of runs:</td>
              <td class="grid-value">{$num_runs}</td>
            </tr>
            <tr>
              <td class="grid-sect-hdr">X T C</td>
            </tr>
            <tr>
              <td class="grid-key">Number of files:</td>
              <td class="grid-value">{$xtc_num_files}</td>
              <td class="grid-key">Size [GB]:</td>
              <td class="grid-value">{$xtc_size_str}</td>
            </tr>
            <tr>
              <td class="grid-key">Archived to tape:</td>
              <td class="grid-value">{$xtc_archived} / {$xtc_num_files}</td>
              <td class="grid-key">On disk:</td>
              <td class="grid-value">{$xtc_local_copy} / {$xtc_num_files}</td>
            </tr>
             <tr>
             <td class="grid-sect-hdr">H D F 5</td>
            </tr>
            <tr>
              <td class="grid-key">Number of files:</td>
              <td class="grid-value">{$hdf5_num_files}</td>
              <td class="grid-key">Size [GB]:</td>
              <td class="grid-value">{$hdf5_size_str}</td>
            </tr>
            <tr>
              <td class="grid-key">Archived to tape:</td>
              <td class="grid-value">{$hdf5_archived} / {$hdf5_num_files}</td>
              <td class="grid-key">On disk:</td>
              <td class="grid-value">{$hdf5_local_copy} / {$hdf5_num_files}</td>
            </tr>
          </tbody></table>
        </div>
        <div id="files-search-filter" style="float: left">
          <div class="group" style="float: left">
            <div class="selector-hdr">R u n s</div>
            <table><thead>
              <tr>
                <td class="selector-option"><input type="radio" name="runs" value="all" checked="checked"></td>
                <td class="selector-value">all</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="runs" value="range"></td>
                <td class="selector-value"><input type="text" name="runs_range" value="{$range_of_runs}" width=10 title="1,3,5,10-20,200"></td>
              </tr>
            </thead></table>
          </div>
          <div class="group" style="float: left">
            <div class="selector-hdr">A r c h i v e d</div>
            <table><thead>
              <tr>
                <td class="selector-option"><input type="radio" name="archived" value="yes_or_no" checked="checked"></td>
                <td class="selector-value">yes or no</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="archived" value="yes"></td>
                <td class="selector-value">yes</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="archived" value="no"></td>
                <td class="selector-value">no</td>
              </tr>
            </thead></table>
          </div>
          <div class="group" style="float: left">
            <div class="selector-hdr">D i s k</div>
            <table><thead>
              <tr>
                <td class="selector-option"><input type="radio" name="local" value="yes_or_no" checked="checked"></td>
                <td class="selector-value">yes or no</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="local" value="yes"></td>
                <td class="selector-value">yes</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="local" value="no"></td>
                <td class="selector-value">no</td>
              </tr>
            </thead></table>
          </div>
          <div style="clear: both;"></div>
          <div class="group" style="float: left">
            <div class="selector-hdr">T y p e s</div>
            <table><thead>
              <tr>
                <td class="selector-option"><input type="checkbox" name="xtc" value="XTC" checked="checked"></td>
                <td class="selector-value">XTC</td>
              </tr>
              <tr>
                <td class="selector-option"><input class="grid-key" type="checkbox" name="hdf5" value="HDF5" checked="checked"></td>
                <td class="selector-value">HDF5</td>
              </tr>
              </thead></table>
          </div>
          <div style="clear: both;"></div>
          <div style="float: right;">
            <button id="button-files-filter-reset" class="not4print">Reset Filter</button>
            <button id="button-files-filter-apply" class="not4print">Apply Filter</button>
            <button id="button-files-filter-import" class="not4print">Import List</button>
          </div>
          <div style="clear: both;"></div>
        </div>
        <div style="clear: both;"></div>
      </div>
      <div id="files-search-result"></div>\n
HERE;
    array_push(
		$tabs_files_subtabs,
    	array(
    		'name' => 'By Runs',
    		'id'   => 'tabs-files-4runs',
    		'html' => $tabs_files_4runs,
    	    'callback' => 'set_current_tab("tabs-files-4runs")'
    	)
    );
    array_push(
		$tabs,
    	array(
    		'name' => 'Data Files',
    		'id'   => 'tabs-files',
    		'html' => DataPortal::tabs_html( "tabs-files-subtabs", $tabs_files_subtabs ),
    	    'callback' => 'set_current_tab("tabs-files")'
    	)
    );


    $tabs_translate_subtabs = array();
    $tabs_translate_manage =<<<HERE
	  <div>
        <div id="translate-search-summary" style="float:left">
          <table><tbody>
            <tr>
              <td class="grid-sect-hdr-first">R u n s</td>
            </tr>
            <tr>
              <td class="grid-key">Number of runs:</td>
              <td class="grid-value">{$num_runs}</td>
            </tr>
            <tr>
              <td class="grid-sect-hdr">T r a n s l a t i o n</td>
            </tr>
            <tr>
              <td class="grid-key">Complete:</td>
              <td class="grid-value">{$hdf5_num_runs_complete}</td>
            </tr>
            <tr>
              <td class="grid-key">Failed:</td>
              <td class="grid-value">{$hdf5_num_runs_failed}</td>
            </tr>
            <tr>
              <td class="grid-key">Waiting:</td>
              <td class="grid-value">{$hdf5_num_runs_wait}</td>
            </tr>
            <tr>
              <td class="grid-key">Being translated:</td>
              <td class="grid-value">{$hdf5_num_runs_translate}</td>
            </tr>
            <tr>
              <td class="grid-key">Other state:</td>
              <td class="grid-value">{$hdf5_num_runs_unknown}</td>
            </tr>
          </tbody></table>
        </div>
        <div id="translate-search-filter" style="float:left">
          <div class="group" style="float:left">
            <div class="selector-hdr">R u n s</div>
            <table><thead>
              <tr>
                <td class="selector-option"><input type="radio" name="runs" value="all" checked="checked"></td>
                <td class="selector-value">all</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="runs" value="range"></td>
                <td class="selector-value"><input type="text" name="runs_range" value="{$range_of_runs}" width=10 title="1,3,5,10-20,200"></td>
              </tr>
            </thead></table>
          </div>
          <div class="group" style="float:left">
            <div class="selector-hdr">T r a n s l a t e d</div>
            <table><thead>
              <tr>
                <td class="selector-option"><input type="radio" name="translated" value="yes_or_no" checked="checked"></td>
                <td class="selector-value">yes or no</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="translated" value="yes"></td>
                <td class="selector-value">yes</td>
              </tr>
              <tr>
                <td class="selector-option"><input type="radio" name="translated" value="no"></td>
                <td class="selector-value">no</td>
              </tr>
            </thead></table>
          </div>
          <div style="clear:both;"></div>
          <div style="float:right;">
            <button id="button-translate-filter-reset" class="not4print">Reset Filter</button>
            <button id="button-translate-filter-apply" class="not4print">Apply Filter</button>
          </div>
          <div style="clear:both;"></div>
        </div>
        <div style="clear:both;"></div>
      </div>
	  <div id="translate-search-result"></div>\n
HERE;
    array_push(
		$tabs_translate_subtabs,
    	array(
    		'name' => 'Manage',
    		'id'   => 'tabs-translate-manage',
    		'html' => $tabs_translate_manage,
    	    'callback' => 'set_current_tab("tabs-translate-manage")'
    	)
    );
    $tabs_translate_history =<<<HERE
      <p>Here be the list of all translation requests. And there will be a filter
      on the top right side to allow.</p>\n
HERE;
    array_push(
		$tabs_translate_subtabs,
    	array(
    		'name' => 'History of Requests',
    		'id'   => 'tabs-translate-history',
    		'html' => $tabs_translate_history,
    	    'callback' => 'set_current_tab("tabs-translate-history")'
    	)
    );
    array_push(
		$tabs,
    	array(
    		'name' => 'XTC/HDF5 Translation',
    		'id'   => 'tabs-translate',
    		'html' => DataPortal::tabs_html( "tabs-translate-subtabs", $tabs_translate_subtabs ),
    	    'callback' => 'set_current_tab("tabs-translate")'
    	)
    );

    $tabs_account =<<<HERE
      <p>User account information, privileges, POSIX groups, other experiments participation,
      subscriptions, etc.</p>\n
HERE;
   	array_push(
		$tabs,
    	array(
    		'name' => 'My Account',
    		'id'   => 'tabs-account',
    		'html' => $tabs_account,
    	    'callback' => 'set_current_tab("tabs-account")'
    	)
    );


	DataPortal::tabs( "tabs", $tabs );
?>

<!----------------------------------------------------------------->






<?php
    DataPortal::end();
?>
<!--------------------- Document End Here -------------------------->


<?php

} catch( FileMgrException $e ) { print $e->toHtml();
} catch( LogBookException $e ) { print $e->toHtml();
} catch( RegDBException   $e ) { print $e->toHtml();
} catch( LogBookException $e ) { print $e->toHtml();
} catch( AuthDBException  $e ) { print $e->toHtml();
}

?>
