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

	$('#button-toggle-group').button();
	$('#button-toggle-group').click(
		function() {
			if( $('#group-members').hasClass   ( 'group-members-hidden' ) ) {
				$('#group-members').removeClass( 'group-members-hidden' )
				                   .addClass   ( 'group-members-visible' );
				$('#button-toggle-group div').removeClass( 'ui-icon-triangle-1-e' )
				                             .addClass   ( 'ui-icon-triangle-1-s' );
			} else {
				$('#group-members').removeClass( 'group-members-visible' )
				                   .addClass   ( 'group-members-hidden' );
				$('#button-toggle-group div').removeClass( 'ui-icon-triangle-1-s' )
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
                    <button id="button-toggle-group" class="not4print" title="click to see/hide the list of members"><div class="ui-icon ui-icon-triangle-1-s"></div></button>
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
      <p>This is the placeholder for most recent messages monitored by the application.
      The old version of the application needs to be redesigned to avoid various sorts
      of conflicts.</p>\n
HERE;
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
<select id="elog-tags-library-{$i}">{$known_tags_html}</select>
<input type="text" id="elog-tag-name-{$i}"  name="tag_name_{$i}"  value="" size=16 title="type new tag here or select a known one from the left" /> =
<input type="text" id="elog-tag-value-{$i}" name="tag_value_{$i}" value="" size=16 title="put an optional value here" /><br>
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

    $tabs_elog_post =<<<HERE
      <form id="elog-form-post" enctype="multipart/form-data" action="/apps-dev/logbook/NewFFEntry4portal.php" method="post">
        <input type="hidden" name="author_account" value="" />
        <input type="hidden" name="id" value="" />
        <input type="hidden" name="run_id" value="" />
        <input type="hidden" name="shift_id" value="" />
        <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />
        <input type="hidden" name="num_tags" value="{$num_tags}" />
        <input type="hidden" name="onsuccess" value="" />
        <input type="hidden" name="relevance_time" value="" />
        <div>
          <div style="float:left;">
	        <span style="font-weight:bold;">Anchor to:</span>
            <div style="margin-top:4px;"></div>
	        <div id="elog-post-context-selector">
		      <input type="radio" id="elog-post-context-experiment" name="scope" value="experiment" checked="checked" /><label for="elog-post-context-experiment">experiment</label>
		      <input type="radio" id="elog-post-context-shift"      name="scope" value="shift"                        /><label for="elog-post-context-shift"     >shift</label>
		      <input type="radio" id="elog-post-context-run"        name="scope" value="run"                          /><label for="elog-post-context-run"       >run</label>
	        </div>
		  </div>
          <div style="float:left; margin-left:20px;">
            <span style="font-weight:bold;">Shift:</span>
            <div style="margin-top:4px;"></div>
            <select id="elog-post-shift">{$shifts_html}</select>
          </div>
		  <div style="float:left; margin-left:20px;">
            <span style="font-weight:bold;">Run #:</span>
            <div style="margin-top:4px;"></div>
            <input type="text" id="elog-post-runnum" value="{$max_run}" size=4 />
            <span id="elog-post-runnum-error" style="color:red;"></span>
          </div>
          <div style="clear:both;"></div>
        </div>
		<div style="margin-top:20px;">
          <div style="float:left;">
            <span style="font-weight:bold;">Post:</span>
            <div style="margin-top:4px;"></div>
            <div id="elog-post-relevance-selector">
		      <input type="radio" id="elog-post-relevance-now"   name="relevance" value="now"   checked="checked" /><label for="elog-post-relevance-now"   title="it will be the actual posting time"      >current time</label>
		      <input type="radio" id="elog-post-relevance-past"  name="relevance" value="past"                    /><label for="elog-post-relevance-past"  title="use date and time selector on the right" >specific time</label>
		      <input type="radio" id="elog-post-relevance-shift" name="relevance" value="shift"                   /><label for="elog-post-relevance-shift" title="within specified shift"                  >within shift</label>
		      <input type="radio" id="elog-post-relevance-run"   name="relevance" value="run"                     /><label for="elog-post-relevance-run"   title="within specified run"                    >within run</label>
		    </div>
		  </div>
          <div style="float:left; margin-left:20px;">
            <span style="font-weight:bold;">Date & time:</span>
            <div style="margin-top:4px;"></div>
            <input type="text" id="elog-post-datepicker" value="{$today}" size=11 />
            <input type="text" id="elog-post-time" value="{$now}"  size=8 />
          </div>
          <div style="clear:both"></div>
		</div>
        <div style="margin-top:20px;">
          <div style="float:left;">
            <span style="font-weight:bold;">Message:</span>
            <div style="margin-top:4px;"></div>
            <textarea name="message_text" rows="12" cols="54" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>
            <div style="margin-top:20px;">
              <span style="font-weight:bold;">Tags (= values):</span>
              <div id="elog-tags" style="margin-top:4px;">{$tags_html}</div>
            </div>
          </div>
          <div style="float:left; margin-left:40px;">
            <span style="font-weight:bold;">Attachments:</span>
            <div id="elog-post-attachments" style="margin-top:4px;">
              <input type="file" name="file2attach_0" onchange="elog.post_add_attachment(this)" />
              <input type="hidden" name="file2attach_0" value="" title="put an optional file description here" /><br>
            </div>
          </div>
          <div style="clear:both;"></div>
        </div>
      </form>
      <div style="margin-top:40px;">
        <button id="elog-submit">Submit</button>
        <button id="elog-reset">Reset</button>
      </div>\n
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
