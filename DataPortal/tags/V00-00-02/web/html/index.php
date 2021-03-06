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
use LogBook\LogBookAuth;
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
if( isset( $_GET['params'] )) {
	$params = explode( ',', trim( $_GET['params'] ));
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

  /* -----------------------------------
   * - Naming convention for shortcuts -
   * -----------------------------------
   *
   *   el          elog
   *   el-l        .....live
   *   el-l-ms     ..........messages
   *   el-l-m      ..........message
   *   el-l-m-d    ....................messages groupped into a day
   *   el-l-m-re   ....................message reply
   *   el-l-m-re-s ....................submit message reply
   *   el-l-m-re-c ....................cancel message reply
   *   el-l-m-ed   ....................message editing
   *   el-l-m-ed-s ....................submit message editing
   *   el-l-m-ed-c ....................cancel message editing
   *   el-l-m-mv   ....................message move
   *   el-l-m-mv-s ....................submit message move
   *   el-l-m-mv-c ....................cancel message move
   *   el-l-c      ..........child of a message
   *   el-l-rs     ..........runs
   *   el-l-r      ..........run
   *   el-p        .....post 
   *
   * Suffixes:
   *
   *   -auto    auto-refresh
   *   -con     container
   *   -hdr     header
   *   -dsc     description
   *   -ctx     context
   *   -dlg     dialog
   *   -dlgs    dialogs
   *   -rdlg    'reply' dialog
   *   -edlg    'edit' dialog
   *   -mdlg    'move' dialog
   *   -tgl     toggler
   *   -hdn     hidden
   *   -vis     visible
   *   -info    information
   *   -subj    subject
   */

  #el-l-mctrl {
    margin-top:10px;
    margin-bottom:30px;
  }

  #el-l-auto {
    padding:8px;
    border:1px solid #A6C9E2;
    border-radius:5px;
    font-size: 80%;
  }

  #el-l-ms {
  }

  #el-l-ms-info {
    margin-right:15px;
    margin-bottom:10px;
    color:maroon;
  }

  .el-l-m-d-tgl,
  .el-l-m-tgl,
  .el-l-c-tgl,
  .el-l-r-tgl,
  .el-l-a-tgl {
    background-color:#ffffff;
    border:1px solid #c0c0c0;
  }

  .el-l-m-d {
    font-size:120%;
    margin-left:10px;
  }

  .el-l-a-dsc {
    font-weight:bold;
    margin-left:10px;
  }

  .el-l-m-time,
  .el-l-c-time {
    margin-left:10px;
  }

  .el-l-m-author,
  .el-l-c-author {
    font-weight:bold;
    margin-left:10px;
    width:80px;
  }

  .el-l-m-subj,
  .el-l-c-subj {
    margin-left:10px;
  }

  .el-l-a {
    margin-right:10px;
    margin-bottom:10px;
    padding:5px;
  }

  div.el-l-a:hover {
    background-color:#d0d0d0;
    border-radius:5px;
  }

  .el-l-m-d-hdr {
    padding:5px;
    background-color:#b9dcf5;
    border:1px solid #A6C9E2;
    border-bottom:0px;
    border-radius:5px;
    border-bottom-left-radius:0px;
    border-bottom-right-radius:0px;
    cursor:pointer;
  }

  div.el-l-m-d-hdr:hover {
    background-color:#A6C9E2;
  }

  .el-l-m-d-con {
    padding:25px;
    border:1px solid #A6C9E2;
    border-top:0px;
    border-bottom:0px;
    border-radius:5px;
    border-top-left-radius:0px;
    border-top-right-radius:0px;
  }

  .el-l-m-hdr,
  .el-l-c-hdr {
    padding:5px;
    background-color:#e0e0e0;
    border:1px solid #d0d0d0;
    border-bottom:0px;
    border-radius:5px;
    border-bottom-left-radius:0px;
    border-bottom-right-radius:0px;
    cursor:pointer;
  }

  div.el-l-m-hdr:hover,
  div.el-l-c-hdr:hover {
    background-color:#d0d0d0;
  }

  .el-l-r-hdr {
    padding:5px;
    background-color:#DEF0CD;
    border:1px solid #B9DD97;
    border-bottom:0px;
    border-radius:5px;
    border-bottom-left-radius:0px;
    border-bottom-right-radius:0px;
    cursor:pointer;
  }

  div.el-l-r-hdr:hover {
    background-color:#B9DD97;
  }

  .el-l-m-con,
  .el-l-c-con {
    padding:25px;
    padding-top:5px;
    border:1px solid #d0d0d0;
    border-top:0px;
    border-radius:5px;
    border-top-left-radius:0px;
    border-top-right-radius:0px;
  }

  .el-l-r-con {
    padding:20px;
    border:1px solid #B9DD97;
    border-top:0px;
    border-radius:5px;
    border-top-left-radius:0px;
    border-top-right-radius:0px;
  }

  .el-l-a-con {
    margin-left:10px;
    padding-top:5px;
    padding-left:20px;
  }

  .el-l-m-body,
  .el-l-c-body {
    padding-top:10px;
    padding-bottom:10px;
  }

  .el-l-m-body pre {
    margin: 0px;
    padding-top: 5px;
    padding-bottom: 10px;
    overflow: auto;
  }

  .el-l-m-as {
    padding:10px;
    border-top:1px solid #c8c8c8;
  }

  .el-l-a-info {
    margin-left:5px;
  }

  .el-l-m-tags {
    padding:10px;
    border-top:1px solid #c8c8c8;
  }

  .el-l-m-rdlg,
  .el-l-m-edlg,
  .el-l-m-mdlg {
    margin-top: 10px;
    margin-bottom: 10px;
    padding: 20px;
    padding-bottom: 10px;
    background-color: #e0e0e0;
    border:1px solid #c0c0c0;
    border-radius: 5px;
  }

  #el-l-ctx {
    position:fixed;
    right:15px;
    bottom:15px;
    border:2px solid;
    border-radius:5px;
    padding:10px;
  }

  #el-l-ctx-exp {
    font-size: 150%;
    font-weight: bold;
  }

  #el-l-ctx-day {
    margin-top: 5px;
    font-size:36px;
  }

  #el-l-ctx-info {
  margin-top:5px;
    font-size:110%;
  }

  .el-l-m-highlight {
    background-color:#ff6666;
    color: #ffffff;
  }

  .el-l-m-d-hdn,
  .el-l-m-hdn,
  .el-l-c-hdn,
  .el-l-r-hdn,
  .el-l-a-hdn,
  .el-l-ctx-hdn,
  .el-l-m-dlg-hdn {
    display: none;
  }

  .el-l-m-d-vis,
  .el-l-m-vis,
  .el-l-c-vis,
  .el-l-r-vis,
  .el-l-a-vis,
  .el-l-ctx-vis,
  .el-l-m-dlg-vis {
    display: block;
  }

  /* small button container */

  .s-b-con {
    margin-left: 5px;
    font-size:80%;
  }

</style>

<!--[if lte IE 10]>
<style type="text/css">
  .el-l-ctx-vis {
    display: none;
  }
</style>

<![endif]--> 

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
elog.author  = '<?=$auth_svc->authName()?>';
elog.exp_id        = '<?=$exper_id?>';
elog.exp = '<?=$experiment->name()?>';
elog.instr = '<?=$experiment->instrument()->name()?>';
elog.rrange   = '<?=$range_of_runs?>';
elog.min_run         = <?=(is_null($min_run)?'null':$min_run)?>;
elog.max_run         = <?=(is_null($max_run)?'null':$max_run)?>;
<?php
	foreach( $logbook_runs as $run ) echo "elog.runs[{$run->num()}]={$run->id()};\n";
	foreach( $logbook_shifts as $shift ) echo "elog.shifts['{$shift->begin_time()->toStringShort()}']={$shift->id()};\n";
?>
elog.editor = <?=(LogBookAuth::instance()->canEditMessages( $experiment->id())?'true':'false')?>

var extra_params = new Array();
<?php
	if( isset($params)) {
		foreach( $params as $p ) {
			$kv = explode(':',$p);
			switch(count($kv)) {
			case 0:
				break;
			case 1:
				$k = $kv[0];
				echo "extra_params['{$k}']=true;\n";
				break;
			default:
				$k = $kv[0];
				$v = $kv[1];
				echo "extra_params['{$k}']='{$v}';\n";
				break;
			}
		}
	}
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
	$('#files-search-filter :input:text[name=runs_range]' ).val( elog.rrange );
	$('#files-search-filter :input:radio[name=archived]' ).val( ['yes_or_no'] );
	$('#files-search-filter :input:radio[name=local]' ).val( ['yes_or_no'] );
	$('#files-search-filter :input:checkbox[name=xtc]' ).val( ['XTC'] );
    $('#files-search-filter :input:checkbox[name=hdf5]' ).val( ['HDF5'] );
}

function search_files( import_format ) {

	var params = { exper_id: elog.exp_id };

	if( $('#files-search-filter :input:radio[name=runs]:checked' ).val() != 'all' ) {
		var runs = $('#files-search-filter :input:text[name=runs_range]' ).val();
		if( runs != elog.rrange )	params.runs = runs;
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
	$('#translate-search-filter :input:text[name=runs_range]' ).val( elog.rrange );
	$('#translate-search-filter :input:radio[name=translated]' ).val( ['yes_or_no'] );
}

function search_translate_requests() {

	var params = {
		exper_id: elog.exp_id,
		show_files: 1
	};

	if( $('#translate-search-filter :input:radio[name=runs]:checked' ).val() != 'all' ) {
		var runs = $('#translate-search-filter :input:text[name=runs_range]' ).val();
		if( runs != elog.rrange )	params.runs = runs;
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
					   	{ exper_id: elog.exp_id, runnum: $(this).val() },
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
						{ exper_id: elog.exp_id, id: $(this).val() },
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
		var url = 'Requests2pdf.php?exper_id='+elog.exp_id+'&show_files';
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
		pfcopy.document.write('<title>Data Portal of Experiment: '+elog.instr+' / '+elog.exp+'</title></head><body><div class="maintext">');
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
    		'id' => 'tabs-experiment',
    		'html' => $tabs_experiment,
            'class' => 'tab-inline-content',
    	    'callback' => 'set_current_tab("tabs-experiment")'
    	)
    );

    /*
     * [ e-log ]
     */
    $tabs_elog_subtabs = array();
    $tabs_elog_recent =<<<HERE
    <div id="el-l-mctrl">
      <div style="float:left; margin-left:20px;">
        <div style="float:left;">
          <div style="font-weight:bold;">Last messages to display:</div>
          <div id="elog-live-range-selector" style="margin-top:4px;">
            <input type="radio" id="elog-live-range-20"    name="range" value="20"  checked="checked" /><label for="elog-live-range-20"    >20</label>
            <input type="radio" id="elog-live-range-100"   name="range" value="100"                   /><label for="elog-live-range-100"   >100</label>
            <input type="radio" id="elog-live-range-shift" name="range" value="shift"                 /><label for="elog-live-range-shift" >shift</label>
            <input type="radio" id="elog-live-range-day"   name="range" value="day"                   /><label for="elog-live-range-day"   >24 hrs</label>
            <input type="radio" id="elog-live-range-week"  name="range" value="week"                  /><label for="elog-live-range-week"  >7 days</label>
            <input type="radio" id="elog-live-range-all"   name="range" value="all"                   /><label for="elog-live-range-all"   >everything</label>
          </div>
        </div>
        <div style="float:left; margin-left:5px;">
          <div style="font-weight:bold;">Show runs:</div>
          <div style="margin-top:4px;">
            <div id="elog-live-runs-selector" style="float:left;">
              <input type="radio" id="elog-live-runs-on"  name="show_runs" value="on"  checked="checked" /><label for="elog-live-runs-on"  >On</label>
              <input type="radio" id="elog-live-runs-off" name="show_runs" value="off"                   /><label for="elog-live-runs-off" >Off</label>
            </div>
          </div>
        </div>
        <div style="clear:both;"></div>
        <div style="margin-top:10px;">
          <div style="float:left;">
            <button id="elog-live-expand"     title="click a few times to expand the whole tree">Expand++</button>
            <button id="elog-live-collapse"   title="each click will collapse the tree to the previous level of detail">Collapse--</button>
            <button id="elog-live-viewattach" title="view attachments of expanded messages">View Attachments</button>
            <button id="elog-live-hideattach" title="hide attachments of expanded messages">Hide Attachments</button>
          </div>
        </div>
      </div>
      <div style="float:right;" id="el-l-auto">
        <div style="font-weight:bold;">Autorefresh:</div>
        <div style="margin-top:4px;">
          <div id="elog-live-refresh-selector" style="float:left;">
            <input type="radio" id="elog-live-refresh-on"  name="refresh" value="on"  checked="checked" /><label for="elog-live-refresh-on"  >On</label>
            <input type="radio" id="elog-live-refresh-off" name="refresh" value="off"                   /><label for="elog-live-refresh-off" >Off</label>
          </div>
          <div style="float:left; margin-left:10px;">
            <select id="elog-live-refresh-interval">
              <option>2</option>
              <option>5</option>
              <option>10</option>
            </select>
            s.
          </div>
          <div style="clear:both;"></div>
        </div>
        <div style="margin-top:8px;">
          <button id="elog-live-refresh" title="check if there are new updates">Check for updates now</button>
        </div>
      </div>
      <div style="clear:both;"></div>
    </div>
    <div id="el-l-ms-action" style="float:left;"></div>
    <div id="el-l-ms-info" style="float:right;"></div>
    <div style="clear:both;"></div>
    <div id="el-l-ms"></div>
    <div id="el-l-ctx" class="el-l-ctx-hdn">
      <div id="el-l-ctx-exp"></div>
      <div id="el-l-ctx-day"></div>
      <div id="el-l-ctx-info"></div>
    </div>\n
HERE;
    array_push(
		$tabs_elog_subtabs,
    	array(
    		'name' => 'Recent (Live)',
    		'id' => 'tabs-elog-recent',
    		'html' => $tabs_elog_recent,
            'class' => 'tab-inline-content',
    	    'callback' => 'set_current_tab("tabs-elog-recent")'
    	)
    );
    $select_tag_html = "<option> - select tag - </option>\n";
    foreach( $logbook_experiment->used_tags() as $tag ) {
    	$select_tag_html .= "<option>{$tag}</option>\n";
    }

    $tags_html = '';

    // TODO: Move this parameter upstream and make it available
    //       to JavaScript which would use th elibrary to fill ouut
    //       tag names.
    //
    $num_tags = 3;
    for( $i = 0; $i < $num_tags; $i++) {
    	$tags_html .=<<<HERE
name:  <select id="elog-tags-library-{$i}">{$select_tag_html}</select>
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
    		'id'   => 'tabs-el-p-as',
    		'html' => <<<HERE
<div id="el-p-as">
  file: <input type="file" name="file2attach_0" onchange="elog.post_add_attachment(this)" />
  description: <input type="text" name="file2attach_0" value="" title="put an optional file description here" /><br>
</div>
HERE
			,
			'callback' => 'set_current_tab("tabs-el-p-as")'
    	)
    );
    array_push(
		$tabs_elog_post_extra,
    	array(
    		'name' => 'Tags',
    		'id'   => 'tabs-el-p-tags',
    		'html' => <<<HERE
<div id="elog-tags" style="margin-top:4px;">{$tags_html}</div>
HERE
			,
			'callback' => 'set_current_tab("tabs-el-p-tags")'
    	)
    );
    array_push(
		$tabs_elog_post_extra,
    	array(
    		'name' => 'Context & Post Time',
    		'id'   => 'tabs-el-p-context',
    		'html' => <<<HERE
<div>
  <div style="float:left;">
    <div style="font-weight:bold;">Context:</div>
    <div id="el-p-context-selector" style="margin-top:4px;">
      <input type="radio" id="el-p-context-experiment" name="scope" value="experiment" checked="checked" /><label for="el-p-context-experiment">experiment</label>
      <input type="radio" id="el-p-context-shift"      name="scope" value="shift"                        /><label for="el-p-context-shift"     >shift</label>
      <input type="radio" id="el-p-context-run"        name="scope" value="run"                          /><label for="el-p-context-run"       >run</label>
    </div>
  </div>
  <div style="float:left; margin-left:10px;">
    <div style="font-weight:bold;">Shift:</div>
    <div style="margin-top:4px;">
      <select id="el-p-shift">{$shifts_html}</select>
    </div>
  </div>
  <div style="float:left; margin-left:10px;">
    <div style="font-weight:bold;">Run:</div>
    <div style="margin-top:4px;">
      <input type="text" id="el-p-runnum" value="{$max_run}" size=4 />
      <span id="el-p-runnum-error" style="color:red;"></span>
    </div>
  </div>
  <div style="clear:both;"></div>
</div>
<div style="margin-top:20px;">
  <div style="float:left;">
    <div style="font-weight:bold;">Post time:</div>
    <div id="el-p-relevance-selector" style="margin-top:4px;">
      <input type="radio" id="el-p-relevance-now"   name="relevance" value="now"   checked="checked" /><label for="el-p-relevance-now"   title="it will be the actual posting time"      >now</label>
      <input type="radio" id="el-p-relevance-past"  name="relevance" value="past"                    /><label for="el-p-relevance-past"  title="use date and time selector on the right" >past</label>
      <input type="radio" id="el-p-relevance-shift" name="relevance" value="shift"                   /><label for="el-p-relevance-shift" title="within specified shift"                  >in shift</label>
      <input type="radio" id="el-p-relevance-run"   name="relevance" value="run"                     /><label for="el-p-relevance-run"   title="within specified run"                    >in run</label>
    </div>
  </div>
  <div style="float:left; margin-left:10px;">
    <div style="font-weight:bold;">&nbsp;</div>
    <div style="margin-top:4px;">
      <input type="text" id="el-p-datepicker" value="{$today}" size=11 />
      <input type="text" id="el-p-time" value="{$now}"  size=8 />
    </div>
  </div>
  <div style="clear:both"></div>
</div>
HERE
			,
			'callback' => 'set_current_tab("tabs-el-p-context")'
    	)
    );
    $tabs_elog_post_extra = DataPortal::tabs_html( "tabs-el-p-subtabs", $tabs_elog_post_extra );
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
    		'id' => 'tabs-elog-post',
    		'html' => $tabs_elog_post,
    		'class' => 'tab-inline-content',
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
    		'id' => 'tabs-elog-search',
    		'html' => $tabs_elog_search,
            'class' => 'tab-inline-content',
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
    		'id' => 'tabs-elog-browse',
            'html' => $tabs_elog_browse,
            'class' => 'tab-inline-content',
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
    		'id' => 'tabs-elog-runs',
            'html' => $tabs_elog_runs,
            'class' => 'tab-inline-content',
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
            'id' => 'tabs-elog-shifts',
            'html' => $tabs_elog_shifts,
            'class' => 'tab-inline-content',
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
            'id' => 'tabs-elog-subscribe',
            'html' => $tabs_elog_subscribe,
            'class' => 'tab-inline-content',
            'callback' => 'set_current_tab("tabs-elog-subscribe")'
    	)
    );

    array_push(
		$tabs,
    	array(
    		'name' => 'e-Log',
    		'id' => 'tabs-elog',
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
           'id' => 'tabs-files-4runs',
            'html' => $tabs_files_4runs,
            'class' => 'tab-inline-content',
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
            'class' => 'tab-inline-content',
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
            'name' => 'History',
            'id'   => 'tabs-translate-history',
            'html' => $tabs_translate_history,
            'class' => 'tab-inline-content',
    	    'callback' => 'set_current_tab("tabs-translate-history")'
    	)
    );
    array_push(
		$tabs,
    	array(
    		'name' => 'HDF5 Translation',
    		'id'   => 'tabs-translate',
    		'html' => DataPortal::tabs_html( "tabs-translate-subtabs", $tabs_translate_subtabs ),
    	    'callback' => 'set_current_tab("tabs-translate")'
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
