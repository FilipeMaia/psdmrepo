<?php

require_once('DataPortal/DataPortal.inc.php');
require_once('RegDB/RegDB.inc.php');
require_once('FileMgr/FileMgr.inc.php');

/* Let a user to select an experiment first if no valid experiment
 * identifier is supplied to the script.
 */
if( !isset( $_GET['exper_id'] )) {
	header("Location: select_experiment.php");
	exit;
}
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' ) die( 'no valid experiment identifier provided to the script' );

$page1 = trim( $_GET['page1'] );
$page2 = trim( $_GET['page2'] );

try {

	$regdb = new RegDB();
	$regdb->begin();

	$experiment = $regdb->find_experiment_by_id( $exper_id );
	if( is_null( $experiment )) die( 'invalid experiment identifier provided to the script' );

    $instrument = $experiment->instrument();

    /* Get the stats for data files
     */
    $num_runs       = 0;
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

<!----------------------------------------------------------------->



<?php
    DataPortal::scripts( "page_specific_init" );
?>


<!------------------ Page-specific JavaScript ---------------------->
<script type="text/javascript">


/* -----------------------------------------
 *             GLOBAL VARIABLES
 * -----------------------------------------
 */
var exper_id = '<?=$exper_id?>';
var experiment_name = '<?=$experiment->name()?>';
var instrument_name = '<?=$experiment->instrument()->name()?>';
var range_of_runs = '<?=$range_of_runs?>';
var page1 = '<?=(isset($page1) ? $page1 : "null")?>';
var page2 = '<?=(isset($page2) ? $page2 : "null")?>';


/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */

function page_specific_init() {

	$('#tabs').tabs();

	init_tab_experiment();
	init_tab_files();
	init_tab_hdf5();

	/* Open the initial tab if explicitly requested. Otherwise the first
	 * tab will be shown.
	 */
    if( page1 != null ) $('#tabs').tabs( 'select', '#tabs-'+page1 );
    if( page2 != null ) $('#tabs-'+page1+'-subtabs').tabs( 'select', '#tabs-'+page1+'-'+page2 );
}

/* ----------------------------------------
 *             TAB: EXPERIMENT
 * ----------------------------------------
 */
function init_tab_experiment() {

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
function init_tab_files() {

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
	$('#files-search-filter :input:text[name=runs_range]' ).val( range_of_runs );
	$('#files-search-filter :input:radio[name=archived]' ).val( ['yes_or_no'] );
	$('#files-search-filter :input:radio[name=local]' ).val( ['yes_or_no'] );
	$('#files-search-filter :input:checkbox[name=xtc]' ).val( ['XTC'] );
    $('#files-search-filter :input:checkbox[name=hdf5]' ).val( ['HDF5'] );
}

function search_files( import_format ) {

	var params = { exper_id: exper_id };

	if( $('#files-search-filter :input:radio[name=runs]:checked' ).val() != 'all' ) {
		var runs = $('#files-search-filter :input:text[name=runs_range]' ).val();
		if( runs != range_of_runs )	params.runs = runs;
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
 function init_tab_hdf5() {

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
	$('#translate-search-filter :input:text[name=runs_range]' ).val( range_of_runs );
	$('#translate-search-filter :input:radio[name=translated]' ).val( ['yes_or_no'] );
}

function search_translate_requests() {

	var params = {
		exper_id: exper_id,
		show_files: 1
	};

	if( $('#translate-search-filter :input:radio[name=runs]:checked' ).val() != 'all' ) {
		var runs = $('#translate-search-filter :input:text[name=runs_range]' ).val();
		if( runs != range_of_runs )	params.runs = runs;
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
					   	{ exper_id: exper_id, runnum: $(this).val() },
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
						{ exper_id: exper_id, id: $(this).val() },
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
		var url = 'Requests2pdf.php?exper_id='+exper_id+'&show_files';
		var winRef = window.open( url, 'Translation Requests' );
	}
}

function printer_friendly( element ) {
	var el = document.getElementById(element);
	if (el) {
		var html = document.getElementById(element).innerHTML;
		var xopen = window.open("about:blank");
		xopen.document.write('<html xmlns="http://www.w3.org/1999/xhtml">');
		xopen.document.write('<head><meta http-equiv="Content-Type" content="text/html; charset=windows-1252" />');
		xopen.document.write('<link rel="stylesheet" type="text/css" href="css/default.css" />');
		xopen.document.write('<link type="text/css" href="css/portal.css" rel="Stylesheet" />');
		xopen.document.write('<title>Data Portal of Experiment: '+instrument_name+' / '+experiment_name+'</title></head><body><div class="maintext">');
		xopen.document.write(html);
		xopen.document.write("</div></body></html>");
		xopen.document.close();
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


  <div id="tabs">
	<ul>
	  <li><a href="#tabs-experiment">Experiment</a></li>
	  <li><a href="#tabs-elog">e-Log</a></li>
	  <li><a href="#tabs-files">Data Files</a></li>
	  <li><a href="#tabs-translate">XTC/HDF5 Translation</a></li>
	  <li><a href="#tabs-account">My Account</a></li>
	</ul>
	<div id="tabs-experiment" class="tab-inline-content">
	  <button id="button-select-experiment">Select another experiment</button>
      <table>
        <tbody cellspacing=4>
          <tr>
            <td class="table_cell_left">Id</td>
            <td class="table_cell_right"><?=$experiment->id()?></td>
          </tr>
          <tr>
            <td class="table_cell_left">Status</td>
            <td class="table_cell_right"><?=DataPortal::decorated_experiment_status_UP( $experiment )?></td>
          </tr>
          <tr>
            <td class="table_cell_left">Begin</td>
            <td class="table_cell_right"><?=$experiment->begin_time()->toStringShort()?></td>
          </tr>
          <tr>
            <td class="table_cell_left">End</td>
            <td class="table_cell_right"><?=$experiment->end_time()->toStringShort()?></td>
          </tr>
          <tr>
            <td class="table_cell_left">Description</td>
            <td class="table_cell_right"><pre style="background-color:#e0e0e0; padding:0.5em;"><?=$experiment->description()?></pre></td>
          </tr>
          <tr>
            <td class="table_cell_left">Contact</td>
            <td class="table_cell_right"><?=DataPortal::decorated_experiment_contact_info( $experiment )?></td>
          </tr>
          <tr>
            <td class="table_cell_left">Leader</td>
            <td class="table_cell_right"><?=$experiment->leader_Account()?></td>
          </tr>
          <tr>
            <td class="table_cell_left table_cell_bottom" valign="top">POSIX Group</td>
            <td class="table_cell_right table_cell_bottom">
              <table cellspacing=0 cellpadding=0><tbody>
                <tr>
                  <td valign="top"><?=$experiment->POSIX_gid()?></td>
                  <td>&nbsp;</td>
                  <td>
                    <button id="button-toggle-group" title="click to see/hide the list of members"><div class="ui-icon ui-icon-triangle-1-s"></div></button>
                    <div id="group-members" class="group-members-hidden">
                      <table><tbody>
                      <?php
                        $idx = 0;
                    	foreach( $experiment->group_members() as $m ) {
                    		$uid   = $m['uid'];
                    		$gecos = $m['gecos'];
                    		echo <<<HERE
                         <tr><td><b>{$uid}</b></td><td>{$gecos}</td></tr>

HERE;
                    	}
                    	?>
                      </tbody></table>
                    </div>
                  </td>
                </tr>
              </tbody></table>
            </td>
          </tr>
        </tbody>
      </table>
	</div>
	<div id="tabs-elog" class="tab-inline-content">
      <p>Electronic LogBook should be seen here. But firt we need to redesign it using JavaScript
      classes to mavoid various sorts of conflicts.</p>
	</div>
	<div id="tabs-files" class="tab-inline-content">
	  <div>
        <div id="files-search-summary" style="float: left">
          <table><tbody>
            <tr>
              <td class="grid-sect-hdr-first">R u n s</td>
            </tr>
            <tr>
              <td class="grid-key">Number of runs:</td>
              <td class="grid-value"><?=$num_runs?></td>
            </tr>
            <tr>
              <td class="grid-sect-hdr">X T C</td>
            </tr>
            <tr>
              <td class="grid-key">Number of files:</td>
              <td class="grid-value"><?=$xtc_num_files?></td>
              <td class="grid-key">Size [GB]:</td>
              <td class="grid-value"><?=$xtc_size_str?></td>
            </tr>
            <tr>
              <td class="grid-key">Archived to tape:</td>
              <td class="grid-value"><?=$xtc_archived?> / <?=$xtc_num_files?></td>
              <td class="grid-key">On disk:</td>
              <td class="grid-value"><?=$xtc_local_copy?> / <?=$xtc_num_files?></td>
            </tr>
             <tr>
             <td class="grid-sect-hdr">H D F 5</td>
            </tr>
            <tr>
              <td class="grid-key">Number of files:</td>
              <td class="grid-value"><?=$hdf5_num_files?></td>
              <td class="grid-key">Size [GB]:</td>
              <td class="grid-value"><?=$hdf5_size_str?></td>
            </tr>
            <tr>
              <td class="grid-key">Archived to tape:</td>
              <td class="grid-value"><?=$hdf5_archived?> / <?=$hdf5_num_files?></td>
              <td class="grid-key">On disk:</td>
              <td class="grid-value"><?=$hdf5_local_copy?> / <?=$hdf5_num_files?></td>
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
                <td class="selector-value"><input type="text" name="runs_range" value="<?=$range_of_runs?>" width=10 title="1,3,5,10-20,200"></td>
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
            <button id="button-files-filter-reset">Reset Filter</button>
            <button id="button-files-filter-apply">Apply Filter</button>
            <button id="button-files-filter-import">Import List</button>
          </div>
          <div style="clear: both;"></div>
        </div>
        <div style="clear: both;"></div>
      </div>
      <div id="files-search-result"></div>
	</div>

 	<div id="tabs-translate">
 
      <div id="tabs-translate-subtabs">
	    <ul>
	      <li><a href="#tabs-translate-manage">Manage</a></li>
	      <li><a href="#tabs-translate-history">History of Requests</a></li>
	    </ul>
	    <div id="tabs-translate-manage" class="tab-inline-content">
	      <div style="float:right;"><a href="javascript:printer_friendly('tabs-translate-manage')" title="Printer friendly version of this page"><img src="img/PRINTER_icon.png" /></a></div>
	      <div style="float:right; margin-right:10px;"><a href="javascript:pdf('translate-manage')" title="PDF version of this page"><img src="img/PDF_icon.jpg" /></a></div>
	      <div stype="clear:both;"></div>
		  <div>
	        <div id="translate-search-summary" style="float:left">
	          <table><tbody>
	            <tr>
	              <td class="grid-sect-hdr-first">R u n s</td>
	            </tr>
	            <tr>
	              <td class="grid-key">Number of runs:</td>
	              <td class="grid-value"><?=$num_runs?></td>
	            </tr>
	            <tr>
	              <td class="grid-sect-hdr">T r a n s l a t i o n</td>
	            </tr>
	            <tr>
	              <td class="grid-key">Complete:</td>
	              <td class="grid-value"><?=$hdf5_num_runs_complete?></td>
	            </tr>
	            <tr>
	              <td class="grid-key">Failed:</td>
	              <td class="grid-value"><?=$hdf5_num_runs_failed?></td>
	            </tr>
	            <tr>
	              <td class="grid-key">Waiting:</td>
	              <td class="grid-value"><?=$hdf5_num_runs_wait?></td>
	            </tr>
	            <tr>
	              <td class="grid-key">Being translated:</td>
	              <td class="grid-value"><?=$hdf5_num_runs_translate?></td>
	            </tr>
	            <tr>
	              <td class="grid-key">Other state:</td>
	              <td class="grid-value"><?=$hdf5_num_runs_unknown?></td>
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
	                <td class="selector-value"><input type="text" name="runs_range" value="<?=$range_of_runs?>" width=10 title="1,3,5,10-20,200"></td>
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
	            <button id="button-translate-filter-reset">Reset Filter</button>
	            <button id="button-translate-filter-apply">Apply Filter</button>
	          </div>
	          <div style="clear:both;"></div>
	        </div>
	        <div style="clear:both;"></div>
	      </div>
	      <div id="translate-search-result"></div>
	    </div>

	    <div id="tabs-translate-history" class="tab-inline-content">
	      <p>Here be the list of all translation requests. And there will be a filter on the top right
	      side to allow.</p>
	    <!-- 
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
	     -->
        </div>
      </div>
    </div>
	<div id="tabs-account" class="tab-inline-content">
      <p>User account information, privileges, POSIX groups, other experiments participation, subscriptions, etc.</p>
	</div>
  </div>

<!----------------------------------------------------------------->






<?php
    DataPortal::end();
?>
<!--------------------- Document End Here -------------------------->


<?php

} catch( AuthDBException  $e ) { print $e->toHtml();
} catch( FileMgrException $e ) { print $e->toHtml();
} catch( LogBookException $e ) { print $e->toHtml();
} catch( RegDBException   $e ) { print $e->toHtml();
} catch( FileMgrException $e ) { print $e->toHtml();
}

?>
