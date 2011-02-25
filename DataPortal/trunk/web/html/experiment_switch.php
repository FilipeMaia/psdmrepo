<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\DataPortal;

use RegDB\RegDB;
use RegDB\RegDBException;

use LusiTime\LusiTime;

/* Let a user to select an experiment first if no valid experiment
 * identifier is supplied to the script.
 */
if( isset( $_GET['instr_name'] )) {
	$instr_name = trim( $_GET['instr_name'] );
	if( $instr_name == '' ) die( 'no valid instrument name provided to the script' );
}

try {

	$regdb = new RegDB();
	$regdb->begin();

	$authdb = new AuthDB();
	$authdb->begin();

	if( isset( $instr_name )) {
		$instrument = $regdb->find_instrument_by_name( $instr_name );
		if( is_null( $instrument )) die( 'unknown instrument name provided to the script' );
	}

?>




<!------------------- Document Begins Here ------------------------->
<?php
    DataPortal::begin( "Experiment Switch" );
?>



<!------------------- Page-specific Styles ------------------------->

<link type="text/css" href="css/portal.css" rel="Stylesheet" />

<style type="text/css">
  .module-is-visible {
    display: block;
  }
  .module-is-hidden {
    display: none;
  }
</style>

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
var page1 = '<?=(isset($instrument) ? $instrument->name() : "null")?>';
var instruments = new Array();
<?php
	$idx = 0;
	foreach( $regdb->instruments() as $instrument )
		if( !$instrument->is_location()) {
			echo <<<HERE
instruments[{$idx}] = "{$instrument->name()}";

HERE;
			$idx++;
		}
?>

/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */

function page_specific_init() {

	$('#instruments').tabs();

	for( var i=0; i< instruments.length; i++)
		init_tab( instruments[i]);

	/* Open the initial tab if explicitly requested. Otherwise the first
	 * tab will be shown.
	 */
    if( page1 != null ) $('#instruments').tabs( 'select', '#instrument-'+page1 );
}

/* --------------------------------------------------------
 *             TOP LEVEL TABS FOR INSTRUMENTS
 * --------------------------------------------------------
 */
function init_tab( instrument ) {
	$('#instrument-'+instrument).tabs();
	$('#button-switch-experiment-'+instrument).button().click(
		function() {
			$('#current-experiment-'+instrument).removeClass( 'module-is-visible' ).addClass( 'module-is-hidden' );
			$('#select-experiment-' +instrument).removeClass( 'module-is-hidden'  ).addClass( 'module-is-visible' );
		}
	);
	$('#button-submit-experiment-'+instrument).button().click(
		function() {
			$('form[name="form-'+instrument+'"]').submit();
		}
	);
	$('#button-cancel-experiment-'+instrument).button().click(
		function() {
			$('#current-experiment-'+instrument).removeClass( 'module-is-hidden'  ).addClass( 'module-is-visible' );
			$('#select-experiment-' +instrument).removeClass( 'module-is-visible' ).addClass( 'module-is-hidden' );
		}
	);
	$( "#search2notify-"+instrument ).autocomplete({
		source: 'search_account.php',
		minLength: 2,
		select: function( event, ui ) {
			var user = eval('('+ui.item.value+')');
			$( '#search2notify-'+instrument ).val( '' );
			$( '<div/>' ).html( '<input type="checkbox" name="notify_other_'+user.uid+'" checked="checked" /> '+user.name+'<br>' ).appendTo( '#registered-'+instrument );
			return false;
		}
	});
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

// The function is called from a dialog for selecting a new experiment to switch
// to when a new experiment is selected from a listbox. In this case we need to
// update the contact information for the PI.

function on_experiment_select( elem, instrument ) {

	var splitString = elem.value.split(',');
	var experiment  = splitString[0];
	$.ajax({
		url: 'experiment_info.php?name='+experiment,
		success: function( data ) {
			var data = eval( '('+data+')' );  // translate into a JSON object
			$('#input-notify-pi-'+instrument).html(
				'<input type="checkbox" name="notify_pi_'+
				data['leader_uid']+'" checked="checked" title="ATTENTION: this option can not be unchecked" onclick="this.checked=true" /> '+
				data['leader_gecos']+'<br>' );
		}
	});
}

</script>
<!----------------------------------------------------------------->



<?php
    DataPortal::body(
    	'Experiment Switch',
    	'',
    	'instrument'
   	);
?>




<!------------------ Page-specific Document Body ------------------->
<?php

	$tabs = array();

	foreach( $regdb->instruments() as $instrument ) {

		if( $instrument->is_location()) continue;

		/* Generate the page for the last switched experiment (if any)
		 */
		$last_experiment_name             = '&lt; No experiment were previously switched by this tool &gt;';
		$last_experiment_id               = '&lt; none &gt;';
		$last_experiment_begin_time       = '&lt; none &gt;';
		$last_experiment_end_time         = '&lt; none &gt;';
		$last_experiment_description      = '&lt; none &gt;';
		$last_experiment_contact          = '&lt; none &gt;';
		$last_experiment_leader           = '&lt; none &gt;';
		$last_experiment_group            = '&lt; none &gt;';
		$last_experiment_switch_time      = '&lt; none &gt;';
		$last_experiment_switch_requestor = '&lt; none &gt;';

		$last_experiment_switch = $regdb->last_experiment_switch( $instrument->name());
		if( !is_null( $last_experiment_switch )) {

			$exper_id = $last_experiment_switch['exper_id'];
			$experiment = $regdb->find_experiment_by_id( $exper_id );

			if( is_null( $experiment ))
				die( "fatal internal error when resolving experiment id={$exper_id} in the database" );

			$last_experiment_name             = $experiment->name();
			$last_experiment_id               = $experiment->id();
			$last_experiment_begin_time       = $experiment->begin_time()->toStringShort();
			$last_experiment_end_time         = $experiment->end_time()->toStringShort();
			$last_experiment_description      = $experiment->description();
			$last_experiment_contact          = DataPortal::decorated_experiment_contact_info( $experiment );
			$last_experiment_leader           = $experiment->leader_account();
			$last_experiment_group            = $experiment->POSIX_gid();
			$last_experiment_switch_time      = LusiTime::from64( $last_experiment_switch['switch_time'] )->toStringShort();
			$last_experiment_switch_requestor = $last_experiment_switch['requestor_uid'];
		}

		$html =<<<HERE
<div id="current-experiment-{$instrument->name()}" class="module-is-visible">
  <button id="button-switch-experiment-{$instrument->name()}">Change</button>
  <br>
  <br>
  <table>
    <tbody>
      <tr>
        <td class="table_cell_left">Name</td>
        <td class="table_cell_right">{$last_experiment_name}</td>
      </tr>
      <tr>
        <td class="table_cell_left">Id</td>
        <td class="table_cell_right">{$last_experiment_id}</td>
      </tr>
      <tr>
        <td class="table_cell_left">Begin</td>
        <td class="table_cell_right">{$last_experiment_begin_time}</td>
      </tr>
      <tr>
        <td class="table_cell_left">End</td>
        <td class="table_cell_right">{$last_experiment_end_time}</td>
      </tr>
      <tr>
        <td class="table_cell_left">Description</td>
        <td class="table_cell_right"><pre style="background-color:#e0e0e0; padding:0.5em;">{$last_experiment_description}</pre></td>
      </tr>
      <tr>
        <td class="table_cell_left">Contact</td>
        <td class="table_cell_right">{$last_experiment_contact}</td>
      </tr>
      <tr>
        <td class="table_cell_left">Leader</td>
        <td class="table_cell_right">{$last_experiment_leader}</td>
      </tr>
      <tr>
        <td class="table_cell_left">POSIX Group</td>
        <td class="table_cell_right">{$last_experiment_group}</td>
      </tr>
      <tr>
        <td class="table_cell_left">Switched On</td>
        <td class="table_cell_right">{$last_experiment_switch_time}</td>
      </tr>
      <tr>
        <td class="table_cell_left table_cell_bottom">Switch Requested By</td>
        <td class="table_cell_right table_cell_bottom">{$last_experiment_switch_requestor}</td>
      </tr>
    </tbody>
  </table>
</div>
HERE;

		/* Generate invisible module with a selector of experiment. This page will
		 * be togled on/off by Change/Cancel buttons.
		 */
		$experiment = null;	// TODO: the first experiment in the list will be selected as
							//       the default candidate for the switch. This needs to
							//       be fixed by selecting the last switched experiment
							//       if the one is available.

		$experiments_html =<<<HERE
<select name="experiment" id="select-experiment-{$instrument->name()}" onchange="on_experiment_select(this,'{$instrument->name()}')">
HERE;
		foreach( $regdb->experiments_for_instrument( $instrument->name()) as $e ) {

			if( is_null( $experiment ))	$experiment = $e; // remember the first experiment in the list.

			$selected_option = $last_experiment_id == $e->id() ? 'selected="selected"' : '';
			$experiments_html .= "<option {$selected_option}>{$e->name()}, id={$e->id()}</option>";
		}
		$experiments_html .= '</select>';

		$opr_account = strtolower($instrument->name()).'opr';

		$opr_account_prev_auth_html = '';
		foreach( $authdb->roles_by( $opr_account, 'LogBook', $instrument->name()) as $r ) {
			if( $opr_account_prev_auth_html == '' ) $opr_account_prev_auth_html .= '<div style="float:left; padding:10px;">';
			$instr    = $r['instr'];
			$exper    = $r['exper'];
			$exper_id = $r['exper_id'];
			$role     = $r['role' ];
			$opr_account_prev_auth_html .=<<<HERE
<input type="checkbox" name="oprelogauth_{$role->name()}_{$exper_id}" checked="checked" /> {$role->name()} @ {$exper} (id={$exper_id})<br>
HERE;
		}
		if( $opr_account_prev_auth_html != '' ) {
			$opr_account_prev_auth_html .=<<<HERE
</div>
<div style="float:left; padding:10px; width:180px;">
  <span style="color:red;">Uncheck those authorizations which may compromise new experiment's
  data privacy/security after accomplishing the switch.</span>
</div>
<div style="clear:both;"></div>
HERE;
		}
		$opr_account_auth_html =<<<HERE
<div style="margin:10px;">
  <div style="float:left; padding-right:40px;">
    <b>For selected experiment:</b><br>
    <div style="padding:10px;">
      <input type="radio" name="oprelogauth" value="" />No Aurhorization<br>
      <input type="radio" name="oprelogauth" value="Reader" />Reader<br>
      <input type="radio" name="oprelogauth" value="Writer" checked="checked" />Writer<br>
      <input type="radio" name="oprelogauth" value="Editor" />Editor
    </div>
  </div>
  <div style="float:left;">
    <b>For previous experiments:</b><br>
    {$opr_account_prev_auth_html}
  </div>
  <div style="clear:both;"></div>
</div>
HERE;

    	$checked_readonly = 'checked="checked" title="ATTENTION: this option can not be unchecked" onclick="this.checked=true"';

    	$admins = array( 'perazzo', 'gapon', 'mcmesser' );
    	$admins_html = '';
    	foreach( $admins as $uid ) {
    		$account = $regdb->find_user_account( $uid );
    		$gecos = $account['gecos'];
    		$email = $account['email'];
    		$admins_html .=<<<HERE
<input type="checkbox" name="notify_admin_{$uid}" {$checked_readonly} /> {$gecos}<br>
HERE;
    	}
    	
    	$instrument_scientists_group = 'ps-'.strtolower($instrument->name());
    	$instrument_scientists_html = '';
    	foreach( $regdb->posix_group_members( $instrument_scientists_group, /* $and_as_primary_group=*/ false ) as $account ) {
    		$uid = $account['uid'];
    		$gecos = $account['gecos'];
    		$instrument_scientists_html .=<<<HERE
<input type="checkbox" name="notify_is_{$uid}" {$checked_readonly} /> {$gecos}<br>
HERE;
    	}

    	$pi_account = $regdb->find_user_account( $last_experiment_leader /*$experiment->leader_account()*/ );
    	$pi_uid   = $pi_account['uid'];
    	$pi_gecos = $pi_account['gecos'];
    	$pi_email = $pi_account['email'];
		$notify_html =<<<HERE
<div style="margin:10px;">
  <div style="float:left;">
    <div style="float:left;">
      <b>{$instrument->name()} instrument scientists (POSIX group '{$instrument_scientists_group}'):</b><br>
      <div style="padding:10px;">
        {$instrument_scientists_html}
      </div>
    </div>
    <div style="float:left; margin-left:40px;">
      <b>PI of the experiment:</b><br>
      <div style="padding:10px;" id="input-notify-pi-{$instrument->name()}">
        <input type="checkbox" name="notify_pi_{$pi_uid}" {$checked_readonly} /> {$pi_gecos}<br>
      </div>
    </div>
    <div style="clear:both;"></div>
    <b>Others:</b><br>
    <div style="padding:10px;" id="registered-{$instrument->name()}">
      {$admins_html}
    </div>
    <div style="margin-top:10px;">
      <b>Add more recipients to be notified (search LDAP):</b> <input type="text" value="" id="search2notify-{$instrument->name()}" />
    </div>
  </div>
  <div style="clear:both;"></div>
  <div style="margin-top:20px;">
    <b>Additional instructions to be sent by e-mail:</b><br>
    <textarea rows="10" cols="64" name="instructions"></textarea>
  </div>
</div>
HERE;

		$html .=<<<HERE
<div id="select-experiment-{$instrument->name()}" class="module-is-hidden">
  <b>HOW TO USE THIS FORM:</b> Press 'Submit' to proceed with the switch and all relevant changes in the system.
  Persons mentioned in the list below will be notified by e-mail. Additional instructions
  (if provided in a text area below) will be also sent to each recipient. Proper adjustments
  to the special 'opr' account will be made as requested. The parameters
  of the switch will be recorded in a database.
  <br>
  <br>
  <button id="button-submit-experiment-{$instrument->name()}">Submit</button>
  <button id="button-cancel-experiment-{$instrument->name()}">Cancel</button>
  <br>
  <br>
  <div style="padding:20px; background-color:#f3f3f3; ">
  <form name="form-{$instrument->name()}" action="ProcessExperimentSwitch.php" method="post">
    <table>
      <tbody>
        <tr>
          <td class="table_cell_left"  valign="top"><div style="margin:10px;">Select experiment</div></td>
          <td class="table_cell_right" valign="top">{$experiments_html}</td>
        </tr>
        <tr>
          <td class="table_cell_left"  valign="top"><div style="margin:10px;">E-log authorizations for '{$opr_account}'</div></td>
          <td class="table_cell_right" valign="top">{$opr_account_auth_html}</td>
        </tr>
        <tr>
          <td class="table_cell_left  table_cell_bottom" valign="top"><div style="margin:10px;">Notify by e-mail</div></td>
          <td class="table_cell_right table_cell_bottom" valign="top">{$notify_html}</td>
        </tr>
      </tbody>
    </table>
    </form>
  </div>
</div>
HERE;

		/* Generate a body of a tab displaying a history of previous switches.
		 */
		$html_history = '';
       	$html_history .= DataPortal::table_begin_html(
			array(
				array( 'name' => 'Experiment',  'width' => 105 ),
				array( 'name' => 'Id',          'width' =>  32 ),
				array( 'name' => 'Switch Time', 'width' =>  90 ),
				array( 'name' => 'By User',     'width' => 160 )
			)
		);
		foreach( $regdb->experiment_switches( $instrument->name()) as $experiment_switch ) {

			$exper_id = $experiment_switch['exper_id'];
			$experiment = $regdb->find_experiment_by_id( $exper_id );

			if( is_null( $experiment ))
				die( "fatal internal error when resolving experiment id={$exper_id} in the database" );
			
			$experiment_portal_url = '<a href="index.php?exper_id='.$experiment->id().'" class="link" title="go to Experiment Data Portal">'.$experiment->name().'</a>';
    		$html_history .= DataPortal::table_row_html(
    			array(
    				$experiment_portal_url,
    				$experiment->id(),
    				LusiTime::from64( $experiment_switch['switch_time'] )->toStringShort(),
    				$experiment_switch['requestor_uid']
   				)
    		);
		}
    	$html_history .= DataPortal::table_end_html();
		
		/* Generate two tabs for each instrument: 'Current' and 'History'
		 */
		$instrument_tabs = array();
		array_push(
			$instrument_tabs,
			array('name' => 'Current Experiment',
				  'id'   => 'instrument-'.$instrument->name().'-current',
				  'html' => $html
			)
		);
		array_push(
			$instrument_tabs,
			array('name' => 'History',
				  'id'   => 'instrument-'.$instrument->name().'-history',
				  'html' => $html_history
			)
		);

		/* Add the instrument to the top-level tab.
		 */
		array_push(
   			$tabs,
   			array('name' => $instrument->name(),
   				  'id'   => 'instrument-'.$instrument->name(),
	   			  'html' => DataPortal::tabs_html( "instrument-{$instrument->name()}-contents", $instrument_tabs  )
   			)
   		);
	}

   	/* Print the whole tab and its contents (including sub-tabs).
   	 */
	DataPortal::tabs( "instruments", $tabs );
?>

<!----------------------------------------------------------------->






<?php
    DataPortal::end();
?>
<!--------------------- Document End Here -------------------------->


<?php

} catch( AuthDBException $e ) { print $e->toHtml();
} catch( RegDBException  $e ) { print $e->toHtml();
}

?>
