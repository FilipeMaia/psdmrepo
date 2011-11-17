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
use RegDB\RegDBAuth;
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

/* If a specific application is requested by a user then open
 * the corresponding tab.
 */
$known_apps = array(
	'experiment' => True,
	'elog'       => True,
	'datafiles'  => True,
	'hdf'        => True );

$select_app = 'experiment';
$select_app_context1 = '';

if( isset( $_GET['app'] )) {
	$app_path = explode( ':', strtolower( trim( $_GET['app'] )));
	$app = $app_path[0];
	if( array_key_exists( $app, $known_apps )) {
		$select_app = $app;
		if( count($app_path) > 1 ) $select_app_context1 = $app_path[1];
	}
}

/* Parse optional parameters which may be used by applications. The parameters
 * will be passed directly into applications for further analysis (syntax, values,
 * etc.).
 */
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

    $can_manage_group = false;
    foreach( array_keys( $regdb->experiment_specific_groups()) as $g ) {
    	
  		if( $g == $experiment->POSIX_gid()) {
	        $can_manage_group = RegDBAuth::instance()->canManageLDAPGroup( $g );
            break;
        }
    }
	$has_data_access =
		$can_manage_group ||
		$regdb->is_member_of_posix_group( 'ps-data', $auth_svc->authName()) ||
		$regdb->is_member_of_posix_group( $logbook_experiment->POSIX_gid(), $auth_svc->authName()) ||
		$regdb->is_member_of_posix_group( 'ps-'.strtolower( $instrument->name()), $auth_svc->authName());

	$has_elog_access = LogBookAuth::instance()->canRead( $logbook_experiment->id());

	$num_runs = $logbook_experiment->num_runs();
    $min_run  = $logbook_experiment->find_first_run();
    $max_run  = $logbook_experiment->find_last_run();

    $logbook_shifts = $logbook_experiment->shifts();

    $document_title = 'Web Portal of Experiment:';
    $document_subtitle = '<a href="select_experiment.php" title="Switch to another experiment">'.$experiment->instrument()->name().'&nbsp;/&nbsp;'.$experiment->name().'</a>';

    $decorated_experiment_status = '<span style="color:#b0b0b0; font-weight:bold;">NOT ACTIVE</span>';
    $last_experiment_switch = $regdb->last_experiment_switch( $instrument->name());
    if( !is_null( $last_experiment_switch ) && ( $exper_id == $last_experiment_switch['exper_id'] )) $decorated_experiment_status = '<span style="color:#ff0000; font-weight:bold;">ACTIVE</span>';
	$decorated_experiment_contact = DataPortal::decorated_experiment_contact_info( $experiment );
	$decorated_min_run = is_null($min_run) ? 'n/a' : $min_run->begin_time()->toStringShort().' (<b>run '.$min_run->num().'</b>)';
	$decorated_max_run = is_null($max_run) ? 'n/a' : $max_run->begin_time()->toStringShort().' (<b>run '.$max_run->num().'</b>)';
	$experiment_group_members     = "<table><tbody>\n";
    $experiment_group_members .= '<tr><td class="table_cell table_cell_left"></td><td class="table_cell table_cell_right"></td></tr>';
	foreach( $experiment->group_members() as $m ) {
    	$uid   = $m['uid'];
    	$gecos = $m['gecos'];
        $experiment_group_members .= '<tr><td class="table_cell table_cell_left">'.$uid.'</td><td class="table_cell table_cell_right">'.$gecos.'</td></tr>';
	}
	$experiment_group_members .= "</tbody></table>\n";
    $experiment_summary_workarea =<<<HERE

<table><tbody>
  <tr><td class="table_cell table_cell_left">Id</td>
      <td class="table_cell table_cell_right">{$experiment->id()}</td></tr>
  <tr><td class="table_cell table_cell_left">Status</td>
      <td class="table_cell table_cell_right">{$decorated_experiment_status}</td></tr>
  <tr><td class="table_cell table_cell_left">Total # of runs taken</td>
      <td class="table_cell table_cell_right">{$num_runs}</td></tr>
  <tr><td class="table_cell table_cell_left">First run</td>
      <td class="table_cell table_cell_right">{$decorated_min_run}</td></tr>
  <tr><td class="table_cell table_cell_left">Last run</td>
      <td class="table_cell table_cell_right">{$decorated_max_run}</td></tr>
  <tr><td class="table_cell table_cell_left">Description</td>
      <td class="table_cell table_cell_right"><pre style="background-color:#e0e0e0; padding:0.5em;">{$experiment->description()}</pre></td></tr>
  <tr><td class="table_cell table_cell_left">Contact</td>
      <td class="table_cell table_cell_right">{$decorated_experiment_contact}</td></tr>
  <tr><td class="table_cell table_cell_left">Leader</td>
      <td class="table_cell table_cell_right">{$experiment->leader_Account()}</td></tr>
  <tr><td class="table_cell table_cell_left table_cell_bottom" valign="top">POSIX Group</td>
      <td class="table_cell table_cell_right table_cell_bottom">
        <table cellspacing=0 cellpadding=0><tbody>
          <tr><td style="font-size: 75%;" valign="top">{$experiment->POSIX_gid()}</td>
              <td>&nbsp;</td>
              <td><span id="exp-group-toggler" class="toggler ui-icon ui-icon-triangle-1-e" title="click to see/hide the list of members"></span>
                  <div id="exp-group-members" class="exp-group-members-hidden">{$experiment_group_members}</div>
              </td></tr>
        </tbody></table>
      </td></tr>
</tbody></table>

HERE;

    if($can_manage_group) {

    	$experiment_manage_group_workarea =<<<HERE

<div style="float:left; margin-left:10px; margin-right:20px; margin-bottom:40px; padding-right:30px; border-right: 1px solid #c0c0c0;">
  <div style="height:55px;">
    <div style="float:left; font-size: 300%; font-family: Times, sans-serif;"><b>{$experiment->POSIX_gid()}</b></div>
    <div style="float:left; margin-left:10px; padding-top:4px;"><button id="exp-m-g-refresh">Refresh</button></div>
    <div style="clear:both;"></div>
  </div>
  <div id="exp-m-g-members-stat"></div>
  <div id="exp-m-g-members" style="margin-top:4px;"></div>
</div>
<div style="float:left; padding-left:10px;">
  <div style="height:55px;">
    <div style="float:left; font-weight:bold; padding-top:8px;">Search users:</div>
    <div style="float:left; margin-left:5px;"><input type="text" style="padding:2px;" id="exp-m-g-string2search" value="" size=16 title="enter the pattern to search then press RETURN" /></div>
    <div style="float:left; margin-left:10px; font-weight:bold; padding-top:6px;">by:</div>
    <div style="float:left; margin-left:5px; padding-top:3px;" id="exp-m-g-scope">
      <input type="radio" id="exp-m-g-uid"   name="scope" value="uid"                         /><label for="exp-m-g-uid"  >UID</label>
      <input type="radio" id="exp-m-g-gecos" name="scope" value="gecos"                       /><label for="exp-m-g-gecos">name</label>
      <input type="radio" id="exp-m-g-both"  name="scope" value="uid_gecos" checked="checked" /><label for="exp-m-g-both" >both</label>
    </div>
    <div style="clear:both;"></div>
  </div>
  <div id="exp-m-g-users-stat"></div>
  <div id="exp-m-g-users" style="margin-top:4px;"></div>
</div>
<div style="clear:both;"></div>

HERE;
    } else {
	    $experiment_manage_group_workarea =<<<HERE
<br><br>
<center>
  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">
    A c c e s s &nbsp; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">
  We're sorry! Your SLAC UNIX account <b>{$auth_svc->authName()}</b> has no proper permissions to manage POSIX
  group <b>{$experiment->POSIX_gid()}</b> associated with the experiment. Normally we assign this task to
  the PI of the experiment. The PI may also delegate the role to another member of the experiment.
  If you're the PI then please contact us by sending an e-mail request to <b>pcds-help</b> (at SLAC). Otherwise
  contact the PI of the experiment.
</div>
HERE;
    }

    if( $has_elog_access ) {

    	$elog_recent_workarea =<<<HERE

<div id="el-l-mctrl">
  <div style="float:left;">
    <div style="font-weight:bold; margin-bottom:2px;">Messages:</div>
    <select name="messages" style="font-size:90%; padding:1px;" title="Select non-blank option to select how many events to load">
      <option selected="selected">100</option>
      <option>shift</option>
      <option>24 hrs</option>
      <option>7 days</option>
      <option>everything</option>
    </select>
  </div>
  <div style="float:left; margin-left:20px;">
    <div style="font-weight:bold; margin-bottom:4px;">Include runs:</div>
    <div id="el-l-rs-selector">
      <input type="radio" id="el-l-rs-on"  name="show_runs" value="on" checked="checked"/><label for="el-l-rs-on" >Yes</label>
      <input type="radio" id="el-l-rs-off" name="show_runs" value="off"/><label for="el-l-rs-off">No</label>
    </div>
  </div>
  <div style="float:left; margin-left:20px;">
    <div style="font-weight:bold; margin-bottom:4px;">Include deleted:</div>
    <div id="el-l-dm-selector">
      <input type="radio" id="el-l-dm-on"  name="show_deleted" value="on" checked="checked"/><label for="el-l-dm-on" >Yes</label>
      <input type="radio" id="el-l-dm-off" name="show_deleted" value="off" /><label for="el-l-dm-off">No</label>
    </div>
  </div>
  <div style="float:right;" class="el-l-auto">
    <div style="float:left;">
      <div style="font-weight:bold; margin-bottom:4px;">Autorefresh:</div>
      <div id="el-l-refresh-selector" style="float:left;">
        <input type="radio" id="el-l-refresh-on"  name="refresh" value="on"  checked="checked" /><label for="el-l-refresh-on"  >On</label>
        <input type="radio" id="el-l-refresh-off" name="refresh" value="off"                   /><label for="el-l-refresh-off" >Off</label>
      </div>
      <div style="float:left; margin-left:10px; font-size:80%;">
        <select id="el-l-refresh-interval">
          <option>2</option>
          <option>5</option>
          <option>10</option>
        </select>
        s.
      </div>
      <div style="clear:both;"></div>
    </div>
    <div style="float:right; margin-left:10px;">
      <button id="el-l-refresh" title="check if there are new messages or runs">Refresh</button>
    </div>    
    <div style="clear:both;"></div>
  </div>
  <div style="clear:both;"></div>
</div>
<div class="el-wa">
  <div class="el-ms-info" id="el-l-ms-info" style="float:left;">&nbsp;</div>
  <div class="el-ms-info" id="el-l-ms-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div class="el-ms-mctrl">
    <button id="el-l-expand"     title="click a few times to expand the whole tree">Expand++</button>
    <button id="el-l-collapse"   title="each click will collapse the tree to the previous level of detail">Collapse--</button>
    <button id="el-l-viewattach" title="view attachments of expanded messages">View Attachments</button>
    <button id="el-l-hideattach" title="hide attachments of expanded messages">Hide Attachments</button>
    <button id="el-l-reverse"    title="show days and messages within each day in reverse order">Show in Reverse Order</button>
  </div>  
  <div class="el-ms" id="el-l-ms"></div>
</div>

HERE;

    $used_tags = $logbook_experiment->used_tags();
    $select_tag_html = "<option> select tag </option>\n";
    foreach( $used_tags as $tag )
    	$select_tag_html .= "<option>{$tag}</option>\n";

    $tags_html = '';
    $num_tags  = 3;
    for( $i = 0; $i < $num_tags; $i++)
    	$tags_html .=<<<HERE
<div style="width: 100%;">
  <select id="elog-tags-library-{$i}">{$select_tag_html}</select>
  <input type="text" class="elog-tag-name" id="elog-tag-name-{$i}" name="tag_name_{$i}" value="" size=16 title="type new tag here or select a known one from the left" />
  <input type="hidden" id="elog-tag-value-{$i}" name="tag_value_{$i}" value="" />
</div>

HERE;
    
    $today = date("Y-m-d");
    $now   = "00:00:00";
    $shifts_html = '';
    foreach( $logbook_shifts as $shift )
    	$shifts_html .= "<option>{$shift->begin_time()->toStringShort()}</option>";
    
    $elog_post_workarea =<<<HERE
<div id="el-p">
  <div>
    <div style="float:left;">
      <div>
        <div id="el-p-message4experiment" class="hidden">
          <div style="font-weight:bold; padding-top:5px; padding-bottom:5px;">Message for the experiment:</div>
        </div>
        <div id="el-p-message4shift">
          <div style="float:left; font-weight:bold; padding-top:5px;">Message for shift</div>
          <div style="float:left; margin-left:5px;">
            <select id="el-p-shift">{$shifts_html}</select>
          </div>
          <div style="clear:both;"></div>
        </div>
        <div id="el-p-message4run" class="hidden">
          <div style="float:left; font-weight:bold; padding-top:5px;">Message for run</div>
          <div style="float:left; margin-left:5px;">
            <input type="text" id="el-p-runnum" value="" size=4 />
          </div>
          <div style="clear:both;"></div>
        </div>
      </div>
      <form id="elog-form-post" enctype="multipart/form-data" action="../logbook/NewFFEntry4portalJSON.php" method="post">
        <input type="hidden" name="id" value="{$experiment->id()}" />
        <input type="hidden" name="scope" value="" />
        <input type="hidden" name="run_num" value="" />
        <input type="hidden" name="shift_id" value="" />
        <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />
        <input type="hidden" name="num_tags" value="{$num_tags}" />
        <input type="hidden" name="onsuccess" value="" />
        <input type="hidden" name="relevance_time" value="" />
        <textarea name="message_text" rows="12" cols="64" style="padding:4px; margin-top:5px;"
                  title="TIPS:\nThe first line of the message body will be used as its subject.\nUse 'run NNN' to post for the run"></textarea>
        <div style="margin-top: 10px;">
          <div style="float:left;">
            <div style="font-weight:bold;">Author:</div>
            <input type="text" name="author_account" value="{$auth_svc->authName()}" size=32 style="padding:2px; margin-top:5px; width:100%;" />
            <div style="margin-top:20px;"> 
              <div style="font-weight:bold;">Tags:</div>
              <div style="margin-top:5px;">{$tags_html}</div>
            </div>
          </div>
          <div style="float:left; margin-left:30px;"> 
            <div style="font-weight:bold;">Attachments:</div>
            <div id="el-p-as" style="margin-top:5px;">
              <div>
                <input type="file" name="file2attach_0" onchange="elog.post_add_attachment()" />
                <input type="hidden" name="file2attach_0" value="" />
              </div>
            </div>
          </div>
          <div style="clear:both;"></div>
        </div>
      </form>
    </div>
    <div style="float:right; margin-left:4px;"><button id="elog-post-reset">Reset form</button></div>
    <div style="float:right; margin-left:5px;"><button id="elog-post-submit">Submit</button></div>
    <div style="clear:both;"></div>
  </div>
  <div style="margin-top:10px;">
    <div style="float:left;">
      <div style="font-weight:bold;">Adjust Post Time:</div>
      <div id="el-p-relevance-selector" style="margin-top:8px;">
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
</div>

HERE;

    $time_title =
        "Valid format:\n".
        "\t".LusiTime::now()->toStringShort()."\n".
        "Also the (case neutral) shortcuts are allowed:\n".
        "\t'b' - the begin time of the experiment\n".
        "\t'e' - the end time of the experiment\n".
        "\t'm' - month (-31 days) ago\n".
        "\t'w' - week (-7 days) ago\n".
        "\t'd' - day (-24 hours) ago\n".
        "\t'y' - since yesterday (at 00:00:00)\n".
        "\t't' - today (at 00:00:00)\n".
        "\t'h' - an hour (-60 minutes) ago";

    $tag2search_html = '<option></option>';
    foreach( $logbook_experiment->used_tags() as $tag ) {
    	$tag2search_html .= "<option>{$tag}</option>\n";
    }
    $author2search_html = '<option></option>';
    foreach( $logbook_experiment->used_authors() as $author ) {
    	$author2search_html .= "<option>{$author}</option>\n";
    }
    $elog_search_workarea =<<<HERE
<div id="el-s-ctrl">
  <div style="float:left;">
    <form id="elog-form-search" action="../logbook/Search.php" method="get">
      <div style="float:left;">
        <div style="font-weight:bold;">Text to search:</div>
        <div><input type="text" name="text2search" value="" size=24 style="font-size:90%; padding:1px; margin-top:5px; width:100%;" /></div>
        <div style="float:left; margin-top:5px;">
          <div style="font-weight:bold;">Tag:</div>
          <div style="margin-top:5px;"><select name="tag" style="font-size:90%; padding:1px;">{$tag2search_html}</select></div>
        </div>
        <div style="float:left; margin-top:5px; margin-left:10px;">
          <div style="font-weight:bold;">Posted by:</div>
          <div style="margin-top:5px;"><select name="author" style="font-size:90%; padding:1px;">{$author2search_html}</select></div>
        </div>
        <div style="clear:both;"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="font-weight:bold; margin-bottom:5px;">Search in:</div>
        <div><input type="checkbox" name="search_in_messages" value="Message" checked="checked" /> message body</div>
        <div><input type="checkbox" name="search_in_tags" value="Tag" /> tags</div>
        <div><input type="checkbox" name="search_in_values" value="Value" /> tag values</div>
        <div><input type="checkbox" name="search_in_deleted" value="Deleted" checked="checked" /> deleted messages</div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="font-weight:bold; margin-bottom:5px;">Posted at:</div>
        <div><input type="checkbox" name="posted_at_instrument" value="Instrument" /> instrument</div>
        <div><input type="checkbox" name="posted_at_experiment" value="Experiment" checked="checked" /> experiment</div>
        <div><input type="checkbox" name="posted_at_shifts" value="Shifts" checked="checked" /> shifts</div>
        <div><input type="checkbox" name="posted_at_runs" value="Runs" checked="checked" /> runs</div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div title="{$time_title}">
          <div style="font-weight:bold;">Begin Time:</div>
          <div><input type="text" name="begin" value="" size=24 style="font-size:90%; padding:1px; margin-top:5px;"/></div>
        </div>
        <div style="margin-top:5px;" title="{$time_title}">
          <div style="font-weight:bold;">End Time:</div>
          <div><input type="text" name="end" value="" size=24 style="font-size:90%; padding:1px; margin-top:5px;"/></div>
        </div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="font-weight:bold;">Around Run(s):</div>
        <div>
          <input type="text" name="runs" value="" size=5 style="font-size:90%; padding:1px; margin-top:5px;"
                 title="Enter a run number or a range of runs
where to look for messages. For a single run
put its number. For a range the correct syntax is: 12-35
Make sure the Begin and End time limits are not used!"/>
        </div>
      </div>
      <div style="clear:both;"></div>
    </form>
  </div>
  <div style="float:right; margin-left:4px;"><button id="elog-search-reset">Reset form</button></div>
  <div style="float:right; margin-left:5px;"><button id="elog-search-submit">Submit</button></div>
  <div style="clear:both;"></div>
</div>
<div class="el-wa">
  <div class="el-ms-info" id="el-s-ms-info" style="float:left;">&nbsp;</div>
  <div class="el-ms-info" id="el-s-ms-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div class="el-ms-mctrl">
    <button id="el-s-expand"     title="click a few times to expand the whole tree">Expand++</button>
    <button id="el-s-collapse"   title="each click will collapse the tree to the previous level of detail">Collapse--</button>
    <button id="el-s-viewattach" title="view attachments of expanded messages">View Attachments</button>
    <button id="el-s-hideattach" title="hide attachments of expanded messages">Hide Attachments</button>
    <button id="el-s-reverse"    title="show days and messages within each day in reverse order">Show in Reverse Order</button>
  </div>  
  <div class="el-ms" id="el-s-ms"></div>
</div>
HERE;

    	$elog_shifts_workarea =<<<HERE
<div id="el-sh-ctrl">
  <div style="float:right; margin-left:5px;"><button id="el-sh-refresh" title="click to refresh the shifts list">Refresh</button></div>
  <div style="clear:both;"></div>
</div>
<div id="el-sh-wa">
  <div class="el-sh-info" id="el-sh-info" style="float:left;">&nbsp;</div>
  <div class="el-sh-info" id="el-sh-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div style="margin-top:10px; font-size:80%;">
    <table style="font-size:120%;"><tbody>
      <tr>
        <td><b>Sort by:</b></td>
        <td><select name="sort" style="padding:1px;">
              <option>begin</option>
              <option>runs</option>
              <option>duration</option>
            </select></td>
        <td><div style="width:20px;"></div></td>
        <td><button id="el-sh-reverse">Show in Reverse Order</button></td>
        <td><div style="width:10px;"></div></td>
        <td><button id="el-sh-new-begin">Begin New Shift</button></td></tr>
    </tbody></table>
    <div id="el-sh-new-wa" class="el-sh-new-hdn">
      <div style="float:left;">
        <form id="elog_new_shift_form" action="../logbook/CreateShift.php" method="post">
          <div style="float:left;">
            <input type="hidden" name="id" value="{$exper_id}" />
            <input type="hidden" name="actionSuccess" value="" />
            <input type="hidden" name="max_crew_size" value="5" />
            <input type="hidden" name="author" value="{$auth_svc->authName()}" />
            <table><tbody>
              <tr>
                <td class="ctable_cell ctable_cell_left" valign="center">Leader:&nbsp;&nbsp;</td>
                <td class="ctable_cell ctable_cell_right"><input type="text" name="leader"  size=20 style="padding:1px;" value="{$auth_svc->authName()}"/></td></tr>
              <tr>
                <td class="ctable_cell ctable_cell_left ctable_cell_bottom" valign="center">Crew:&nbsp;&nbsp;</td>
                <td class="ctable_cell ctable_cell_right"><input type="text" name="member0" value="" size=20 style="padding:1px;" /></td></tr>
              <tr>
                <td class="ctable_cell ctable_cell_left ctable_cell_bottom"></td>
                <td class="ctable_cell ctable_cell_right"><input type="text" name="member1" value="" size=20 style="padding:1px;" /></td></tr>
              <tr>
                <td class="ctable_cell ctable_cell_left ctable_cell_bottom"></td>
                <td class="ctable_cell ctable_cell_right"><input type="text" name="member2" value="" size=20 style="padding:1px;" /></td></tr>
              <tr>
                <td class="ctable_cell ctable_cell_left ctable_cell_bottom"></td>
                <td class="ctable_cell ctable_cell_right"><input type="text" name="member3" value="" size=20 style="padding:1px;" /></td></tr>
              <tr>
                <td class="ctable_cell ctable_cell_left ctable_cell_bottom"></td>
                <td class="ctable_cell ctable_cell_right"><input type="text" name="member4" value="" size=20 style="padding:1px;" /></td></tr>
            </tbody></table>
          </div>
          <div style="float:left; margin-left:20px;">
            <table><tbody>
              <tr>
                <td class="ctable_cell ctable_cell_left  ctable_cell_bottom" valign="center">Goals:&nbsp;&nbsp;</td>
                <td class="ctable_cell ctable_cell_right ctable_cell_bottom"><textarea rows="10" cols="64" name="goals" style="padding:4px;"></textarea ></td></tr>
            </tbody></table>
          </div>
        </form>
      </div>
      <div style="float:left; margin-left:20px;"><button id="el-sh-new-submit">Submit</button></div>
      <div style="float:left; margin-left:10px;"><button id="el-sh-new-cancel">Cancel</button></div>
      <div style="clear:both;"></div>
    </div>
  </div>
  <div id="el-sh-list"></div>
</div>
HERE;

    	$elog_runs_workarea =<<<HERE
<div id="el-r-ctrl">
  <div style="float:right; margin-left:5px;"><button id="el-r-refresh" title="click to refresh the runs list">Refresh</button></div>
  <div style="clear:both;"></div>
</div>
<div id="el-r-wa">
  <div class="el-r-info" id="el-r-info" style="float:left;">&nbsp;</div>
  <div class="el-r-info" id="el-r-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div style="margin-top:10px; font-size:80%;">
    <table style="font-size:120%;"><tbody>
      <tr>
        <td><b>Sort by:</b></td>
        <td><select name="sort" style="padding:1px;">
              <option>run</option>
              <option>duration</option>
            </select></td>
        <td><div style="width:20px;"></div></td>
        <td><button id="el-r-reverse">Show in Reverse Order</button></td>
      </tr>
    </tbody></table>
  </div>
  <div id="el-r-list"></div>
</div>
HERE;

    	$elog_attachments_workarea =<<<HERE
<div id="el-at-ctrl">
  <div style="float:right; margin-left:5px;"><button id="el-at-refresh" title="click to refresh the attachments list">Refresh</button></div>
  <div style="clear:both;"></div>
</div>
<div id="el-at-wa">
  <div class="el-at-info" id="el-at-info" style="float:left;">&nbsp;</div>
  <div class="el-at-info" id="el-at-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div style="margin-top:10px; font-size:80%;">
    <table style="font-size:120%;"><tbody>
      <tr>
        <td><b>Sort by:</b></td>
        <td><select name="sort" style="padding:1px;">
              <option>posted</option>
              <option>author</option>
              <option>name</option>
              <option>type</option>
              <option>size</option>
            </select></td>
        <td><div style="width:20px;"></div></td>
        <td><b>View as:</b></td>
        <td><select name="view" style="padding:1px;">
              <option>table</option>
              <option>thumbnails</option>
              <option>hybrid</option>
            </select></td>
        <td><div style="width:20px;"></div></td>
        <td><b>Mix-in runs:</b></td>
        <td><select name="runs" style="padding:1px;">
              <option>no</option>
              <option>yes</option>
              </select></td>
        <td><div style="width:20px;"></div></td>
        <td><button id="el-at-reverse">Show in Reverse Order</button></td>
      </tr>
    </tbody></table>
  </div>
  <div id="el-at-list"></div>
</div>
HERE;

    	$elog_subscribe_workarea =<<<HERE

<div class="el-subscribe" id="el-subscribed" style="display:none;">
  <h3 style="font-size:140%;">Your subscription:</h3>
  <div style="padding-left:10px;">
    <p align="justify">Your SLAC UNIX account <b>{$auth_svc->authName()}</b> is already subscribed to receive automated e-mail
       notifications on various e-log events of this experiment. The notifications are sent
       onto your SLAC e-mail address:</p>
    <div style="padding-left: 10px;">
      <b>{$auth_svc->authName()}@slac.stanford.edu</b>
      <button style="margin-left:10px;" id="el-unsubscribe" title="stop receiving automatic notifications">Unsubscribe</button>
    </div>
    <p align="justify">You may subscribe or unsubscribe at any time. You'll receive a confirmation
       message shortly after unsubscribing.</p>
  </div>
</div>
<div class="el-subscribe" id="el-unsubscribed" style="display:none;">
  <h3 style="font-size:140%;">Your subscription:</h3>
  <div style="padding-left:10px;">
    <p align="justify">At the moment, your SLAC UNIX account <b>{$auth_svc->authName()}</b> is not subscribed to receive automated
       e-mail notifications on various e-log events of this experiment. If you choose to do so
       then notifications will be sent onto your SLAC e-mail address:</p>
    <div style="padding-left: 10px;">
      <b>{$auth_svc->authName()}@slac.stanford.edu</b>
      <button style="margin-left:10px;" id="el-subscribe" title="start receiving automatic notifications">Subscribe</button>
    </div>
    <p align="justify">You may subscribe or unsubscribe at any time. You'll receive a confirmation
       message shortly after subscribing. If your primary e-mail address differs from
       the one mentioned above then make sure you set proper e-mail forwarding from
       SLAC to your primary address. Also check if your SPAM filter won't be blocking
       messages with the following properties:</p>
    <div>
      <table><tbody>
        <tr><td class="table_cell table_cell_left" >From</td>
            <td class="table_cell table_cell_right">LCLS E-Log [apache@slac.stanford.edu]</td></tr>
        <tr><td class="table_cell table_cell_left  table_cell_bottom">Subject</td>
            <td class="table_cell table_cell_right table_cell_bottom">[ {$instrument->name()} / {$experiment->name()} ]</td></tr>
      </tbody></table>
    </div>
    <p align="justify">And here is the final remark: do not try to reply to e-log messages! Injecting
       replies into e-Log stream via e-mail transport is not presently implemented.
      We're still debating whether this would be a useful feature to have in the Portal.</p>
  </div>
</div>
<div id="el-subscribe-all"></div>

HERE;
    	
    } else {

    	$no_elog_access_message =<<<HERE
<br><br>
<center>
  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">
    A c c e s s &nbsp; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">
  We're sorry! Your SLAC UNIX account <b>{$auth_svc->authName()}</b> has no proper permissions to access
  this page. The page access is reserved to members of group <b>{$experiment->POSIX_gid()}</b> associated with the experiment,
  <b>{$instrument->name()}</b> instrument scientists, and <b>PCDS</b> operations crew.
  If you want to be a member of the group then contact directly the PI of the experiment.
  In all other cases please contact us by sending an e-mail request to <b>pcds-help</b> (at SLAC).
</div>
HERE;

    	$elog_recent_workarea      = $no_elog_access_message;
    	$elog_post_workarea        = $no_elog_access_message;
    	$elog_search_workarea      = $no_elog_access_message;
    	$elog_shifts_workarea      = $no_elog_access_message;
    	$elog_runs_workarea        = $no_elog_access_message;
    	$elog_attachments_workarea = $no_elog_access_message;
    	$elog_subscribe_workarea   = $no_elog_access_message;

    }
    if( $has_data_access ) {

    	$datafiles_summary_workarea =<<<HERE
<div id="datafiles-summary-ctrl">
  <div style="float:right;"><button id="datafiles-summary-refresh" title="click to refresh the summary information">Refresh</button></div>
  <div style="clear:both;"></div>
</div>
<div id="datafiles-summary-wa">
  <div class="datafiles-info" id="datafiles-summary-info" style="float:right;">&nbsp;</div><div style="clear:both;"></div>
  <table><tbody>
    <tr><td class="table_cell table_cell_left"># of runs</td>
        <td class="table_cell table_cell_right" id="datafiles-summary-runs">no data</td></tr>
    <tr><td class="table_cell table_cell_left">First run #</td>
        <td class="table_cell table_cell_right" id="datafiles-summary-firstrun">no data</td></tr>
    <tr><td class="table_cell table_cell_left">Last run #</td>
        <td class="table_cell table_cell_right" id="datafiles-summary-lastrun">no data</td></tr>
    <tr><td class="table_cell table_cell_left" valign="center">XTC</td>
        <td class="table_cell table_cell_right">
          <table cellspacing=0 cellpadding=0><tbody>
            <tr><td class="table_cell table_cell_left">Size [GB]</td>
                <td class="table_cell table_cell_right" id="datafiles-summary-xtc-size">no data</td></tr>
            <tr><td class="table_cell table_cell_left"># of files</td>
                <td class="table_cell table_cell_right" id="datafiles-summary-xtc-files">no data</td></tr>
            <tr><td class="table_cell table_cell_left">Archived to HPSS</td>
                <td class="table_cell table_cell_right" id="datafiles-summary-xtc-archived">no data</td></tr>
            <tr><td class="table_cell table_cell_left  table_cell_bottom">Available on disk</td>
                <td class="table_cell table_cell_right table_cell_bottom" id="datafiles-summary-xtc-disk">no data</td></tr>
            </tbody></table>
        </td></tr>
    <tr><td class="table_cell table_cell_left table_cell_bottom" valign="center">HDF5</td>
        <td class="table_cell table_cell_right table_cell_bottom">
          <table cellspacing=0 cellpadding=0><tbody>
            <tr><td class="table_cell table_cell_left">Size [GB]</td>
                <td class="table_cell table_cell_right" id="datafiles-summary-hdf5-size">no data</td></tr>
            <tr><td class="table_cell table_cell_left"># of files</td>
                <td class="table_cell table_cell_right" id="datafiles-summary-hdf5-files">no data</td></tr>
            <tr><td class="table_cell table_cell_left">Archived to HPSS</td>
                <td class="table_cell table_cell_right" id="datafiles-summary-hdf5-archived">no data</td></tr>
            <tr><td class="table_cell table_cell_left  table_cell_bottom">Available on disk</td>
                <td class="table_cell table_cell_right table_cell_bottom" id="datafiles-summary-hdf5-disk">no data</td></tr>
            </tbody></table>
        </td></tr>
  </tbody></table>
</div>
HERE;

    	$datafiles_files_workarea =<<<HERE
<div id="datafiles-files-ctrl">
  <div style="float:left;">
    <div style="float:left;">
      <div style="font-weight:bold;">Search runs:</div>
      <div style="margin-top:5px;">
        <input type="text" name="runs" style="font-size:90%; padding:1px;" title="Put a range of runs to activate the filter. Use the following syntax: 1,3,5,10-20,211"></input>
      </div>
    </div>
    <div style="float:left; margin-left:20px;">
      <div style="font-weight:bold;">Types:</div>
      <div style="margin-top:5px;">
        <select name="types" style="font-size:90%; padding:1px;" title="Select non-blank option to activate the filter">
          <option>any</option>
          <option>XTC</option>
          <option>HDF5</option>
        </select>
      </div>
    </div>
    <div style="float:left; margin-left:20px;">
      <div style="font-weight:bold;">Created:</div>
      <div style="margin-top:5px;">
        <select name="created" style="font-size:90%; padding:1px;" title="Select non-blank option to activate the filter">
          <option>any</option>
          <option>1 hr</option>
          <option>12 hrs</option>
          <option>24 hrs</option>
          <option>7 days</option>
          <option>1 month</option>
          <option>1 year</option>
          </select>
      </div>
    </div>
    <div style="float:left; margin-left:20px;">
      <div style="font-weight:bold;">Checksum:</div>
      <div style="margin-top:5px;">
        <select name="checksum" style="font-size:90%; padding:1px;" title="Select non-blank option to activate the filter">
          <option>any</option>
          <option>none</option>
          <option>is known</option>
        </select>
      </div>
    </div>
    <div style="float:left; margin-left:20px;">
      <div style="font-weight:bold;">Archived:</div>
      <div style="margin-top:5px;">
        <select name="archived" style="font-size:90%; padding:1px;" title="Select non-blank option to activate the filter">
          <option>any</option>
          <option>yes</option>
          <option>no</option>
        </select>
      </div>
    </div>
    <div style="float:left; margin-left:20px;">
      <div style="font-weight:bold;">On disk:</div>
      <div style="margin-top:5px;">
        <select name="local" style="font-size:90%; padding:1px;" title="Select non-blank option to activate the filter">
          <option>any</option>
          <option>yes</option>
          <option>no</option>
        </select>
      </div>
    </div>
    <div style="clear:both;"></div>
  </div>
  <div style="float:right; margin-left:5px;"><button id="datafiles-files-refresh" title="click to refresh the file list according to the last filter">Refresh</button></div>
  <div style="clear:both;"></div>
</div>
<div id="datafiles-files-wa">
  <div class="datafiles-info" id="datafiles-files-info" style="float:left;">&nbsp;</div>
  <div class="datafiles-info" id="datafiles-files-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div style="margin-top:10px;" >
    <table style="font-size:80%;"><tbody>
      <tr>
        <td style="font-size:120%;"><b>Sort by:</b></td>
        <td>
          <select name="sort" style="padding:1px;">
            <option>run</option>
            <option>name</option>
            <option>type</option>
            <option>size</option>
            <option>created</option>
            <option>archived</option>
            <option>disk</option>
          </select>
        </td>
        <td><div style="font-size:120%; padding-left:10px;"><button id="datafiles-files-reverse">Show in Reverse Order</button></div></td>
        <td style="font-size:120%; padding-left: 20px;"><b>Columns:</b></td>
        <td><input type="checkbox" name="type" /></td><td>Type</td>
        <td><input type="checkbox" name="size" /></td><td>Size</td>
        <td><input type="checkbox" name="created" /></td><td>Created</td>
      </tr><tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td><input type="checkbox" name="archived" /></td><td>Archived</td>
        <td><input type="checkbox" name="local" /></td><td>On disk</td>
        <td><input type="checkbox" name="checksum" /></td><td>Checksum</td>
      </tr>
    </tbody></table>
  </div>
  <div id="datafiles-files-list" style="margin-top:5px;"></div>
</div>
HERE;

    	$hdf_manage_workarea = <<<HERE
<div id="hdf-manage-ctrl">
  <div style="float:left;">
    <div style="float:left;">
      <div style="font-weight:bold;">Search runs:</div>
      <div style="margin-top:5px;">
        <input type="text" name="runs" style="font-size:90%; padding:1px;" title="Put a range of runs to activate the filter. Use the following syntax: 1,3,5,10-20,211"></input>
      </div>
    </div>
    <div style="float:left; margin-left:20px;">
      <div style="font-weight:bold;">Translation status:</div>
      <div style="margin-top:5px;">
        <select name="status" style="font-size:90%; padding:1px;" title="Select non-blank option to activate the filter">
          <option>any</option>
          <option>FINISHED</option>
          <option>FAILED</option>
          <option>TRANSLATING</option>
          <option>QUEUED</option>
          <option>NOT-TRANSLATED</option>
          </select>
      </div>
    </div>
    <div style="clear:both;"></div>
  </div>
  <div style="float:right; margin-left:5px;"><button id="hdf-manage-refresh" title="click to refresh the file list according to the last filter">Refresh</button></div>
  <div style="clear:both;"></div>
</div>
<div id="hdf-manage-wa">
  <div class="hdf-info" id="hdf-manage-info" style="float:left;">&nbsp;</div>
  <div class="hdf-info" id="hdf-manage-updated" style="float:right;">&nbsp;</div>
  <div style="clear:both;"></div>
  <div id="hdf-manage-list" style="margin-top:5px;"></div>
</div>
HERE;

    	$hdf_history_workarea = <<<HERE
HERE;

    	$hdf_translators_workarea = <<<HERE
HERE;

    } else {

    	$no_data_access_message =<<<HERE
<br><br>
<center>
  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">
    A c c e s s &nbsp; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">
  We're sorry! Your SLAC UNIX account <b>{$auth_svc->authName()}</b> has no proper permissions to access
  this page. The page access is reserved to members of group <b>{$experiment->POSIX_gid()}</b> associated with the experiment.
  Please, contact the PI to request your account to be included into the group.
</div>
HERE;

		$datafiles_summary_workarea = $no_data_access_message;
		$datafiles_files_workarea   = $no_data_access_message;
		$hdf_manage_workarea        = $no_data_access_message;
    	$hdf_history_workarea       = $no_data_access_message;
		$hdf_translators_workarea   = $no_data_access_message;
    }

?>


<!------------------- Document Begins Here ------------------------->

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>

<title>Web Portal of Experiment: <?php echo $instrument->name()?> / <?php echo $experiment->name()?></title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui-1.8.7.custom.css" rel="Stylesheet" />

<link type="text/css" href="css/common.css" rel="Stylesheet" />
<link type="text/css" href="css/ELog.css" rel="Stylesheet" />
<link type="text/css" href="css/Exper.css" rel="Stylesheet" />
<link type="text/css" href="css/Data.css" rel="Stylesheet" />
<link type="text/css" href="css/Hdf.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery-1.5.1.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.8.7.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="js/Utilities.js"></script>
<script type="text/javascript" src="js/ELog.js"></script>
<script type="text/javascript" src="js/Exper.js"></script>
<script type="text/javascript" src="js/Data.js"></script>
<script type="text/javascript" src="js/Hdf.js"></script>

<!----------- Window layout styles and supppot actions ----------->

<style type="text/css">

  body {
    margin: 0;
    padding: 0;
  }
  #p-top {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 129px;
    background-color: #e0e0e0;
  }
  #p-top-header {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 92px;
    background-color: #ffffff;
  }
  #p-top-title {
    width: 100%;
    height: 61px;
  }
  #p-context-header {
    width: 100%;
    height: 36px;
    background-color: #E0E0E0;
    border-bottom: 1px solid #0b0b0b;
  }
  #p-left {
    position: absolute;
    left: 0;
    top: 130px;
    width: 200px;
    overflow: auto;
  }
  #p-splitter {
    position: absolute;
    left: 200px;
    top: 130px;
    width: 1px;
    overflow: none;
    cursor: e-resize;
    border-left: 1px solid #a0a0a0;
    border-right: 1px solid #a0a0a0;
  }
  #p-bottom {
    z-index: 100;
    position: absolute;
    left: 0;
    bottom: 0;
    height: 20px;
    width: 100%;
    background-color: #a0a0a0;
    border-top: 1px solid #c0c0c0;
  }
  #p-status {
    padding: 2px;
    font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
    font-size: 75%;
  }
  #p-center {
    position: relative;
    top:130px;
    margin: 0px 0px 20px 203px;
    overflow: auto;
    background-color: #ffffff;
    border-left: 1px solid #a0a0a0;
  }

</style>

<script type="text/javascript">

function resize() {
	$('#p-left').height($(window).height()-125-20);
	$('#p-splitter').height($(window).height()-125-20);
	$('#p-center').height($(window).height()-125-20);
}

/* Get mouse position relative to the document.
 */
function getMousePosition(e) {

	var posx = 0;
	var posy = 0;
	if (!e) var e = window.event;
	if (e.pageX || e.pageY) 	{
		posx = e.pageX;
		posy = e.pageY;
	}
	else if (e.clientX || e.clientY) 	{
		posx = e.clientX + document.body.scrollLeft
			+ document.documentElement.scrollLeft;
		posy = e.clientY + document.body.scrollTop
			+ document.documentElement.scrollTop;
	}
	return {'x': posx, 'y': posy };
}

function move_split(e) {
	var pos = getMousePosition(e);
	$('#p-left').css('width', pos['x']);
	$('#p-splitter').css('left', pos['x']);
	$('#p-center').css('margin-left', pos['x']+3);
}

$(function() {

	resize();

	var mouse_down = false;

	$('#p-splitter').mousedown (function(e) { mouse_down = true; return false; });

	$('#p-left'    ).mousemove(function(e) { if(mouse_down) move_split(e); });
	$('#p-center'  ).mousemove(function(e) { if(mouse_down) move_split(e); });

	$('#p-left'    ).mouseup   (function(e) { mouse_down = false; });
	$('#p-splitter').mouseup   (function(e) { mouse_down = false; });
	$('#p-center'  ).mouseup   (function(e) { mouse_down = false; });
});


</script>


<!--------------------------------------------------------->


<style type="text/css">

#p-title,
#p-subtitle {
  font-family: "Times", serif;
  font-size: 32px;
  font-weight: bold;
  text-align: left;
}
#p-subtitle {
  margin-left: 10px;
  color: #0071bc;
}
#p-login {
  font-size: 70%;
  font-family: Arial, Helvetica, Verdana, Sans-Serif;
}

a, a.link {
  text-decoration: none;
  font-weight: bold;
  color: #0071bc;
}
a:hover, a.link:hover {
  color: red;
}

span.toggler {
  background-color: #ffffff;
  border: 1px solid #c0c0c0;
  border-radius: 4px;
  -moz-border-radius: 4px;
  cursor: pointer;
}

#p-menu {
  font-family: Arial, sans-serif;
  font-size: 14px;
  height: 32px;
  width: 100%;
  border: 0;
  padding: 0;
  /*
  border-bottom: 1px solid #c0c0c0;
  */
}

#p-context {
  margin-left: 0px;
  padding-top: 10px;
  padding-left: 10px;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 12px;
}
#p-search, #p-post {
  padding-top: 2px;
  padding-right: 10px;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 11px;
}

div.m-item {

  margin-left: 3px;
  margin-top: 3px;

  padding: 5px;
  padding-left: 10px;
  padding-right: 10px;

  background: #DFEFFC url(/jquery/css/custom-theme/images/ui-bg_glass_85_dfeffc_1x400.png) 50% 50% repeat-x;

  color: #0071BC;

  border-top: 2px solid #c0c0c0;
  /*border-bottom: 1px solid #c0c0c0;*/
  border-right: 2px solid #c0c0c0;

  border-radius: 5px;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;

  -moz-border-radius: 5px;
  -moz-border-radius-bottomleft: 0;
  -moz-border-radius-bottomright: 0;

  cursor: pointer;
}

div.m-item:hover {
  background: #d0e5f5 url(/jquery/css/custom-theme/images/ui-bg_glass_75_d0e5f5_1x400.png) 50% 50% repeat-x;
}
div.m-item-first {
  margin-left: 0px;
  float: left;

  border-top-left-radius: 0;

  -moz-border-radius-topleft: 0;
}
.m-item-next {
  float: left;
}
.m-item-last {
  float: left;
/*
  float: right;
  margin-right: 0px;
  border-left: 1px solid #c0c0c0;
  border-right: 0;
  */
}
.m-item-end {
  clear: both;
}
div.m-select {
  font-weight: bold;
  background: #e0e0e0;
  /*
  border-bottom: 1px solid #e0e0e0;
  */
}

#v-menu {
  width: 100%;
  height: 100%;
  background: url('img/menu-bg-gradient-4.png') repeat;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 75%;
}
#menu-title {
  height: 10px;
}
div.v-item {
  padding: 4px;
  padding-left: 10px;
  cursor: pointer;
}
div.v-item:hover {
  background:#f0f0f0;
}
.v-select {
  font-weight: bold;
}
.v-group {
  padding: 4px;
  padding-left: 10px;
  cursor: pointer;
}
.v-group-members {
  padding: 4px;
  padding-left: 20px;
}
.v-group-members-hidden {
  display: none;
}
.v-group-members-visible {
  display: block;
}
.application-workarea {
  overflow: auto;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 75%;
}
.hidden  { display: none; }
.visible { display: block; }

</style>

<script type="text/javascript">

/* ----------------------------------------
 * Authentication and authorization context
 * ----------------------------------------
 */
var auth_is_authenticated="<?php echo $auth_svc->isAuthenticated()?>";
var auth_type="<?php echo $auth_svc->authType()?>";
var auth_remote_user="<?php echo $auth_svc->authName()?>";

var auth_webauth_token_creation="<?php echo $_SERVER['WEBAUTH_TOKEN_CREATION']?>";
var auth_webauth_token_expiration="<?php echo $_SERVER['WEBAUTH_TOKEN_EXPIRATION']?>";

function refresh_page() {
    window.location = "<?php echo $_SERVER['REQUEST_URI']?>";
}

/*
 * Session expiration timer for WebAuth authentication.
 */
var auth_timer = null;
function auth_timer_restart() {
    if( auth_is_authenticated && ( auth_type == 'WebAuth' ))
        auth_timer = window.setTimeout('auth_timer_event()', 1000 );
}

var auth_last_secs = null;
function auth_timer_event() {

    var auth_expiration_info = document.getElementById('auth_expiration_info');
    var now = mktime();
    var seconds = auth_webauth_token_expiration - now;
    if( seconds <= 0 ) {
        $('#popupdialogs').html(
        	'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
        	'Your WebAuth session has expired. Press <b>Ok</b> or use <b>Refresh</b> button'+
        	'of the browser to renew your credentials.</p>'
        );
        $('#popupdialogs').dialog({
        	resizable: false,
        	modal: true,
        	buttons: {
        		'Ok': function() {
        			$(this).dialog('close');
        			refresh_page();
        		}
        	},
        	title: 'Session Expiration Notification'
        });
        return;
    }
    var hours_left   = Math.floor(seconds / 3600);
    var minutes_left = Math.floor((seconds % 3600) / 60);
    var seconds_left = Math.floor((seconds % 3600) % 60);

    var hours_left_str = hours_left;
    if( hours_left < 10 ) hours_left_str = '0'+hours_left_str;
    var minutes_left_str = minutes_left;
    if( minutes_left < 10 ) minutes_left_str = '0'+minutes_left_str;
    var seconds_left_str = seconds_left;
    if( seconds_left < 10 ) seconds_left_str = '0'+seconds_left_str;

    auth_expiration_info.innerHTML=
        '<b>'+hours_left_str+':'+minutes_left_str+'.'+seconds_left_str+'</b>';

    auth_timer_restart();
}

function logout() {
	$('#popupdialogs').html(
		'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
    	'This will log yout out from the current WebAuth session. Are you sure?</p>'
	 );
	$('#popupdialogs').dialog({
		resizable: false,
		modal: true,
		buttons: {
			"Yes": function() {
				$( this ).dialog('close');
	            document.cookie = 'webauth_wpt_krb5=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
	            document.cookie = 'webauth_at=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
	            refresh_page();
			},
			Cancel: function() {
				$(this).dialog('close');
			}
		},
		title: 'Session Logout Warning'
	});
}

/* --------------------------------------------------- 
 * The starting point where the JavaScript code starts
 * ---------------------------------------------------
 */
$(document).ready(function(){
	auth_timer_restart();
});


/* -----------------------------------------
 *             GLOBAL VARIABLES
 * -----------------------------------------
 */
elog.author = '<?=$auth_svc->authName()?>';
elog.exp_id = '<?=$exper_id?>';
elog.exp = '<?=$experiment->name()?>';
elog.instr = '<?=$experiment->instrument()->name()?>';
<?php
	foreach( $logbook_shifts as $shift ) echo "elog.shifts['{$shift->begin_time()->toStringShort()}']={$shift->id()};\n";
?>
elog.editor = <?=(LogBookAuth::instance()->canEditMessages( $experiment->id())?'true':'false')?>;
<?php
	foreach( $used_tags as $tag ) echo "elog.used_tags.push('{$tag}');\n";
?>

// Calback function for e-log to be used after successfully posting
// a new message.
//
// TODO: Consider a more general mechanism allowing applications to invoke
//       callbacks in a context of this thread. That would probably require
//       the current thread to be re-implemented as an object, which would
//       open a possibility to pass that objkect's 'this' into applications
//       as 'parent' (a provider of sertain services).
//
elog.post_onsuccess = function() {
	for(var id in applications) {
		var a = applications[id];
		if(a.name == 'elog') {
			$('#p-menu').children('#'+id).each(function() {	m_item_selected(this); });
			v_item_selected($('#v-menu > #elog > .v-item#recent'));
			a.select('recent');
			break;
		}
	}
};

exper.posix_group = '<?=$experiment->POSIX_gid()?>';

datafiles.exp_id = '<?=$exper_id?>';
hdf.exp_id = '<?=$exper_id?>';

var select_app = '<?=$select_app?>';
var select_app_context1 = '<?=$select_app_context1?>';

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

/* ----------------------------------------------
 *             CONTEXT MANAGEMENT
 * ----------------------------------------------
 */
var current_tab = 'applications';

function set_current_tab( tab ) {
	current_tab = tab;
}
/*
function set_context(app) {
	var ctx = '<b>'+app.full_name+'</b> &gt;';
	if(app.context1) ctx += ' <b>'+app.context1+'</b>';
	if(app.context2) ctx += ' &gt; <b>'+app.context2+'</b>';
	if(app.context3) ctx += ' &gt; <b>'+app.context3+'</b>';;
	$('#p-context').html(ctx);
}
*/
function set_context(app) {
	var ctx = app.full_name+' &gt;';
	if(app.context1) ctx += ' '+app.context1;
	if(app.context2) ctx += ' &gt; '+app.context2;
	if(app.context3) ctx += ' &gt; '+app.context3;;
	$('#p-context').html(ctx);
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

function printer_friendly() {
	if( current_application != null ) {
		var wa_id = current_application.name;
		if(current_application.context1 != '') wa_id += '-'+current_application.context1;
		$('#p-center .application-workarea#'+wa_id).printElement({
			leaveOpen: true,
			printMode: 'popup',
			printBodyOptions: {
            	styleToAdd:'font-size:12px;'
            }
		});
	}	
}



/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */
var applications = {
	'p-appl-experiment' : exper,
	'p-appl-elog'       : elog,
	'p-appl-datafiles'  : datafiles,
	'p-appl-hdf'        : hdf,
	'p-appl-help'       : new p_appl_help()			// TODO: implement it the same wa as for e-Log
};

var current_application = null;

function v_item_group(item) {
	var parent = $(item).parent();
	if(parent.hasClass('v-group-members')) return parent.prev();
	return null;
}

/* Event handler for application selections from the top-level menu bar:
 * - fill set the current application context.
 */
function m_item_selected(item) {

	current_application = applications[item.id];

	$('.m-select').removeClass('m-select');
	$(item).addClass('m-select');
	$('#p-left > #v-menu .visible').removeClass('visible').addClass('hidden');
	$('#p-left > #v-menu > #'+current_application.name).removeClass('hidden').addClass('visible');

	$('#p-center .application-workarea.visible').removeClass('visible').addClass('hidden');
	var wa_id = current_application.name;
	if(current_application.context1 != '') wa_id += '-'+current_application.context1;
	$('#p-center .application-workarea#'+wa_id).removeClass('hidden').addClass('visible');

	current_application.select_default();
	if(current_application.context2 == '')
		v_item_selected($('#v-menu > #'+current_application.name).children('.v-item#'+current_application.context1));
	else
		v_item_selected($('#v-menu > #'+current_application.name+' > #'+current_application.context1).next().children('.v-item#'+current_application.context2));
	
	set_context(current_application);
}

/* Event handler for vertical menu group selections:
 * - only show/hide children (if any).
 */
function v_group_selected(group) {
	var toggler = $(group).children('.ui-icon');
	if(toggler.hasClass('ui-icon-triangle-1-s')) {
		toggler.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
		$(group).next().removeClass('v-group-members-visible').addClass('v-group-members-hidden');
	} else {
		toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
		$(group).next().removeClass('v-group-members-hidden').addClass('v-group-members-visible');
	}
}

/* Event handler for vertical menu item (actual commands) selections:
 * - dim the poreviously active item (and if applies - its group)
 * - hightlight the new item (and if applies - its group)
 * - change the current context
 * - execute the commands
 * - switch the work area (make the old one invisible, and the new one visible)
 */
function v_item_selected(item) {
	var item = $(item);
	if($(item).hasClass('v-select')) return;

	$('#'+current_application.name).find('.v-item.v-select').each(function(){
		$(this).children('.ui-icon').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
		$(this).removeClass('v-select');
		var this_group = v_item_group(this);
		if(this_group != null) this_group.removeClass('v-select');
	});

	$(item).children('.ui-icon').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
	$(item).addClass('v-select');

	var group = v_item_group(item);
	if(group != null) {

		/* Force the group to unwrap
		 *
		 * NOTE: This migth be needed of the current method is called out of
		 *       normal sequence.
		 *
		 * TODO: Do it "right" when refactoring the menu classes.
		 */
		var toggler = $(group).children('.ui-icon');
		if(!toggler.hasClass('ui-icon-triangle-1-s')) {
			toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(group).next().removeClass('v-group-members-hidden').addClass('v-group-members-visible');
		}

		/* Hide the older work area
		 */
		var wa_id = current_application.name;
		if(current_application.context1 != '') wa_id += '-'+current_application.context1;
		$('#p-center > #application-workarea > #'+wa_id).removeClass('visible').addClass('hidden');

		/* Activate new application
		 */
		group.addClass('v-select');
		current_application.select(group.attr('id'), item.attr('id'));

		/* display the new work area
		 */
		wa_id = current_application.name;
		if(current_application.context1 != '') wa_id += '-'+current_application.context1;
		$('#p-center > #application-workarea > #'+wa_id).removeClass('hidden').addClass('visible');

	} else {

		/* Hide the older work area
		 */
		var wa_id = current_application.name;
		if(current_application.context1 != '') wa_id += '-'+current_application.context1;
		$('#p-center > #application-workarea > #'+wa_id).removeClass('visible').addClass('hidden');

		current_application.select(item.attr('id'), null);

		/* display the new work area
		 */
		wa_id = current_application.name;
		if(current_application.context1 != '') wa_id += '-'+current_application.context1;
		$('#p-center > #application-workarea > #'+wa_id).removeClass('hidden').addClass('visible');
	}
	set_context(current_application);
}

$(function() {

	$('.m-item' ).click(function() { m_item_selected (this); });
	$('.v-group').click(function() { v_group_selected(this); });
	$('.v-item' ).click(function() { v_item_selected (this); });

	function simple_search() {
		for(var id in applications) {
			var application = applications[id];
			if(application.name == 'elog') {
				$('#p-menu').children('#'+id).each(function() {	m_item_selected(this); });
				v_item_selected($('#v-menu > #elog').children('.v-item#search'));
				application.select('search');
				application.simple_search($('#p-search-elog-text').val());
				break;
			}
		}
	}
	$('#p-search-elog-text').keyup(function(e) { if(($('#p-search-elog-text').val() != '') && (e.keyCode == 13)) simple_search(); });
	function simple_post() {
		for(var id in applications) {
			var application = applications[id];
			if(application.name == 'elog') {
				$('#p-menu').children('#'+id).each(function() {	m_item_selected(this); });
				v_item_selected($('#v-menu > #elog > #post').next().children('.v-item#experiment'));
				application.select('post','experiment');
				application.simple_post4experiment($('#p-post-elog-text').val());
				break;
			}
		}
	}
	$('#p-post-elog-text').keyup(function(e) {
		if(($('#p-post-elog-text').val() != '') && (e.keyCode == 13)) simple_post();
	});

	// Finally, activate the selected application.
	//
	for(var id in applications) {
		var application = applications[id];
		if(application.name == select_app) {
			$('#p-menu').children('#p-appl-'+select_app).each(function() { m_item_selected(this); });
			if( '' != select_app_context1 ) {
				v_item_selected($('#v-menu > #'+select_app+' > #'+select_app_context1));
				application.select(select_app_context1);
			}
		}
	}
});

/* TODO: Merge these application objects into statically create JavaScript
 *       objects like elog. This should result in a better code encapsulation
 *       and management.
 *
 *       Implement similar objects (JS + CSS) for other applications.
 */

function p_appl_help() {
	var that = this;
	var context2_default = {
		'' : ''
	};
	this.name = 'help';
	this.full_name = 'Help';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	$('#p-center > #application-workarea > #help').html('<center>The help area</center>');
	$('#p-left > #v-menu > #help').html('<center>The workarea of the HDF5 translation</center>');

	return this;
}

</script>

</head>

<body onresize="resize()">

<div id="p-top">
<div id="p-top-header">
  <div id="p-top-title">
    <div style="float:left;  padding-left:15px; padding-top:10px;">
      <span id="p-title"><?php echo $document_title?></span>
      <span id="p-subtitle"><?php echo $document_subtitle?></span>
    </div>
    <div style="float:right; padding-right:4px;">
      <table><tbody><tr>
        <td valign="bottom">
          <div style="float:right; margin-right:10px;" class="not4print"><a href="javascript:printer_friendly('tabs-experiment')" title="Printer friendly version of this page"><img src="img/PRINTER_icon.gif" style="border-radius: 5px;" /></a></div>
          <div style="clear:both;" class="not4print"></div>
        </td>
        <td>
          <table id="p-login"><tbody>
            <tr>
              <td>&nbsp;</td>
              <td>[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]</td>
            </tr>
            <tr>
              <td>Welcome,&nbsp;</td>
              <td><p><b><?php echo $auth_svc->authName()?></b></p></td>
            </tr>
            <tr>
              <td>Session expires in:&nbsp;</td>
              <td><p id="auth_expiration_info"><b>00:00.00</b></p></td>
            </tr>
          </tbody></table>
        </td>
      </tr></tbody></table>
    </div>
    <div style="clear:both;"></div>
  </div>
  <div id="p-menu">
    <div class="m-item m-item-first m-select" id="p-appl-experiment">Experiment</div>
    <div class="m-item m-item-next" id="p-appl-elog">e-Log</div>
    <div class="m-item m-item-next" id="p-appl-datafiles">File Manager</div>
    <div class="m-item m-item-next" id="p-appl-hdf">HDF5 Translation</div>
    <div class="m-item m-item-last" id="p-appl-help">Help</div>
    <div class="m-item-end"></div>
  </div>
  <div id="p-context-header">
    <div id="p-context" style="float:left"></div>
    <div id="p-search" style="float:right">
      search e-log: <input type="text" id="p-search-elog-text" value="" size=16 title="enter text to search in e-Log, then press RETURN to proceed"  style="font-size:80%; padding:1px; margin-top:6px;" />
    </div>
    <div id="p-post" style="float:right">
      post in e-log: <input type="text" id="p-post-elog-text" value="" size=32 title="enter text to post in e-Log, then press RETURN to proceed"  style="font-size:80%; padding:1px; margin-top:6px;" />
    </div>
    <div style="clear:both;"></div>
  </div>
</div>
</div>
<div id="p-left">

  <div id="v-menu">

    <div id="menu-title"></div>

    <div id="experiment" class="visible">
      <div class="v-item" id="summary">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Summary</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group" id="manage">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >Manage</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group-members v-group-members-hidden">
        <div class="v-item v-item-first" id="group">
          <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
          <div class="link" style="float:left;" >POSIX group</div>
          <div style="clear:both;"></div>
        </div>
      </div>
    </div>

    <div id="elog" class="hidden">
      <div class="v-item" id="recent">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
        <div class="link" style="float:left;" >Recent (Live)</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group" id="post">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >Post</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group-members v-group-members-hidden">
        <div class="v-item v-item-first" id="experiment">
          <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
          <div class="link" style="float:left;" >for experiment</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="shift">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >for shift</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="run">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >for run</div>
          <div style="clear:both;"></div>
        </div>
      </div>
      <div class="v-item" id="search">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >Search</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="shifts">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Shifts</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="runs">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Runs</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="attachments">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Attachments</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="subscribe">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Subscribe</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="datafiles" class="hidden">
      <div class="v-item" id="summary">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Summary</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="files">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Files</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="hdf" class="hidden">
      <div class="v-item" id="manage">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Manage</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="history">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >History</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="translators">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Translators</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="help" class="hidden">
      No menu for help yet
    </div>

  </div>

</div>

<div id="p-splitter"></div>

<div id="p-bottom">
  <div id="p-status">
    <center>- status bar to be here at some point -</center>
  </div>
</div>

<div id="p-center">
  <div id="application-workarea">
    <div id="experiment-summary" class="application-workarea hidden"><?php echo $experiment_summary_workarea ?></div>
    <div id="experiment-manage"  class="application-workarea hidden"><?php echo $experiment_manage_group_workarea ?></div>
    <div id="elog-recent"        class="application-workarea hidden"><?php echo $elog_recent_workarea ?></div>
    <div id="elog-post"          class="application-workarea hidden"><?php echo $elog_post_workarea ?></div>
    <div id="elog-search"        class="application-workarea hidden"><?php echo $elog_search_workarea ?></div>
    <div id="elog-shifts"        class="application-workarea hidden"><?php echo $elog_shifts_workarea ?></div>
    <div id="elog-runs"          class="application-workarea hidden"><?php echo $elog_runs_workarea ?></div>
    <div id="elog-attachments"   class="application-workarea hidden"><?php echo $elog_attachments_workarea ?></div>
    <div id="elog-subscribe"     class="application-workarea hidden"><?php echo $elog_subscribe_workarea ?></div>
    <div id="datafiles-summary"  class="application-workarea hidden"><?php echo $datafiles_summary_workarea ?></div>
    <div id="datafiles-files"    class="application-workarea hidden"><?php echo $datafiles_files_workarea ?></div>
    <div id="hdf-manage"         class="application-workarea hidden"><?php echo $hdf_manage_workarea ?></div>
    <div id="hdf-history"        class="application-workarea hidden"><?php echo $hdf_history_workarea ?></div>
    <div id="hdf-translators"    class="application-workarea hidden"><?php echo $hdf_translators_workarea ?></div>
    <div id="help"               class="application-workarea hidden"></div>
  </div>
  <div id="popupdialogs" style="display:none;"></div>
</div>
</body>

</html>

<!--------------------- Document End Here -------------------------->


<?php

} catch( FileMgrException  $e ) { print $e->toHtml(); }
  catch( LusiTimeException $e ) { print $e->toHtml(); }
  catch( RegDBException    $e ) { print $e->toHtml(); }
  catch( LogBookException  $e ) { print $e->toHtml(); }
  catch( AuthDBException   $e ) { print $e->toHtml(); }

?>
