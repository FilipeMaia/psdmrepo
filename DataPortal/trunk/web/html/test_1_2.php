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
    $num_runs = 0;
    $min_run = null;
    $max_run = null;
    $logbook_runs = $logbook_experiment->runs();
    foreach( $logbook_runs as $run ) {
    	$num_runs++;
  		if( is_null( $min_run )) {
  			$min_run = $run;
  			$max_run = $run;
  		} else {
    		if( $run->num() < $min_run->num()) $min_run = $run;
  			if( $run->num() > $max_run->num()) $max_run = $run;
  		}
    }
    $range_of_runs = ( is_null($min_run) || is_null($max_run)) ? 'no runs taken yet' : $min_run->num().' .. '.$max_run->num();

    $logbook_shifts = $logbook_experiment->shifts();

    $document_title = 'Data Portal of Experiment:';
    $document_subtitle = '<a href="select_experiment.php" title="Switch to another experiment">'.$experiment->instrument()->name().'&nbsp;/&nbsp;'.$experiment->name().'</a>';

    $decorated_experiment_status = '<span style="color:#b0b0b0; font-weight:bold;">NOT ACTIVE</span>';
    $last_experiment_switch = $regdb->last_experiment_switch( $instrument->name());
    if( !is_null( $last_experiment_switch ) && ( $exper_id == $last_experiment_switch['exper_id'] )) $decorated_experiment_status = '<span style="color:#ff0000; font-weight:bold;">ACTIVE</span>';
	$decorated_experiment_contact = DataPortal::decorated_experiment_contact_info( $experiment );
	$decorated_min_run = is_null($min_run) ? '' : $min_run->begin_time()->toStringShort().' (<b>run '.$min_run->num().'</b>)';
	$decorated_max_run = is_null($max_run) ? '' : $max_run->end_time()->toStringShort().' (<b>run '.$max_run->num().'</b>)';
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
	$can_manage_group = false;
    foreach( array_keys( $regdb->experiment_specific_groups()) as $g ) {
    	
  		if( $g == $experiment->POSIX_gid()) {
	        $can_manage_group = RegDBAuth::instance()->canManageLDAPGroup( $g );
            break;
        }
    }
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
  If you're the PI then please contact us by sending a e-mail request to <b>pcds-help</b> (at SLAC). Otherwise
  contact the PI of the experiment.
</div>
HERE;
    }

    $elog_recent_workarea =<<<HERE

<div id="el-l-mctrl">
  <div style="float:left;">
    <div style="font-weight:bold; margin-bottom:4px;">Show runs:</div>
    <div id="el-l-rs-selector">
      <input type="radio" id="el-l-rs-on"  name="show_runs" value="on"  checked="checked" /><label for="el-l-rs-on"  >On</label>
      <input type="radio" id="el-l-rs-off" name="show_runs" value="off"                   /><label for="el-l-rs-off" >Off</label>
    </div>
  </div>
  <div style="float:right;" class="el-l-auto">
    <div style="font-weight:bold; margin-bottom:4px;">Autorefresh:</div>
    <div id="el-l-refresh-selector" style="float:left;">
      <input type="radio" id="el-l-refresh-on"  name="refresh" value="on"  checked="checked" /><label for="el-l-refresh-on"  >On</label>
      <input type="radio" id="el-l-refresh-off" name="refresh" value="off"                   /><label for="el-l-refresh-off" >Off</label>
    </div>
    <div style="float:left; margin-left:10px;">
      <select id="el-l-refresh-interval">
        <option>2</option>
        <option>5</option>
        <option>10</option>
      </select>
      s.
    </div>
    <div style="float:left; margin-left:10px;">
      <button id="el-l-refresh" title="check if there are new updates">Check for updates</button>
    </div>
    <div style="clear:both;"></div>
  </div>
  <div style="clear:both;"></div>
</div>
<div class="el-wa">
  <div class="el-ms-mctrl">
    <button id="el-l-expand"     title="click a few times to expand the whole tree">Expand++</button>
    <button id="el-l-collapse"   title="each click will collapse the tree to the previous level of detail">Collapse--</button>
    <button id="el-l-viewattach" title="view attachments of expanded messages">View Attachments</button>
    <button id="el-l-hideattach" title="hide attachments of expanded messages">Hide Attachments</button>
  </div>  
  <div class="el-ms-info" id="el-l-ms-info" style="float:right;"></div><div style="clear:both;"></div>
  <div class="el-ms" id="el-l-ms"></div>
</div>

HERE;

    $num_tags = 3;
    $tags_html = '';
    for( $i = 0; $i < $num_tags; $i++) {
    	$select_tag_html = "<option> select tag </option>\n";
    	foreach( $logbook_experiment->used_tags() as $tag ) {
    		$select_tag_html .= "<option>{$tag}</option>\n";
    	}
    	$tags_html .=<<<HERE
<div style="width: 100%;">
  <select id="elog-tags-library-{$i}">{$select_tag_html}</select>
  <input type="text" id="elog-tag-name-{$i}" name="tag_name_{$i}"  value="" size=16 title="type new tag here or select a known one from the left" />
  <input type="hidden" id="elog-tag-value-{$i}" name="tag_value_{$i}" value="" />
</div>

HERE;
    }

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
            <input type="text" id="el-p-runnum" value="{$max_run->num()}" size=2 />
            <span id="el-p-runnum-error" style="color:red;"></span>
          </div>
          <div style="clear:both;"></div>
        </div>
      </div>
      <form id="elog-form-post" enctype="multipart/form-data" action="/apps-dev/logbook/NewFFEntry4portal.php" method="post">
        <input type="hidden" name="id" value="{$experiment->id()}" />
        <input type="hidden" name="scope" value="" />
        <input type="hidden" name="run_id" value="" />
        <input type="hidden" name="shift_id" value="" />
        <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />
        <input type="hidden" name="num_tags" value="{$num_tags}" />
        <input type="hidden" name="onsuccess" value="" />
        <input type="hidden" name="relevance_time" value="" />
        <textarea name="message_text" rows="12" cols="64" style="padding:4px; margin-top:5px;" title="the first line of the message body will be used as its subject" ></textarea>
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
    <form id="elog-form-search" action="/apps-dev/logbook/Search.php" method="get">

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
      <div style="clear:both;"></div>
    </form>
  </div>
  <div style="float:right; margin-left:4px;"><button id="elog-search-reset">Reset form</button></div>
  <div style="float:right; margin-left:5px;"><button id="elog-search-submit">Submit</button></div>
  <div style="clear:both;"></div>
</div>
<div class="el-wa">
  <div class="el-ms-mctrl">
    <button id="el-s-expand"     title="click a few times to expand the whole tree">Expand++</button>
    <button id="el-s-collapse"   title="each click will collapse the tree to the previous level of detail">Collapse--</button>
    <button id="el-s-viewattach" title="view attachments of expanded messages">View Attachments</button>
    <button id="el-s-hideattach" title="hide attachments of expanded messages">Hide Attachments</button>
  </div>  
  <div class="el-ms-info" id="el-s-ms-info" style="float:right;"></div><div style="clear:both;"></div>
  <div class="el-ms" id="el-s-ms"></div>
</div>
HERE;
    
?>


<!------------------- Document Begins Here ------------------------->

<!DOCTYPE html"> 
<html>
<head>

<title>Test</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui-1.8.7.custom.css" rel="Stylesheet" />
<link type="text/css" href="css/ELog4test1_2.css" rel="Stylesheet" />
<link type="text/css" href="css/Exper4test1_2.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery-1.5.1.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.8.7.custom.min.js"></script>
<script type="text/javascript" src="js/Utilities.js"></script>
<script type="text/javascript" src="js/ELog4test1_2.js"></script>
<script type="text/javascript" src="js/Exper4test1_2.js"></script>

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
    height: 124px;
    border-bottom: 1px solid #a0a0a0;
    background-color: #e0e0e0;
  }
  #p-top-header {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 87px;
    background-color: #ffffff;
    z-index: 9999;
  }
  #p-left {
    position: absolute;
    left: 0;
    top: 125px;
    width: 200px;
    overflow: auto;
  }
  #p-splitter {
    position: absolute;
    left: 200px;
    top: 125px;
    width: 1px;
    overflow: none;
    cursor: e-resize;
    border-left: 1px solid #a0a0a0;
    border-right: 1px solid #a0a0a0;
  }
  #p-bottom {
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
    top:125px;
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
  font-size: 75%;
  height: 25px;
  width: 100%;
  border: 0;
  padding: 0;
  border-bottom: 1px solid #c0c0c0;
}

#p-context {
  padding-top: 12px;
  padding-left: 15px;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 75%;
}
#p-search {
  padding-top: 2px;
  padding-right: 10px;
  font-size: 75%;
}

div.m-item {

  margin-left: 2px;
  margin-top: 3px;

  padding: 3px;
  padding-left: 10px;
  padding-right: 10px;

  background: #DFEFFC url(/jquery/css/custom-theme/images/ui-bg_glass_85_dfeffc_1x400.png) 50% 50% repeat-x;

  border-top: 1px solid #c0c0c0;
  border-bottom: 1px solid #c0c0c0;
  border-right: 1px solid #c0c0c0;

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
  margin-left: 6px;
  float: left;
}
.m-item-next {
  float: left;
}
.m-item-last {
  float: right;
  margin-right: 6px;
  border-left: 1px solid #c0c0c0;
  border-right: 0;
}
.m-item-end {
  clear: both;
}
div.m-select {
  font-weight: bold;
  background: #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
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
elog.rrange = '<?=$range_of_runs?>';
elog.min_run = <?=(is_null($min_run)?'null':$min_run->num())?>;
elog.max_run = <?=(is_null($max_run)?'null':$max_run->num())?>;
<?php
	foreach( $logbook_runs as $run ) echo "elog.runs[{$run->num()}]={$run->id()};\n";
	foreach( $logbook_shifts as $shift ) echo "elog.shifts['{$shift->begin_time()->toStringShort()}']={$shift->id()};\n";
?>
elog.editor = <?=(LogBookAuth::instance()->canEditMessages( $experiment->id())?'true':'false')?>;

exper.posix_group = '<?=$experiment->POSIX_gid()?>';

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
	var el = document.getElementById( current_tab );
	if (el) {
		var html = document.getElementById(current_tab).innerHTML;
		var pfcopy = window.open("about:blank");
		pfcopy.document.write('<html xmlns="http://www.w3.org/1999/xhtml">');
		pfcopy.document.write('<head><meta http-equiv="Content-Type" content="text/html; charset=windows-1252" />');
		pfcopy.document.write('<link rel="stylesheet" type="text/css" href="css/default.css" />');
		pfcopy.document.write('<link type="text/css" href="css/portal.css" rel="Stylesheet" />');
		pfcopy.document.write('<link type="text/css" href="css/ELog.css" rel="Stylesheet" />');
		pfcopy.document.write('<style type="text/css"> .not4print { display:none; }	</style>');
		pfcopy.document.write('<title>Data Portal of Experiment: '+elog.instr+' / '+elog.exp+'</title></head><body><div class="maintext">');
		pfcopy.document.write(html);
		pfcopy.document.write("</div></body></html>");
		pfcopy.document.close();
	}
}



/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */
var applications = null;
var current_application = 'experiment';

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
				v_item_selected($('#v-menu > #elog > #search').next().children('.v-item#simple'));
				application.select('search','simple');
				application.simple_search($('#p-search-elog-text').val());
				break;
			}
		}
	}
	$('#p-search-elog-text').keyup(function(e) { if(($('#p-search-elog-text').val() != '') && (e.keyCode == 13)) simple_search(); });

	applications = {
		'p-appl-experiment' : exper,
		'p-appl-elog'       : elog,
		'p-appl-datafiles'  : new p_appl_datafiles(),	// TODO: implement it the same wa as for e-Log
		'p-appl-hdf5'       : new p_appl_hdf5(),		// TODO: implement it the same wa as for e-Log
		'p-appl-help'       : new p_appl_help()			// TODO: implement it the same wa as for e-Log
	};

	/* TODO: Fix this, either by recpecting the GET parameter of this PHP script
	 *       or by analyzing the code. (better not to do the last thing).
	 */
	current_application = applications['p-appl-experiment'];
	current_application.select_default();
	v_item_selected($('#v-menu > #experiment').children('.v-item#summary'));
});

/* TODO: Merge these application objects into statically create JavaScript
 *       objects like elog. This should result in a better code encapsulation
 *       and management.
 *
 *       Implement similar objects (JS + CSS) for other applications.
 */

function p_appl_datafiles() {
	var that = this;
	var context2_default = {
		'' : ''
	};
	this.name = 'datafiles';
	this.full_name = 'Data Files';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	$('#p-center > #application-workarea > #datafiles').html('<center>The work area for data files</center>');
	$('#p-left > #v-menu > #datafiles').html('<center>The menu area of the data files</center>');

	return this;
}

function p_appl_hdf5() {
	var that = this;
	var context2_default = {
		'' : ''
	};
	this.name = 'hdf5';
	this.full_name = 'HDF5 Translation';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	$('#p-center > #application-workarea > #hdf5').html('<center>The workarea of the HDF5 translation</center>');
	$('#p-left > #v-menu > #hdf5').html('<center>The workarea of the HDF5 translation</center>');

	return this;
}

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
  <div id="header">
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
    <div class="m-item m-item-first m-select" id="p-appl-experiment">Experment</div>
    <div class="m-item m-item-next" id="p-appl-elog">e-Log</div>
    <div class="m-item m-item-next" id="p-appl-datafiles">Data Files</div>
    <div class="m-item m-item-next" id="p-appl-hdf5">HDF5 Translation</div>
    <div class="m-item m-item-last" id="p-appl-help">Help</div>
    <div class="m-item-end"></div>
  </div>
  <div>
    <div id="p-context" style="float:left"></div>
    <div id="p-search" style="float:right">
      search e-log: <input type="text" id="p-search-elog-text" value="" size=16 title="text to search in e-Log"  style="font-size:80%; padding:1px; margin-top:6px;" />
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
      <div class="v-group" id="recent">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
        <div class="link" style="float:left;" >Recent (Live)</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group-members v-group-members-visible">
        <div class="v-item v-item-first" id="20">
          <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
          <div class="link" style="float:left;" >20</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="100">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >100</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="12h">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >shift</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="24h">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >24 hrs</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="7d">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >7 days</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >everything</div>
          <div style="clear:both;"></div>
        </div>
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
      <div class="v-group" id="search">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >Search</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group-members v-group-members-hidden">
        <div class="v-item v-item-first" id="simple">
          <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
          <div class="link" style="float:left;" >simple</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="advanced">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >advanced</div>
          <div style="clear:both;"></div>
        </div>
      </div>
      <div class="v-item" id="browse">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Browse</div>
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
      <div class="v-item" id="subscribe">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Subscribe</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="datafiles" class="hidden">
      No menu for data files yet
    </div>

    <div id="hdf5" class="hidden">
      No menu for HDF5 translation yet
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
    <div id="elog-recent"   class="application-workarea hidden"><?php echo $elog_recent_workarea ?></div>
    <div id="elog-post"     class="application-workarea hidden"><?php echo $elog_post_workarea ?></div>
    <div id="elog-search"   class="application-workarea hidden"><?php echo $elog_search_workarea ?></div>
    <div id="datafiles"     class="application-workarea hidden"></div>
    <div id="hdf5"          class="application-workarea hidden"></div>
    <div id="help"          class="application-workarea hidden"></div>
  </div>
  <div id="popupdialogs" style="display:none;"></div>
</div>
</body>

</html>

<!--------------------- Document End Here -------------------------->


<?php

} catch( FileMgrException $e ) { print $e->toHtml();
} catch( LogBookException $e ) { print $e->toHtml();
} catch( RegDBException   $e ) { print $e->toHtml();
} catch( LogBookException $e ) { print $e->toHtml();
} catch( AuthDBException  $e ) { print $e->toHtml();
}

?>
