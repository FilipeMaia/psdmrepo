<?php

require_once( 'lusitime/lusitime.inc.php' );
require_once( 'authdb/authdb.inc.php' );

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

$document_title = 'PCDS Cable Management:';
$document_subtitle = 'Neo-CAPTAR';

$required_field_html = '<span style="color:red; font-size:110%; font-weight:bold;"> * </span>';

try {

	$authdb = AuthDB::instance();
	$authdb->begin();

?>


<!------------------- Document Begins Here ------------------------->


<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title><?php echo $document_title ?></title>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui-1.8.7.custom.css" rel="Stylesheet" />
<link type="text/css" href="css/common.css" rel="Stylesheet" />
<link type="text/css" href="css/neocaptar.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery-1.5.1.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.8.7.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="js/Utilities.js"></script>
<script type="text/javascript" src="js/neocaptar_projects.js"></script>
<script type="text/javascript" src="js/neocaptar_dictionary.js"></script>
<script type="text/javascript" src="js/neocaptar_search.js"></script>
<script type="text/javascript" src="js/neocaptar_admin.js"></script>


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

#p-menu {
  font-family: Arial, sans-serif;
  font-size: 14px;
  height: 32px;
  width: 100%;
  border: 0;
  padding: 0;
}

#p-context {
  margin-left: 0px;
  padding-top: 10px;
  padding-left: 10px;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 12px;
}
#p-search {
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
}
.m-item-end {
  clear: both;
}
div.m-select {
  font-weight: bold;
  background: #e0e0e0;
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
.application-workarea {
  overflow: auto;
  padding: 20px;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 75%;
}
.application-workarea-instructions {
  padding-bottom: 20px;
  width: 720px;
}
.hidden  { display: none; }
.visible { display: block; }

</style>


<script type="text/javascript">


/* ------------------------------------------------
 *          VERTICAL SPLITTER MANAGEMENT
 * ------------------------------------------------
 */
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

/* ---------------------------------------------
 *          AUTHENTICATION MANAGEMENT
 * ---------------------------------------------
 */
var auth_is_authenticated="<?php echo $authdb->isAuthenticated()?>";
var auth_type="<?php echo $authdb->authType()?>";
var auth_remote_user="<?php echo $authdb->authName()?>";

var auth_webauth_token_creation="<?php echo $_SERVER['WEBAUTH_TOKEN_CREATION']?>";
var auth_webauth_token_expiration="<?php echo $_SERVER['WEBAUTH_TOKEN_EXPIRATION']?>";

function refresh_page() {
    window.location = "<?php echo $_SERVER['REQUEST_URI']?>";
}

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

$(function() {
	auth_timer_restart();
});

/* ----------------------------------------------
 *             CONTEXT MANAGEMENT
 * ----------------------------------------------
 */
var current_tab = '';

function set_current_tab( tab ) {
	current_tab = tab;
}

function set_context(app) {
	var ctx = app.full_name+' &gt;';
	if(app.context) ctx += ' '+app.context;
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

function printer_friendly() {
	if( current_application != null ) {
		var wa_id = current_application.name;
		if(current_application.context != '') wa_id += '-'+current_application.context;
		$('#p-center .application-workarea#'+wa_id).printElement({
			leaveOpen: true,
			printMode: 'popup',
			printBodyOptions: {
            	styleToAdd:'font-size:10px;'
            }
		});
	}	
}

function ask_yes_no( title, msg, on_yes, on_cancel ) {
	$('#popupdialogs').html(
		'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+msg+'</p>'
	 );
	$('#popupdialogs').dialog({
		resizable: false,
		modal: true,
		buttons: {
			"Yes": function() {
				$( this ).dialog('close');
				if( on_yes != null ) on_yes();
			},
			Cancel: function() {
				$(this).dialog('close');
				if( on_cancel != null ) on_cancel();
			}
		},
		title: title
	});
}

function report_error( msg, on_cancel ) {
	$('#popupdialogs').html(
		'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+msg+'</p>'
	 );
	$('#popupdialogs').dialog({
		resizable: false,
		modal: true,
		buttons: {
			Cancel: function() {
				$(this).dialog('close');
				if( on_cancel != null ) on_cancel();
			}
		},
		title: 'Error'
	});
}


/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */
var applications = {
	'p-appl-projects'   : projects,
	'p-appl-dictionary' : dict,
	'p-appl-search'     : search,
	'p-appl-admin'      : admin
};

var current_application = null;

var select_app = 'projects';
var select_app_context = 'search';

/* Event handler for application selections from the top-level menu bar:
 * - fill set the current application context.
 */
function m_item_selected(item) {

	if( current_application == applications[item.id] ) return;
	if(( current_application != null ) && ( current_application != applications[item.id] )) {
		current_application.if_ready2giveup( function() {
			m_item_selected_impl(item);
		});
		return;
	}
	m_item_selected_impl(item);
}

function m_item_selected_impl(item) {

	current_application = applications[item.id];

	$('.m-select').removeClass('m-select');
	$(item).addClass('m-select');
	$('#p-left > #v-menu .visible').removeClass('visible').addClass('hidden');
	$('#p-left > #v-menu > #'+current_application.name).removeClass('hidden').addClass('visible');

	$('#p-center .application-workarea.visible').removeClass('visible').addClass('hidden');
	var wa_id = current_application.name;
	if(current_application.context != '') wa_id += '-'+current_application.context;
	$('#p-center .application-workarea#'+wa_id).removeClass('hidden').addClass('visible');

	current_application.select_default();
	v_item_selected($('#v-menu > #'+current_application.name).children('.v-item#'+current_application.context));
	
	set_context(current_application);
}

/* Event handler for vertical menu item (actual commands) selections:
 * - dim the poreviously active item
 * - hightlight the new item
 * - change the current context
 * - execute the commands
 * - switch the work area (make the old one invisible, and the new one visible)
 */
function v_item_selected(item) {

 	var item = $(item);
	if($(item).hasClass('v-select')) return;

	if( current_application.context != item.attr('id')) {
		current_application.if_ready2giveup( function() {
			v_item_selected_impl(item);
		});
		return;
	}
	v_item_selected_impl(item);
}

function v_item_selected_impl(item) {

	$('#'+current_application.name).find('.v-item.v-select').each(function(){
		$(this).children('.ui-icon').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
		$(this).removeClass('v-select');
	});

	$(item).children('.ui-icon').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
	$(item).addClass('v-select');

	/* Hide the older work area
	 */
	var wa_id = current_application.name;
	if(current_application.context != '') wa_id += '-'+current_application.context;
	$('#p-center > #application-workarea > #'+wa_id).removeClass('visible').addClass('hidden');

	current_application.select(item.attr('id'));

	/* display the new work area
	 */
	wa_id = current_application.name;
	if(current_application.context != '') wa_id += '-'+current_application.context;
	$('#p-center > #application-workarea > #'+wa_id).removeClass('hidden').addClass('visible');

	set_context(current_application);
}

$(function() {

	$('.m-item' ).click(function() { m_item_selected (this); });
	$('.v-item' ).click(function() { v_item_selected (this); });

	$('#p-search-text').keyup(function(e) { if(($(this).val() != '') && (e.keyCode == 13)) global_simple_search(); });

	// Finally, activate the selected application.
	//
	for(var id in applications) {
		var application = applications[id];
		if(application.name == select_app) {
			$('#p-menu').children('#p-appl-'+select_app).each(function() { m_item_selected(this); });
			if( '' != select_app_context ) {
				v_item_selected($('#v-menu > #'+select_app+' > #'+select_app_context));
				application.select(select_app_context);
			}
		}
	}
});


/* ----------------------------------------------------
 *             CROSS_APPLICATION DATA
 * ------------------------------------------------------
 */

// TODO: These structures have to be re-populated from the database
// at the first use and after any modifications made to cables from within
// the current application.
//

var global_routing = [
    'HV:TDFEHF02'
];

var global_instr = [ '0', '5', '7' ];

/* ------------------------------------------------------
 *             CROSS_APPLICATION EVENT HANDLERS
 * ------------------------------------------------------
 */
function global_switch_context(application_name, context_name) {
	var that = this;
	this.application_name = application_name;
	this.context_name = context_name;
	this.execute = function() {
		for(var id in applications) {
			var application = applications[id];
			if(application.name == that.application_name) {
				$('#p-menu').children('#'+id).each(function() {	m_item_selected(this); });
				v_item_selected($('#v-menu > #'+that.application_name).children('.v-item#'+that.context_name));
				application.select(that.context_name);
				break;
			}
		}
	};
	return this;
}
function global_simple_search() {
	var cxt = new global_switch_context('search', 'cables');
	cxt.execute();
	application.simple_search($('#p-search-text').val());
}

</script>

</head>

<body onresize="resize()">

<div id="p-top">
  <div id="p-top-header">
    <div id="p-top-title">
      <div style="float:left; padding-left:15px; padding-top:10px;">
        <span id="p-title"><?php echo $document_title?></span>
        <span id="p-subtitle"><?php echo $document_subtitle?></span>
      </div>
      <div style="float:right; padding-right:4px;">
        <table><tbody><tr>
          <td valign="bottom">
            <div style="float:right; margin-right:10px;" class="not4print"><a href="javascript:printer_friendly()" title="Printer friendly version of this page"><img src="img/PRINTER_icon.gif" style="border-radius: 5px;" /></a></div>
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
                <td><p><b><?php echo $authdb->authName()?></b></p></td>
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
      <div class="m-item m-item-first m-select" id="p-appl-projects">Projects</div>
      <div class="m-item m-item-next" id="p-appl-dictionary">Dictionary</div>
      <div class="m-item m-item-next" id="p-appl-search">Search</div>
      <div class="m-item m-item-last" id="p-appl-admin">Admin</div>
      <div class="m-item-end"></div>
    </div>
    <div id="p-context-header">
      <div id="p-context" style="float:left"></div>
      <div id="p-search" style="float:right">
        quick search: <input type="text" id="p-search-text" value="" size=16 title="enter text to search in the application, then press RETURN to proceed"  style="font-size:80%; padding:1px; margin-top:6px;" />
      </div>
      <div style="clear:both;"></div>
    </div>
  </div>
</div>

<div id="p-left">

<div id="v-menu">

    <div id="menu-title"></div>

    <div id="projects" class="visible">
      <div class="v-item" id="search">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
        <div style="float:left;" >Search</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="create">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >Create Project</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="dictionary" class="hidden">
      <div class="v-item" id="types">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Types</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="locations">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Locations</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="routings">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Routings</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="search" class="hidden">
      <div class="v-item" id="cables">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Cables</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="admin" class="hidden">
      <div class="v-item" id="cablenumbers">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Cable Numbers</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="jobs">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Job assignments</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="access">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Access Rights</div>
        <div style="clear:both;"></div>
      </div>
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

    <!-- An interface for displaying all known projects -->
    <div id="projects-search" class="application-workarea hidden">

      <div id="projects-search-controls">
        <table style="font-size:90%;"><tbody>

          <!-- Controls for selecting projects for display and updating the list of
            -- the selected projects.
            -->
          <tr>
            <td><b>Project status:</b></td>
            <td><select name="sort" style="padding:1px;">
                  <option>any</option>
                  <option>in-progress</option>
                  <option>submitted</option>
                </select></td>
            <td></td>
            <td><b>Owner:</b></td>
            <td><select name="sort" style="padding:1px;">
                  <option>any</option>
                  <option>gapon</option>
                  <option>perazzo</option>
                </select></td>
            <td><b>Created: </b></td>
            <td><input type="text" size=6 id="projects-search-after" title="specify the start time if applies" />
                <b>&mdash;</b>
                <input type="text" size=6 id="projects-search-before" title="specify the end time if applies" /></td>
            <td><div style="width:20px;"></div></td>
             
            <!-- Search commands -->
            <td><button id="projects-search-search" title="refresh the projects list">Search</button></td>
            <td><button id="projects-search-reset"  title="reset the serach form to the default state">Reset Form</button></td>
          </tr>

          <!-- Controls to change various display and presentation options (sorting, etc.)
            -- for the projects dynamically loaded from a Web service.
            -->
          <tr>
            <td><b>Sort by:</b></td>
            <td><select name="sort" style="padding:1px;">
                  <option>created</option>
                  <option>owner</option>
                  <option>title</option>
                  <option>cable</option>
                  <option>status</option>
                  <option>changed</option>
                  <option>due</option>
                </select></td>
            <td></td>
            <td><b>Order:</b></td>
            <td><input type="checkbox" name="reverse" ></input>reverse</td>
            </td>
          </tr>
        </tbody></table>
      </div>
      <div style="float:right;" id="projects-search-info">&nbsp;</div>
      <div style="clear:both;"></div>

      <!-- The projects display -->
      <div id="projects-search-display">

        <!-- Table header -->
        <div id="projects-search-header">
          <div style="float:left; margin-left:20px; width:150px;"><span class="proj-table-hdr">Created</span></div>
          <div style="float:left;                   width: 70px;"><span class="proj-table-hdr">Owner</span></div>
          <div style="float:left;                   width:300px;"><span class="proj-table-hdr">Title</span></div>
          <div style="float:left;                   width:120px;"><span class="proj-table-hdr">Due by</span></div>
          <div style="float:left;                   width: 70px;"><span class="proj-table-hdr"># Cables:</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Pln</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Rgs</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Lbl</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Fbr</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Rdy</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Ins</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Cms</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Dmg</span></div>
          <div style="float:left;                   width: 35px;"><span class="proj-table-hdr">Rtr</span></div>
          <div style="float:left;                   width:100px;"><span class="proj-table-hdr">Status</span></div>
          <div style="float:left;                   width:120px;"><span class="proj-table-hdr">Last changed</span></div>
          <div style="clear:both;"></div>
        </div>

        <!-- Table body is loaded dynamically by the application -->
        <div id="projects-search-list"></div>

      </div>

    </div>

    <div id="projects-create" class="application-workarea hidden">
      <div style="margin-bottom:20px; border-bottom:1px dashed #c0c0c0;">
        <div style="float:left;">
          <form id="projects-create-form">
            <table style="font-size:95%;"><tbody>
              <tr>
                <td><b>Owner:<?=$required_field_html?></b></td>  <td><select            name="owner" class="projects-create-form-element">
                                                                        <option>gapon</option>
                                                                        <option>perazzo</option>
                                                                      </select></td>
                <td><b>Title:<?=$required_field_html?></b></td><td><input type="text" name="name" size="50" class="projects-create-form-element" value="" /></td>
              </tr>
              <tr>
                <td><b>Descr: </b></td>
                <td colspan="4"><textarea cols=54 rows=4 name="descr" class="projects-create-form-element" style="padding:4px;" title="Here be the project description"></textarea></td>
              </tr>
              <tr>
                <td><b>Due by:</b></td><td><input type="text" name="due" size="6" class="projects-create-form-element" value="" /></td>
              </tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left; padding:5px;">
          <button id="projects-create-save">Save</button>
          <button id="projects-create-reset">Reset Form</button>
        </div>
        <div style="clear:both;"></div>
      </div>
      <?=$required_field_html?> required feild
    </div>


    <div id="dictionary-types" class="application-workarea hidden">
      <div class="application-workarea-instructions" style="float:left; " >
        <table><tbody>
          <tr>
            <td nowrap="nowrap" valign="top" class="table_cell table_cell_left table_cell_bottom ">USAGE NOTES:</td>
            <td class="table_cell table_cell_right table_cell_bottom table_cell_top ">
              This pages displays a dictionary of cable types, related connectors and pin lists.
              Authorized users are also allowed to modify the contents of the dictionary.
              More specific instructions will be added here later...
            </td>
          </tr>
        </tbody></table>
      </div>
      <div style="float:left; margin-left:40px; " ><button id="dictionary-types-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div id="dictionary-types-info" >&nbsp;</div>

      <div class="dict-section dict-section-first" style="float:left;">
        <div class="dict-header" >
          <div style="float:left; width:40px;  ">&nbsp;</div>
          <div style="float:left; width:120px; ">Cable Type</div>
          <div style="float:left; width:120px; ">Created</div>
          <div style="float:left; width:80px;  ">By</div>
          <div style="float:left; width:60px;  "># Refs</div>
          <div style="clear:both;              "></div>
        </div>
        <div class="dict-body" id="dictionary-types-cables"></div>
        <div class="dict-input">
          <div style="float:left; width:40px; padding-top: 4px; "><b>add</b> &rarr;</div>
          <div style="float:left; "><input type="text" size="12" name="cable2add" title="fill in new cable type, press RETURN to save" /></div>
          <div style="clear:both; "></div>
        </div>
      </div>

      <div class="dict-section" style="float:left;">
        <div class="dict-header">
          <div style="float:left; width:40px;  ">&nbsp;</div>
          <div style="float:left; width:120px; ">Connector</div>
          <div style="float:left; width:120px; ">Created</div>
          <div style="float:left; width:80px;  ">By</div>
          <div style="float:left; width:60px;  "># Refs</div>
          <div style="clear:both;              "></div>
        </div>
        <div class="dict-body" id="dictionary-types-connectors"></div>
        <div class="dict-input">
          <div style="float:left; width:40px; padding-top: 4px; "><b>add</b> &rarr;</div>
          <div style="float:left; "><input type="text" size="12" name="connector2add" title="fill in new connector type, press RETURN to save" /></div>
          <div style="clear:both; "></div>
        </div>
      </div>

      <div class="dict-section" style="float:left;">
        <div class="dict-header">
          <div style="float:left; ">&nbsp;</div>
          <div style="float:left; width:120px; ">Pin List</div>
          <div style="float:left; width:120px; ">Created</div>
          <div style="float:left; width:80px;  ">By</div>
          <div style="float:left; width:60px;  "># Refs</div>
          <div style="clear:both;              "></div>
        </div>
        <div class="dict-body" id="dictionary-types-pinlists"></div>
        <div class="dict-input">
          <div style="float:left; width:40px; padding-top: 4px; "><b>add</b> &rarr;</div>
          <div style="float:left; "><input type="text" size="12" name="pinlist2add" title="fill in new pin list type, press RETURN to save" /></div>
          <div style="clear:both; "></div>
        </div>
      </div>

      <div style="clear:both;"></div>

    </div>

    <div id="dictionary-locations" class="application-workarea hidden">

      <div class="application-workarea-instructions" style="float:left; " >
        <table><tbody>
          <tr>
            <td nowrap="nowrap" valign="top" class="table_cell table_cell_left table_cell_bottom ">USAGE NOTES:</td>
            <td class="table_cell table_cell_right table_cell_bottom table_cell_top ">
              This pages displays a dictionary of locations and racks.
              Authorized users are also allowed to modify the contents of the dictionary.
              More specific instructions will be added here later...
            </td>
          </tr>
        </tbody></table>
      </div>
      <div style="float:left; margin-left:40px; " ><button id="dictionary-locations-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div id="dictionary-locations-info" >&nbsp;</div>

      <div class="dict-section dict-section-first" style="float:left;">
        <div class="dict-header" >
          <div style="float:left; width:40px;  ">&nbsp;</div>
          <div style="float:left; width:120px; ">Location</div>
          <div style="float:left; width:120px; ">Created</div>
          <div style="float:left; width:80px;  ">By</div>
          <div style="float:left; width:60px;  "># Refs</div>
          <div style="clear:both;              "></div>
        </div>
        <div class="dict-body" id="dictionary-locations-locations"></div>
        <div class="dict-input">
          <div style="float:left; width:40px; padding-top: 4px; "><b>add</b> &rarr;</div>
          <div style="float:left; "><input type="text" size="12" name="location2add" title="fill in new location name, press RETURN to save" /></div>
          <div style="clear:both; "></div>
        </div>
      </div>

      <div class="dict-section" style="float:left;">
        <div class="dict-header">
          <div style="float:left; width:40px;  ">&nbsp;</div>
          <div style="float:left; width:120px; ">Rack</div>
          <div style="float:left; width:120px; ">Created</div>
          <div style="float:left; width:80px;  ">By</div>
          <div style="float:left; width:60px;  "># Refs</div>
          <div style="clear:both;              "></div>
        </div>
        <div class="dict-body" id="dictionary-locations-racks"></div>
        <div class="dict-input">
          <div style="float:left; width:40px; padding-top: 4px; "><b>add</b> &rarr;</div>
          <div style="float:left; "><input type="text" size="12" name="rack2add" title="fill in new rack name, press RETURN to save" /></div>
          <div style="clear:both; "></div>
        </div>
      </div>

      <div style="clear:both;"></div>

    </div>

    <div id="dictionary-routings"  class="application-workarea hidden">Cable routings</div>


    <!-- The DEMO version of an area for searching cables and displaying
      -- results. This form will be reimplemented to be more dynamic.
      -- In particular, the following options need to be obtained/loaded
      -- from a database:
      --
      --   o a list of projects 
      --   o a list of known systems
      --
      -- It's also possible to have an auto-complete dialog for values
      -- which have already been used before. This should accelerate
      -- the use of the search form.
      -->

    <div id="search-cables" class="application-workarea hidden">
      <div style="border-bottom:1px dashed #c0c0c0; margin-bottom:10px;">
        <div style="float:left;">
          <form id="search-cables-form">
            <table style="font-size:95%;"><tbody>
              <tr>
                <td><b>Project:</b></td>
                <td>
                  <select name="project">
                    <option></option>
                    <option>no project</option>
                    <option>LAN for CXI</option>
                    <option>InfiniBand Network for FEH</option>
                  </select>
                </td>
                <td><b>System:</b></td>
                <td><input type="text" name="system" value=""></input></td>
                <td><b>Source (loc):</b></td>
                <td>
                  <select name="source_loc">
                    <option></option>
                    <option>AMO</option>
                    <option>CXI</option>
                    <option>B999</option>
                  </select>
                </td>
              </tr>
              <tr>
                <td><b>Cable #:</b></td>
                <td><input type="text" name="cable_number" value=""></input></td>
                <td><b>Function:</b></td>
                <td><input type="text" name="function" value=""></input></td>
                <td><b>Dest (loc):</b></td>
                <td>
                  <select name="destination_loc">
                    <option></option>
                    <option>AMO</option>
                    <option>CXI</option>
                    <option>B999</option>
                  </select>
                </td>
              </tr>
              <tr>
                <td><b>Job #:</b></td>
                <td><input type="text" name="job_number" value=""></input></td>
                <td><b>Type:</b></td>
                <td>
                  <select name="type">
                    <option></option>
                    <option>1PR18OTN</option>
                    <option>CAT6STLN</option>
                    <option>CNT195</option>
                    <option>CAT6UTLN</option>
                  </select>
                </td>
              </tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left; margin-left:20px; padding:5px;">
          <button id="search-cables-search">Search</button>
          <button id="search-cables-reset">Reset Form</button>
        </div>
        <div style="clear:both;"></div>
      </div>
      <div style="border-bottom:1px dashed #c0c0c0; margin-bottom:15px;">
      <form id="search-cables-display-form">
        <div style="float:left;">
          <table style="font-size:95%;"><tbody>
            <tr>
              <td valign="top"><b>Sort by:</b></td>
              <td>
                <select name="sort">
                  <option>project</option>
                  <option>job #</option>
                  <option>cable #</option>
                  <option>system</option>
                  <option>function</option>
                  <option>source</option>
                  <option>destination</option>
                </select>
              </td>
            </tr>
            <tr>
              <td></td>
              <td><input type="checkbox" name="reverse" ></input>reverse</td>
              </td>
            </tr>
          </tbody></table>
        </div>
        <div style="float:left; margin-left:20px; padding-left:10px; /*border-left:1px dashed #c0c0c0;*/">
          <table style="font-size:95%;"><tbody>
            <tr>
              <td valign="top"><b>Display:</b></td>
              <td><input type="checkbox" name="project"                      ></input>project</td>
              <td><input type="checkbox" name="job"                          ></input>job #</td>
              <td><input type="checkbox" name="system"      checked="checked"></input>system</td>
            </tr>
            <tr>
              <td></td>
              <td><input type="checkbox" name="function"    checked="checked"></input>function</td>
              <td><input type="checkbox" name="source"      checked="checked"></input>source</td>
              <td><input type="checkbox" name="destination" checked="checked"></input>destination</td>
            </tr>
          </tbody></table>
        </div>
        <div style="float:left; margin-left:20px; padding-left:10px; /*border-left:1px dashed #c0c0c0;*/">
          <table style="font-size:95%;"><tbody>
            <tr>
              <td valign="top"><b>Export to:</b></td>
              <td></td>
              <td><a class="link" href="" target="_blank" title="Microsoft Excel 2008 File"><img src="img/EXCEL_icon.gif" /></a></td>
              <td><a class="link" href="" target="_blank" title="Text File to be embeded into Confluence Wiki"><img src="img/WIKI_icon.png"/></a></td>
              <td><a class="link" href="" target="_blank" title="Plain Text File"><img src="img/TEXT_icon.png" /></a></td>
            </tr>
          </tbody></table>
        </div>
        <div id="search-cables-info" style="float:right;">&nbsp;</div>
        <div style="clear:both;"></div>
      </form>
      </div>
      <div id="search-cables-result"></div>
    </div>

    <div id="admin-cablenumbers" class="application-workarea hidden">View/manage cable numbers allocation</div>
    <div id="admin-jobs"         class="application-workarea hidden">Manage job number assignments to authorized personell</div>
    <div id="admin-access"       class="application-workarea hidden">View/manage access privileges to the application.<br/>
                                                                     Add/remove user accounts allowed to modify the database contents.<br/>
                                                                     This page is supposed to be available only to the administrators of this service.</div>
  </div>

  <div id="popupdialogs" style="display:none;"></div>
</div>

</body>
</html>


<!--------------------- Document End Here -------------------------->

<?php

	$authdb->commit();

} catch( LusiTimeException $e ) { print $e->toHtml(); }
  catch( AuthDBException   $e ) { print $e->toHtml(); }

?>
