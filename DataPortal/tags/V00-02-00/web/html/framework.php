<?php

require_once( 'lusitime/lusitime.inc.php' );
require_once( 'authdb/authdb.inc.php' );


use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

$document_title = 'Document Title:';
$document_subtitle = 'Document Sub-Title';

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

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />
<link type="text/css" href="css/common.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>

<script type="text/javascript" src="js/Utilities.js"></script>


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
		pfcopy.document.write('<title>Web Portal of Experiment: '+elog.instr+' / '+elog.exp+'</title></head><body><div class="maintext">');
		pfcopy.document.write(html);
		pfcopy.document.write("</div></body></html>");
		pfcopy.document.close();
	}
}



/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */
var applications = {
	'p-appl-first'  : new p_appl_first(),
	'p-appl-second' : new p_appl_second(),
	'p-appl-third'  : new p_appl_third(),
	'p-appl-forth'  : new p_appl_forth(),
	'p-appl-fifth'  : new p_appl_fifth()
};

var current_application = null;

var select_app = 'first';
var select_app_context1 = 'first-1';

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
			if(application.name == 'second') {
				$('#p-menu').children('#'+id).each(function() {	m_item_selected(this); });
				v_item_selected($('#v-menu > #second').children('.v-item#second_1'));
				application.select('second_1');
				application.simple_search($('#p-search-text').val());
				break;
			}
		}
	}
	$('#p-search-text').keyup(function(e) { if(($('#p-search-text').val() != '') && (e.keyCode == 13)) simple_search(); });

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

function p_appl_first() {
	var that = this;
	var context2_default = {
		'2' : '1'
	};
	this.name = 'first';
	this.full_name = 'First';
	this.context1 = '1';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	this.select_default = function() {
		this.context1 = '1';
		this.context2 = '';
	}
	return this;
}
function p_appl_second() {
	var that = this;
	var context2_default = {
		'2' : '1'
	};
	this.name = 'second';
	this.full_name = 'Second';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	this.select_default = function() {
		this.context1 = '1';
		this.context2 = '';
	}
	return this;
}
function p_appl_third() {
	var that = this;
	var context2_default = {
		'' : ''
	};
	this.name = 'third';
	this.full_name = 'Third';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	this.select_default = function() {
	}
	return this;
}
function p_appl_forth() {
	var that = this;
	var context2_default = {
		'' : ''
	};
	this.name = 'forth';
	this.full_name = 'Forth';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	this.select_default = function() {
	}
	return this;
}
function p_appl_fifth() {
	var that = this;
	var context2_default = {
		'' : ''
	};
	this.name = 'fifth';
	this.full_name = 'Fifth';
	this.context1 = '';
	this.context2 = '';
	this.select = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};
	this.select_default = function() {
	}
	return this;
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
      <div class="m-item m-item-first m-select" id="p-appl-first">First</div>
      <div class="m-item m-item-next" id="p-appl-second">Second</div>
      <div class="m-item m-item-next" id="p-appl-third">Third</div>
      <div class="m-item m-item-next" id="p-appl-forth">Forth</div>
      <div class="m-item m-item-last" id="p-appl-fifth">Fifth</div>
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

    <div id="first" class="visible">
      <div class="v-item" id="first-1">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >1</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group" id="first-2">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >2</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group-members v-group-members-hidden">
        <div class="v-item v-item-first" id="first-2-1">
          <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
          <div class="link" style="float:left;" >1</div>
          <div style="clear:both;"></div>
        </div>
      </div>
    </div>

    <div id="second" class="hidden">
      <div class="v-item" id="second-1">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
        <div class="link" style="float:left;" >1</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group" id="second-2">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >2</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-group-members v-group-members-hidden">
        <div class="v-item v-item-first" id="second-2-1">
          <div class="ui-icon ui-icon-triangle-1-s" style="float:left;"></div>
          <div class="link" style="float:left;" >1</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="second-2-2">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >2</div>
          <div style="clear:both;"></div>
        </div>
        <div class="v-item" id="second-2-3">
          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
          <div class="link" style="float:left;" >3</div>
          <div style="clear:both;"></div>
        </div>
      </div>
      <div class="v-item" id="second-3">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div class="link" style="float:left;" >3</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="second-4">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >4</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="second-5">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >5</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="second-6">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >6</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="second-7">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >7</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="third" class="hidden">
      <div class="v-item" id="third-1">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >1</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="third-2">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >2</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="forth" class="hidden">
      <div class="v-item" id="forth-1">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >1</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="forth-2">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >2</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="forth-3">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >3</div>
        <div style="clear:both;"></div>
      </div>
    </div>

    <div id="fifth" class="hidden">
      No menu here
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
    <div id="first-first-1"   class="application-workarea hidden">1.1</div>
    <div id="first-first-2"   class="application-workarea hidden">1.2</div>
    <div id="second-second-1" class="application-workarea hidden">2.1</div>
    <div id="second-second-2" class="application-workarea hidden">2.2</div>
    <div id="second-second-3" class="application-workarea hidden">2.3</div>
    <div id="second-second-4" class="application-workarea hidden">2.4</div>
    <div id="second-second-5" class="application-workarea hidden">2.5</div>
    <div id="second-second-6" class="application-workarea hidden">2.6</div>
    <div id="second-second-7" class="application-workarea hidden">2.7</div>
    <div id="third-third-1"   class="application-workarea hidden">3.1</div>
    <div id="third-third-2"   class="application-workarea hidden">3.2</div>
    <div id="forth-forth-1"   class="application-workarea hidden">4.1</div>
    <div id="forth-forth-2"   class="application-workarea hidden">4.2</div>
    <div id="forth-forth-3"   class="application-workarea hidden">4.3</div>
    <div id="fifth"           class="application-workarea hidden">5</div>
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
