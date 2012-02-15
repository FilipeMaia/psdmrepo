<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;


$document_title = 'PCDS Cable Management:';
$document_subtitle = 'Neo-CAPTAR';

$required_field_html = '<span style="color:red; font-size:110%; font-weight:bold;"> * </span>';

try {

	$authdb = AuthDB::instance();
	$authdb->begin();

    $neocaptar = NeoCaptar::instance();
	$neocaptar->begin();
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
  height: 35px;
  background-color: #E0E0E0;
  border-bottom: 2px solid #a0a0a0;
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
  margin-top: 5px;

  padding: 5px;
  padding-left: 10px;
  padding-right: 10px;

  background: #DFEFFC url(/jquery/css/custom-theme/images/ui-bg_glass_85_dfeffc_1x400.png) 50% 50% repeat-x;

  color: #0071BC;

  border-right: 2px solid #a0a0a0;

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
.section1,
.section2,
.section3 {
  margin: 0.25em 0 0.25em 0;
  padding: 0.25em;
  padding-left: 0.5em;

  background-color: #DEF0CD;
  border: 2px solid #a0a0a0;

  border-left:0;
  border-top:0;
  border-radius: 5px;
  -moz-border-radius: 5px;

  font-family: "Times", serif;
  font-size: 36px;
  font-weight: bold;
  text-align:left;
}
.section2 {
  font-size: 28px;
}
.section3 {
  font-size: 18px;
}
.hidden  { display: none; }
.visible { display: block; }

#popupdialogs {
  display: none;
}

#infodialogs {
  display: none;
  padding: 20px;
}
#editdialogs {
  display: none;
  padding: 20px;
}

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

function report_info( title, msg ) {
	$('#infodialogs').html(msg);
	$('#infodialogs').dialog({
		resizable: true,
		modal: true,
/*
        buttons: {
			Cancel: function() {
				$(this).dialog('close');
			}
		},
*/
		title: title
	});
}

function report_action( title, msg ) {
	return $('#infodialogs').
        html(msg).
        dialog({
    		resizable: true,
        	modal: true,
            buttons: {
                Cancel: function() {
                    $(this).dialog('close');
                }
            },
            title: title
        });
}

function edit_dialog( title, msg, on_save, on_cancel ) {
	$('#editdialogs').html(msg);
	$('#editdialogs').dialog({
		resizable: true,
		modal: true,
		buttons: {
			Save: function() {
				$(this).dialog('close');
				if( on_save != null ) on_save();
			},
			Cancel: function() {
				$(this).dialog('close');
				if( on_cancel != null ) on_cancel();
			}
		},
		title: title
	});
}

/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */
var global_current_user = {
    uid:                '<?php echo $authdb->authName(); ?>',
    is_administrator:    <?php echo $neocaptar->is_administrator()?'1':'0'; ?>,
    can_manage_projects: <?php echo $neocaptar->can_manage_projects()?'1':'0'; ?>
};
var global_users = [];
<?php
    foreach( $neocaptar->users() as $user ) {
        echo "global_users.push('{$user->uid()}');\n";
    }
?>
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

    // Make sure the dictionaries are loaded
    //
    dict.init();

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

/* ------------------------------------------------------
 *             CROSS_APPLICATION EVENT HANDLERS
 * ------------------------------------------------------
 */
function global_switch_context(application_name, context_name) {
	for(var id in applications) {
		var application = applications[id];
		if(application.name == application_name) {
			$('#p-menu').children('#'+id).each(function() {	m_item_selected(this); });
			v_item_selected($('#v-menu > #'+application_name).children('.v-item#'+context_name));
            if( context_name != null) application.select(context_name);
            else application.select_default();
			return application;
		}
	}
    return null;
}
function global_simple_search                     ()            { global_switch_context('search',  'cables').simple_search($('#p-search-text').val()); }
function global_search_cable_by_cablenumber       (cablenumber) { global_switch_context('search',  'cables').search_cable_by_cablenumber       (cablenumber); }
function global_search_cables_by_prefix           (prefix)      { global_switch_context('search',  'cables').search_cables_by_prefix           (prefix); }
function global_search_cables_by_jobnumber        (jobnumber)   { global_switch_context('search',  'cables').search_cables_by_jobnumber        (jobnumber); }
function global_search_cables_by_jobnumber_prefix (prefix)      { global_switch_context('search',  'cables').search_cables_by_jobnumber_prefix (prefix); }
function global_search_cables_by_dict_cable_id    (id)          { global_switch_context('search',  'cables').search_cables_by_dict_cable_id    (id); }
function global_search_cables_by_dict_connector_id(id)          { global_switch_context('search',  'cables').search_cables_by_dict_connector_id(id); }
function global_search_cables_by_dict_pinlist_id  (id)          { global_switch_context('search',  'cables').search_cables_by_dict_pinlist_id  (id); }
function global_search_cables_by_dict_location_id (id)          { global_switch_context('search',  'cables').search_cables_by_dict_location_id (id); }
function global_search_cables_by_dict_rack_id     (id)          { global_switch_context('search',  'cables').search_cables_by_dict_rack_id     (id); }
function global_search_cables_by_dict_routing_id  (id)          { global_switch_context('search',  'cables').search_cables_by_dict_routing_id  (id); }
function global_search_cables_by_dict_instr_id    (id)          { global_switch_context('search',  'cables').search_cables_by_dict_instr_id    (id); }
function global_search_project_by_id              (id)          { global_switch_context('projects','search').search_project_by_id              (id); }
function global_search_projects_by_owner          (uid)         { global_switch_context('projects','search').search_projects_by_owner          (uid); }

function global_export_cables(search_params,outformat) {
    search_params.format = outformat;
    var html = '<img src="../logbook/images/ajaxloader.gif" />';
    var dialog = report_action('Generating Document: '+outformat,html);
    var jqXHR = $.get(
        '../portal/neocaptar_cable_search.php', search_params,
        function(data) {
            if( data.status != 'success' ) {
                report_error( data.message );
                dialog.dialog('close');
                return;
            }
            var html = 'Document is ready to be downloaded from this location: <a class="link" href="'+data.url+'" target="_blank" >'+data.name+'</a>';
            dialog.html(html);
        },
        'JSON'
    ).error(
        function () {
            report_error('failed because of: '+jqXHR.statusText);
            dialog.dialog('close');
        }
    ).complete(
        function () {
        }
    );
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
      <div class="v-item" id="instrs">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Instructions</div>
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
      <div class="v-item" id="jobnumbers">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Job Numbers</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="access">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Access Control</div>
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
            <td><b>Project title:</b></td>
            <td><input type="text" size=6 name="title" title="put in a text to match then press RETURN to search" /></td>
            <td></td>
            <td><b>Owner:</b></td>
            <td><select name="owner" style="padding:1px;">
<?php
    print "<option></option>";
    foreach( $neocaptar->known_project_owners() as $owner)
        print "<option>{$owner}</option>";
?>
                </select></td>
            <td><b>Created: </b></td>
            <td><input type="text" size=6 name="begin" title="specify the begin time if applies" />
                <b>&mdash;</b>
                <input type="text" size=6 name="end" title="specify the end time if applies" /></td>
            <td><div style="width:20px;"></div></td>
            <td><button name="search" title="refresh the projects list">Search</button></td>
            <td><button name="reset"  title="reset the search form to the default state">Reset Form</button></td>
          </tr>
        </tbody></table>
      </div>
      <div style="float:right;" id="projects-search-info">&nbsp;</div>
      <div style="clear:both;"></div>

      <!-- The projects display -->
      <div id="projects-search-display">

        <!-- Table header -->
        <div id="projects-search-header">
          <div style="float:left; margin-left:20px; width:100px;"><span class="proj-table-hdr">Created</span></div>
          <div style="float:left;                   width: 70px;"><span class="proj-table-hdr">Owner</span></div>
          <div style="float:left;                   width:300px;"><span class="proj-table-hdr">Title</span></div>
          <div style="float:left; margin-right: 9px; border-right:1px solid #000000; width: 60px;"><span class="proj-table-hdr">Cables</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Pln</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Reg</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Lbl</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Fbr</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Rdy</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Ins</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Com</span></div>
          <div style="float:left;                   width: 40px;"><span class="proj-table-hdr">Dmg</span></div>
          <div style="float:left;margin-right: 19px; border-right:1px solid #000000; width: 60px;"><span class="proj-table-hdr">Rtr</span></div>
          <div style="float:left;                   width:100px;"><span class="proj-table-hdr">Deadline</span></div>
          <div style="float:left;                   width:160px;"><span class="proj-table-hdr">Modified</span></div>
          <div style="clear:both;"></div>
        </div>

        <!-- Table body is loaded dynamically by the application -->
        <div id="projects-search-list"><div style="color:maroon; margin-top:10px;">Use the search form to find projects...</div></div>

      </div>

    </div>

    <div id="projects-create" class="application-workarea hidden">
<?php
    if( $neocaptar->can_manage_projects())
        print <<<HERE
      <div style="margin-bottom:20px; border-bottom:1px dashed #c0c0c0;">
        <div style="float:left;">
          <form id="projects-create-form">
            <table style="font-size:95%;"><tbody>
              <tr>
                <td><b>Owner:{$required_field_html}</b></td><td><input type="text" name="owner" size="5"  class="projects-create-form-element" style="padding:2px;" value="{$authdb->authName()}" /></td>
              </tr>
              <tr>
                <td><b>Title:{$required_field_html}</b></td><td><input type="text" name="title"  size="50" class="projects-create-form-element" style="padding:2px;" value="" /></td>
              </tr>
              <tr>
                <td><b>Descr: </b></td><td colspan="4"><textarea cols=54 rows=4 name="description" class="projects-create-form-element" style="padding:4px;" title="Here be the project description"></textarea></td>
              </tr>
              <tr>
                <td><b>Due by:{$required_field_html}</b></td><td><input type="text" name="due_time" size="6" class="projects-create-form-element" value="" /></td>
              </tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left; padding:5px;">
          <div>
            <button id="projects-create-save">Create</button>
            <button id="projects-create-reset">Reset Form</button>
          </div>
          <div style="margin-top:5px;" id="projects-create-info" >&nbsp;</div>
        </div>
        <div style="clear:both;"></div>
      </div>
      {$required_field_html} required feild
HERE;
      else {
          $admin_access_href = "javascript:global_switch_context('admin','access')";
        print <<<HERE
<br><br>
<center>
  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">
    A c c e s s &nbsp; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">
  We're sorry! Your SLAC UNIX account <b>{$authdb->authName()}</b> has no sufficient permissions for this operation.
  Normally we assign this task to authorized <a href="{$admin_access_href}">project managers</a>.
  Please contact administrators of this application if you think you need to create/manage projects.
  A list of administrators can be found in the <a href="{$admin_access_href}">Access Control</a> section of the <a href="{$admin_access_href}">Admin</a> tab of this application.
</div>
HERE;
      }
?>
    </div>

    <div id="dictionary-types" class="application-workarea hidden">
      <div><button id="dictionary-types-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left;">
        <div style="margin-top:20px; margin-bottom:10px;">
          <div style="float:left; padding-top: 4px; ">Add new cable type:</div>
          <div style="float:left; "><input type="text" size="12" name="cable2add" title="fill in new cable type, press RETURN to save" /></div>
          <div style="float:left; padding-top: 4px; ">(8 chars)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-types-cables"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px; margin-bottom:10px;">
          <div style="float:left; padding-top:4px; ">Add new connector type:</div>
          <div style="float:left; "><input type="text" size="12" name="connector2add" title="fill in new connector type, press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; ">(7 chars)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-types-connectors"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px; margin-bottom:10px;">
          <div style="float:left; padding-top:4px; ">Add pin list:</div>
          <div style="float:left; "><input type="text" size="12" name="pinlist2add" title="fill in new pin list type, press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; ">(16 chars)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-types-pinlists"></div>
      </div>
      <div style="clear:both;"></div>
    </div>

    <div id="dictionary-locations" class="application-workarea hidden">
      <div><button id="dictionary-locations-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left;">
        <div style="margin-top:20px; margin-bottom:10px;">
          <div style="float:left; padding-top: 4px; ">Add new location:</div>
          <div style="float:left; "><input type="text" size="12" name="location2add" title="fill in new location name, press RETURN to save" /></div>
          <div style="float:left; padding-top: 4px; ">(6 chars)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-locations-locations"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px; margin-bottom:10px;">
          <div style="float:left; padding-top:4px; ">Add new rack:</div>
          <div style="float:left; "><input type="text" size="12" name="rack2add" title="fill in new rack name, press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; ">(6 chars)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-locations-racks"></div>
      </div>
      <div style="clear:both; "></div>
    </div>

    <div id="dictionary-routings" class="application-workarea hidden">
      <div><button id="dictionary-routings-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="margin-top:20px; margin-bottom:10px;">
        <div style="float:left; padding-top:4px; ">Add new routing:</div>
        <div style="float:left; "><input type="text" size="12" name="routing2add" title="fill in new routing name, press RETURN to save" /></div>
        <div style="float:left; padding-top:4px; "></div>
        <div style="clear:both; "></div>
      </div>
      <div id="dictionary-routings-routings"></div>
    </div>


    <div id="dictionary-instrs" class="application-workarea hidden">
      <div><button id="dictionary-instrs-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="margin-top:20px; margin-bottom:10px;">
        <div style="float:left; padding-top: 4px; ">Add new instruction:</div>
        <div style="float:left; "><input type="text" size="1" name="instr2add" title="fill in new instr name, press RETURN to save" /></div>
        <div style="float:left; padding-top: 4px; "> (1 digit)</div>
        <div style="clear:both; "></div>
      </div>
      <div id="dictionary-instrs-instrs"></div>
    </div>

    <div id="search-cables" class="application-workarea hidden">
      <div style="border-bottom: 1px solid #000000;">
        <div style="float:left;">
          <form id="search-cables-form">
            <table style="font-size:95%;"><tbody>
              <tr><td><b>Cable #</b>    </td><td><input type="text" name="cable"           size="6"  value="" title="full or partial cable number"        ></input></td>
                  <td><b>Cable Type</b> </td><td><input type="text" name="cable_type"      size="6"  value="" title="full or partial cable type"          ></input></td>
                  <td><b>Device</b>     </td><td><input type="text" name="device"          size="12" value="" title="full or partial device name"         ></input></td>
                  <td><b>Origin Loc.</b></td><td><input type="text" name="origin_loc"      size="3"  value="" title="full or partial origin location"     ></input></td></tr>
              <tr><td><b>Job #</b>      </td><td><input type="text" name="job"             size="6"  value="" title="full or partial job number"          ></input></td>
                  <td><b>Routing</b>    </td><td><input type="text" name="routing"         size="6"  value="" title="full or partial routing"             ></input></td>
                  <td><b>Function</b>   </td><td><input type="text" name="func"            size="12" value="" title="full or partial function name"       ></input></td>
                  <td><b>Dest. Loc.</b> </td><td><input type="text" name="destination_loc" size="3"  value="" title="full or partial destination location"></input></td></tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left; margin-left:20px; padding:5px;">
          <button id="search-cables-search">Search</button>
          <button id="search-cables-reset">Reset Form</button>
        </div>
        <div style="clear:both;"></div>
      </div>
      <div style="padding:10px">
        <div style="padding-bottom:10px;">
          <div style="float:left;  padding-top:5px; padding-bottom:10px;">
            <button class="export" name="excel" title="Export into Microsoft Excel 2007 File"><img src="img/EXCEL_icon.gif" /></button>
          </div>
          <div style="float:right;" id="search-cables-info">&nbsp;</div>
          <div style="clear:both;"></div>
          <div id="search-cables-display">
            <input type="checkbox" name="project" checked="checked"></input>project
            <input type="checkbox" name="job"     checked="checked"></input>job #
            <input type="checkbox" name="cable"   checked="checked"></input>cable #
            <input type="checkbox" name="device"  checked="checked"></input>device
            <input type="checkbox" name="func"    checked="checked"></input>function
            <input type="checkbox" name="length"  checked="checked"></input>length
            <input type="checkbox" name="routing" checked="checked"></input>routing
            <input type="checkbox" name="sd"      checked="checked"></input>source & destination
          </div>
        </div>
        <div id="search-cables-result"></div>
      </div>
    </div>

    <div id="admin-cablenumbers" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-cablenumbers-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div style="margin-top:30px;">
        <div class="section3">Ranges</div>
        <div id="admin-cablenumbers-cablenumbers"></div>
      </div>
    </div>

    <div id="admin-jobnumbers" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-jobnumbers-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div style="margin-top:30px;">
        <div class="section3">Ranges</div>
        <div id="admin-jobnumbers-jobnumbers"></div>
      </div>
      <div style="margin-top:30px;">
        <div class="section3">Allocated Numbers</div>
        <div id="admin-jobnumbers-allocations"></div>
      </div>
    </div>

    <div id="admin-access" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-access-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div style="margin-top:30px;">
        <div class="section3">Administrators</div>
        <div style="padding-left:20px; padding-top:20px;">
          <div style="float:left; padding-top: 4px; ">Add new administrator:</div>
          <div style="float:left; "><input type="text" size="12" name="administrator2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
          <div style="float:left; padding-top: 4px; ">(valid UNIX account)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="admin-access-administrators"></div>
      </div>
      <div style="margin-top:30px;">
        <div class="section3">Project Managers</div>
        <div style="padding-left:20px; padding-top:20px;">
          <div style="float:left; padding-top: 4px; ">Add new project manager:</div>
          <div style="float:left; "><input type="text" size="12" name="projmanager2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
          <div style="float:left; padding-top: 4px; ">(valid UNIX account)</div>
          <div style="clear:both; "></div>
        </div>
        <div id="admin-access-projmanagers"></div>
      </div>
    </div>

  </div>

  <div id="popupdialogs" ></div>
  <div id="infodialogs"  ></div>
  <div id="editdialogs"  ></div>

</div>

</body>
</html>


<!--------------------- Document End Here -------------------------->

<?php

	$authdb->commit();
	$neocaptar->commit();

} catch( AuthDBException    $e ) { print $e->toHtml(); }
  catch( LusiTimeException  $e ) { print $e->toHtml(); }
  catch( NeoCaptarException $e ) { print $e->toHtml(); }

?>
