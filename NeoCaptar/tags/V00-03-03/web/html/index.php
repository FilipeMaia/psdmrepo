<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

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


<!-- Document Begins Here -->


<!DOCTYPE html>
<html>

<head>
<title><?php echo $document_title ?></title>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="css/common.css" rel="Stylesheet" />
<link type="text/css" href="css/neocaptar.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>

<script type="text/javascript" src="js/projects.js"></script>
<script type="text/javascript" src="js/dictionary.js"></script>
<script type="text/javascript" src="js/search.js"></script>
<script type="text/javascript" src="js/admin.js"></script>

<script type="text/javascript" src="../webfwk/js/config.js"></script>
<script type="text/javascript" src="../webfwk/js/Table.js"></script>
<script type="text/javascript" src="../webfwk/js/Utilities.js"></script>


<!-- Window layout styles and support actions -->

<style type="text/css">

body {
  margin: 0;
  padding: 0;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
  font-size: 11px;
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
  padding-top:   15px;
  padding-right: 10px;
  font-size:     11px;
  font-family:   Arial, Helvetica, Verdana, Sans-Serif;
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
#p-center {
  position: relative;
  top:130px;
  margin: 0px 0px 20px 203px;
  margin-bottom: 0px;
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
  font-weight: bold;
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
  font-size: 12px;
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
  /*font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;*/
  /*font-size: 75%;*/
}
.section1,
.section2,
.section3 {
  margin: 0.25em 0 0.25em 0;
  padding: 0.25em;
  padding-left: 0.5em;

  /*
  background-color: #DEF0CD;
  */
  border: 2px solid #a0a0a0;

  border-left:0;
  border-top:0;
  border-right:0;
  /*
  border-radius: 5px;
  -moz-border-radius: 5px;
  */
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
  font-size: 11px;
}
#popupdialogs-varable-size {
  display: none;
  font-size: 11px;
}

#infodialogs {
  display: none;
  padding: 20px;
  font-size: 11px;
}
#editdialogs {
  display: none;
  padding: 20px;
  font-size: 11px;
}

</style>


<script type="text/javascript">

var config = new config_create('neocaptar') ;

/* ------------------------------------------------
 *          VERTICAL SPLITTER MANAGEMENT
 * ------------------------------------------------
 */
function resize() {
    $('#p-left').height($(window).height()-125-5);
    $('#p-splitter').height($(window).height()-125-5);
    $('#p-center').height($(window).height()-125-5);
}

/* Get mouse position relative to the document.
 */
function getMousePosition(e) {

    var posx = 0;
    var posy = 0;
    if (!e) var e = window.event;
    if (e.pageX || e.pageY) {
        posx = e.pageX;
        posy = e.pageY;
    } else if (e.clientX || e.clientY) {
        posx = e.clientX + document.body.scrollLeft+document.documentElement.scrollLeft;
        posy = e.clientY + document.body.scrollTop +document.documentElement.scrollTop;
    }
    return {'x': posx, 'y': posy };
}

function move_split(e) {
    var pos = getMousePosition(e);
    $('#p-left'    ).css('width',       pos['x']);
    $('#p-splitter').css('left',        pos['x']);
    $('#p-center'  ).css('margin-left', pos['x']+3);
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
    var ctx = app.full_name+' :';
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
        modal: true,
        title: 'e-mail: '+user
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
                if(on_yes) on_yes();
            },
            Cancel: function() {
                $(this).dialog('close');
                if(on_cancel) on_cancel();
            }
        },
        title: title
    });
}

function ask_for_input( title, msg, on_ok, on_cancel ) {
    $('#popupdialogs-varable-size').html(
'<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+msg+'</p>'+
'<div><textarea rows=4 cols=60></textarea/>'
     );
    $('#popupdialogs-varable-size').dialog({
        resizable: true,
        modal: true,
        width:  470,
        height: 300,
        buttons: {
            "Ok": function() {
                var user_input = $('#popupdialogs-varable-size').find('textarea').val();
                $( this ).dialog('close');
                if(on_ok) on_ok(user_input);
            },
            Cancel: function() {
                $(this).dialog('close');
                if(on_cancel) on_cancel();
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
                if(on_cancel) on_cancel();
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
        title: title
    });
}
function report_info_table( title, hdr, rows ) {
    var table = new Table('infodialogs', hdr, rows);
    table.display();
    $('#infodialogs').dialog({
        width: 720,
        height: 800,
        resizable: true,
        modal: true,
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
        width: 640,
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
    is_other:            <?php echo $neocaptar->is_other()?'1':'0'; ?>,
    is_administrator:    <?php echo $neocaptar->is_administrator()?'1':'0'; ?>,
    can_manage_projects: <?php echo $neocaptar->can_manage_projects()?'1':'0'; ?>,
    has_dict_priv:       <?php echo $neocaptar->has_dict_priv()?'1':'0'; ?>
};

var global_users = [];
var global_projmanagers = [];
<?php
    foreach( $neocaptar->users() as $user ) {
        echo "global_users.push('{$user->uid()}');\n";
        if( $user->is_administrator() || $user->is_projmanager()) echo "global_projmanagers.push('{$user->uid()}');\n";
    }
?>
function global_get_projmanagers() {
    var projmanagers = admin.projmanagers();
    if(projmanagers) return projmanagers;
    return global_projmanagers;
}
var applications = {
    'p-appl-projects'   : projects,
    'p-appl-dictionary' : dict,
    'p-appl-search'     : search,
    'p-appl-admin'      : admin
};

var current_application = null;

var select_app = 'projects';
var select_app_context = 'search';
<?php
$known_apps = array(
    'projects' => True,
    'dictionary' => True,
    'search' => True,
    'admin' => True
);
if( isset( $_GET['app'] )) {
    $app_path = explode( ':', strtolower(trim($_GET['app'])));
    $app = $app_path[0];
    if( array_key_exists( $app, $known_apps )) {
        echo "select_app = '{$app}';";
        echo "select_app_context = '".(count($app_path) > 1 ? $app_path[1] : "")."';";
    }
}
?>
var select_params = {
    project_id:null,
    cable_id:null
};
<?php
if( isset( $_GET['project_id'] )) {
    $project_id = intval(trim($_GET['project_id']));
    if($project_id) echo "select_params.project_id = {$project_id};";
}
?>
<?php
if( isset( $_GET['cable_id'] )) {
    $cable_id = intval(trim($_GET['cable_id']));
    if($cable_id) echo "select_params.cable_id = {$cable_id};";
}
?>
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
    admin.init();

    // Finally, activate the selected application.
    //
    for(var id in applications) {
        var application = applications[id];
        if(application.name == select_app) {
            $('#p-menu').children('#p-appl-'+select_app).each(function() { m_item_selected(this); });
            if( '' == select_app_context ) {
                v_item_selected($('#v-menu > #'+select_app+' > #'+application.default_context));
                application.select_default();
            } else {
                v_item_selected($('#v-menu > #'+select_app+' > #'+select_app_context));
                application.select(select_app_context);            
            }
            switch(application.name) {
            case 'projects':
                switch(application.context) {
                case 'search':
                    if(select_params.project_id) global_search_project_by_id(select_params.project_id);
                    break;
                }
                break;
            case 'search':
                switch(application.context) {
                case 'cables':
                    if(select_params.cable_id) global_search_cable_by_id(select_params.cable_id);
                    break;
                }
                break;
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
            $('#p-menu').children('#'+id).each(function() {    m_item_selected(this); });
            v_item_selected($('#v-menu > #'+application_name).children('.v-item#'+context_name));
            if( context_name != null) application.select(context_name);
            else application.select_default();
            return application;
        }
    }
    return null;
}
function global_simple_search                      ()            { global_switch_context('search',  'cables').simple_search($('#p-search-text').val()); }
function global_search_cable_by_cablenumber        (cablenumber) { global_switch_context('search',  'cables').search_cable_by_cablenumber        (cablenumber); }
function global_search_cable_by_id                 (id)          { global_switch_context('search',  'cables').search_cable_by_id                 (id); }
function global_search_cables_by_prefix            (prefix)      { global_switch_context('search',  'cables').search_cables_by_prefix            (prefix); }
function global_search_cables_by_cablenumber_range (range_id)    { global_switch_context('search',  'cables').search_cables_by_cablenumber_range (range_id); }
function global_search_cables_by_jobnumber         (jobnumber)   { global_switch_context('search',  'cables').search_cables_by_jobnumber         (jobnumber); }
function global_search_cables_by_jobnumber_prefix  (prefix)      { global_switch_context('search',  'cables').search_cables_by_jobnumber_prefix  (prefix); }
function global_search_cables_by_dict_cable_id     (id)          { global_switch_context('search',  'cables').search_cables_by_dict_cable_id     (id); }
function global_search_cables_by_dict_connector_id (id)          { global_switch_context('search',  'cables').search_cables_by_dict_connector_id (id); }
function global_search_cables_by_dict_pinlist_id   (id)          { global_switch_context('search',  'cables').search_cables_by_dict_pinlist_id   (id); }
function global_search_cables_by_dict_location_id  (id)          { global_switch_context('search',  'cables').search_cables_by_dict_location_id  (id); }
function global_search_cables_by_dict_rack_id      (id)          { global_switch_context('search',  'cables').search_cables_by_dict_rack_id      (id); }
function global_search_cables_by_dict_routing_id   (id)          { global_switch_context('search',  'cables').search_cables_by_dict_routing_id   (id); }
function global_search_cables_by_dict_instr_id     (id)          { global_switch_context('search',  'cables').search_cables_by_dict_instr_id     (id); }
function global_search_project_by_id               (id)          { global_switch_context('projects','search').search_project_by_id               (id); }
function global_search_projects_by_owner           (uid)         { global_switch_context('projects','search').search_projects_by_owner           (uid); }
function global_search_projects_coowned_by         (uid)         { global_switch_context('projects','search').search_projects_by_coowner         (uid); }
function global_search_projects_by_jobnumber       (jobnumber)   { global_switch_context('projects','search').search_projects_by_jobnumber       (jobnumber); }
function global_search_projects_by_jobnumber_prefix(prefix)      { global_switch_context('projects','search').search_projects_by_jobnumber_prefix(prefix); }

function global_search_cables_by_dict_device_location_id (id)   { global_switch_context('search',  'cables').search_cables_by_dict_device_location_id (id); }
function global_search_cables_by_dict_device_region_id   (id)   { global_switch_context('search',  'cables').search_cables_by_dict_device_region_id   (id); }
function global_search_cables_by_dict_device_component_id(id)   { global_switch_context('search',  'cables').search_cables_by_dict_device_component_id(id); }

function global_export_cables(search_params, outformat) {
    search_params.format = outformat;
    var url = '../neocaptar/ws/cable_search.php?'+$.param(search_params, true) ;
    window.open(url) ;
}
function global_truncate_cable    (str) { return str.substring(0, 8); }
function global_truncate_connector(str) { return str.substring(0, 8); }
function global_truncate_pinlist  (str) { return str.substring(0,16); }
function global_truncate_location (str) { return str.substring(0, 6); }
function global_truncate_rack     (str) { return str.substring(0, 6); }
function global_truncate_routing  (str) { return str.substring(0,50); }
function global_truncate_instr    (str) { return str.substring(0, 3); }
function global_truncate_func     (str) { return str.substring(0,33); }
function global_truncate_length   (str) { return str.substring(0, 4); }
function global_truncate_ele      (str) { return str.substring(0, 2); }
function global_truncate_side     (str) { return str.substring(0, 1); }
function global_truncate_slot     (str) { return str.substring(0, 6); }
function global_truncate_conn     (str) { return str.substring(0, 8); }
function global_truncate_station  (str) { return str.substring(0, 6); }

function global_truncate_device          (str) { return str.substring(0,18); }
function global_truncate_device_location (str) { return str.substring(0, 3); }
function global_truncate_device_region   (str) { return str.substring(0, 4); }
function global_truncate_device_component(str) { return str.substring(0, 3); }
function global_truncate_device_counter  (str) { return str.substring(0, 2); }
function global_truncate_device_suffix   (str) { return str.substring(0, 3); }

function global_cable_status2rank(status) {
    switch(status) {
        case 'Planned':      return 0;
        case 'Registered':   return 1;
        case 'Labeled':      return 2;
        case 'Fabrication':  return 3;
        case 'Ready':        return 4;
        case 'Installed':    return 5;
        case 'Commissioned': return 6;
        case 'Damaged':      return 7;
        case 'Retired':      return 8;
    }
    return -1;
}
function global_cable_sorter_by_status     (a,b) { return global_cable_status2rank(a.status) - global_cable_status2rank(b.status); }
function sort_as_text                      (a,b) { return a == b ? 0 : ( a < b ? -1 : 1 ); }
function global_cable_sorter_by_project    (a,b) { return sort_as_text(a.project_title,    b.project_title); }
function global_cable_sorter_by_job        (a,b) { return sort_as_text(a.job,              b.job); }
function global_cable_sorter_by_cable      (a,b) { return sort_as_text(a.cable,            b.cable); }
function global_cable_sorter_by_device     (a,b) { return sort_as_text(a.device,           b.device); }
function global_cable_sorter_by_function   (a,b) { return sort_as_text(a.func,             b.func); }
function global_cable_sorter_by_origin     (a,b) { return sort_as_text(a.origin.name,      b.origin.name); }
function global_cable_sorter_by_destination(a,b) { return sort_as_text(a.destination.name, b.destination.name); }
function global_cable_sorter_by_modified   (a,b) { return a.modified.time_64 - b.modified.time_64; }

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
      <div id="p-login" style="float:right;" >
        <div style="float:left; padding-top:20px;" class="not4print" >
          <a href="javascript:printer_friendly()" title="Printer friendly version of this page"><img src="img/PRINTER_icon.gif" style="border-radius: 5px;" /></a>
        </div>
        <div style="float:left; margin-left:10px;" >
          <table><tbody>
            <tr>
              <td>&nbsp;</td>
              <td>[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]</td></tr>
            <tr>
              <td>User:&nbsp;</td>
              <td><b><?php echo $authdb->authName()?></b></td></tr>
            <tr>
              <td>Session expires in:&nbsp;</td>
              <td id="auth_expiration_info"><b>00:00.00</b></td></tr>
          </tbody></table>
        </div>
        <div style="clear:both;" class="not4print"></div>
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
        quick search: <input type="text" id="p-search-text" value="" size=16 title="enter full or partial cable number, then press RETURN to proceed"  style="font-size:80%; padding:1px; margin-top:6px;" />
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
      <div class="v-item" id="pinlists">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Pinlists (drawings)</div>
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
      <div class="v-item" id="devices">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Device Name</div>
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
      <div class="v-item" id="access">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >Access Control</div>
        <div style="clear:both;"></div>
      </div>
      <div class="v-item" id="notifications">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>
        <div style="float:left;" >E-mail Notifications</div>
        <div style="clear:both;"></div>
      </div>
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
    </div>

  </div>
</div>

<div id="p-splitter"></div>

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
            <td><input type="text" size=32 name="title" title="put in a text to match then press RETURN to search" /></td>
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
          <div style="float:left; margin-left:20px; width: 90px;"><span class="proj-table-hdr">Created</span></div>
          <div style="float:left;                   width: 70px;"><span class="proj-table-hdr">Owner</span></div>
          <div style="float:left;                   width:300px;"><span class="proj-table-hdr">Title</span></div>
          <div style="float:left;                   width: 70px;"><span class="proj-table-hdr">Job #</span></div>
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
          <div style="float:left;                   width: 80px;"><span class="proj-table-hdr">Deadline</span></div>
          <div style="float:left;                   width: 80px;"><span class="proj-table-hdr">Modified</span></div>
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
          <div style="margin-bottom:10px; width:480px;">
            When making a clone of an existing project make sure the new project title differs
            from the original one. All cables of the original project will be copied into the new one.
            The copied cables will all be put into the 'Planned' state, and they won't have numbers.
            Make sure the cables are properly edited to avoid potential conflicts with the original
            ones before finalizing cable labesl.
          </div>
          <form id="projects-create-form">
            <table><tbody>
              <tr><td><b>Project to clone:</b></td>
                  <td><input type="text" name="project2clone" size="16" class="projects-create-form-element" style="padding:2px;" value="" /></td></tr>
              <tr><td>&nbsp;</td></tr>
              <tr><td><b>Owner:{$required_field_html}</b></td>
                  <td><input type="text" name="owner" size="5" class="projects-create-form-element" style="padding:2px;" value="{$authdb->authName()}" /></td></tr>
              <tr><td><b>Title:{$required_field_html}</b></td>
                  <td><input type="text" name="title"  size="50" class="projects-create-form-element" style="padding:2px;" value="" /></td></tr>
              <tr><td><b>Descr: </b></td>
                  <td colspan="4"><textarea cols=54 rows=4 name="description" class="projects-create-form-element" style="padding:4px;" title="Here be the project description"></textarea></td></tr>
              <tr><td><b>Due by:{$required_field_html}</b></td>
                  <td><input type="text" name="due_time" size="6" class="projects-create-form-element" value="" /></td></tr>
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
      <div style="margin-top:20px; margin-bottom:20px; width:720px;">
        <p>Cable and connector types can be grouped together as a matrix representing
        so called <b>"N-to-M bidirectional association"</b>. On one hand, a cable type may be
        associated with one or many different connector types. On the other hand, one connector type may be used in association with
        one or many different cable types. To facilitate viewing and managing relationships between cable and connector types
        this page provides two views onto the associations each from a different prospective.
        The <b>Cable Type View</b> shows which connector types "make sense" for a given cable type. The other view (the <b>Connector Type View</b>)
        shows which cable types are "suitable" for the given connector type.
        New types can be added/removed through either view. Once a new type is added at one view an opposite view would be automatically updated.
        </p>
      </div>

      <div id="tabs" style="font-size:12px;">
        <ul>
          <li><a href="#cables2connectors">Cable Type View</a></li>
          <li><a href="#connectors2cables">Connector Type View</a></li>
        </ul>

        <div id="cables2connectors" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="float:left;">
              <div style="margin-top:20px;">
                <div style="float:left; "><input type="text" size="12" name="cable2add" title="fill in new cable type (limit 8 characters), then press RETURN to save" /></div>
                <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new cable type here</div>
                <div style="clear:both; "></div>
              </div>
              <div id="dictionary-types-cables"></div>
            </div>
            <div style="float:left; margin-left:20px;">
              <div style="margin-top:20px;">
                <div style="float:left; "><input type="text" size="12" name="connector2add" title="fill in new connector type (limit 7 characters), then press RETURN to save" /></div>
                <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new connector type here</div>
                <div style="clear:both; "></div>
              </div>
              <div id="dictionary-types-connectors" ></div>
            </div>
            <div style="clear:both;"></div>
          </div>
        </div>
      
        <div id="connectors2cables" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="float:left;">
              <div style="margin-top:20px;">
                <div style="float:left; "><input type="text" size="12" name="connector2add" title="fill in new connector type (limit 7 characters), then press RETURN to save" /></div>
                <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new connector type here</div>
                <div style="clear:both; "></div>
              </div>
              <div id="dictionary-types-connectors-reverse"></div>
            </div>
            <div style="float:left; margin-left:20px;">
              <div style="margin-top:20px;">
                <div style="float:left;"><input type="text" size="12" name="cable2add" title="fill in new cable type (limit 8 characters), then press RETURN to save" /></div>
                <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new cable type here</div>
                <div style="clear:both; "></div>
              </div>
              <div id="dictionary-types-cables-reverse"></div>
            </div>
            <div style="clear:both;"></div>
          </div>
        </div>

      </div>
    </div>

    <div id="dictionary-pinlists" class="application-workarea hidden">
      <div><button id="dictionary-pinlists-reload" title="reload the dictionary from the database">Reload</button></div>
      <div>
        <div style="margin-top:20px;">
          <div style="float:left; "><input type="text" size="12" name="pinlist2add" title="fill in new pin list type (limit 16 characters), then press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new pinlist here</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-pinlists-pinlists"></div>
      </div>
    </div>

    <div id="dictionary-locations" class="application-workarea hidden">
      <div><button id="dictionary-locations-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left;">
        <div style="margin-top:20px;">
          <div style="float:left; "><input type="text" size="12" name="location2add" title="fill in new location name (limit 6 characters), then press RETURN to save" /></div>
              <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new location here</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-locations-locations"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px;">
          <div style="float:left; "><input type="text" size="12" name="rack2add" title="fill in new rack name (limit 6 characters), then press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new rack here</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-locations-racks"></div>
      </div>
      <div style="clear:both; "></div>
    </div>

    <div id="dictionary-routings" class="application-workarea hidden">
      <div><button id="dictionary-routings-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="margin-top:20px;">
        <div style="float:left; "><input type="text" size="32" name="routing2add" title="fill in new routing name (limit 50 characters), then press RETURN to save" /></div>
        <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new routing here</div>
        <div style="clear:both; "></div>
      </div>
      <div id="dictionary-routings-routings"></div>
    </div>

    <div id="dictionary-devices" class="application-workarea hidden">
      <div><button id="dictionary-devices-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left;">
        <div style="margin-top:20px;">
          <div style="float:left; "><input type="text" size="3" name="device_location2add" title="fill in new location (exactly 3 characters), then press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new location of instruments (LLL) here</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-devices-locations"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px;">
          <div style="float:left; "><input type="text" size="3" name="device_region2add" title="fill in new region (3 or 4 characters), then press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new region (RRR or RRRR) here</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-devices-regions"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px;">
          <div style="float:left; "><input type="text" size="3" name="device_component2add" title="fill in new component(exactly 3 characters), then press RETURN to save" /></div>
          <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new component (CCC) here</div>
          <div style="clear:both; "></div>
        </div>
        <div id="dictionary-devices-components"></div>
      </div>
      <div style="clear:both;"></div>
    </div>

    <div id="dictionary-instrs" class="application-workarea hidden">
      <div><button id="dictionary-instrs-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="margin-top:20px;">
        <div style="float:left; "><input type="text" size="1" name="instr2add" title="fill in new instr name (limit 3 digits), then press RETURN to save" /></div>
        <div style="float:left; padding-top:4px; color:maroon;">  &larr; add new instruction here</div>
        <div style="clear:both; "></div>
      </div>
      <div id="dictionary-instrs-instrs"></div>
    </div>

    <div id="search-cables" class="application-workarea hidden">
      <div style="border-bottom: 1px solid #000000;">
        <div style="float:left;">
          <form id="search-cables-form">
            <table style="font-size:95%;"><tbody>
              <tr><td><b>Cable #</b>     </td><td><input type="text" name="cable"           size="6"  value="" title="full or partial cable number"    ></input></td>
                  <td><b>Cable Type</b>  </td><td><input type="text" name="cable_type"      size="6"  value="" title="full or partial cable type"      ></input></td>
                  <td><b>Device</b>      </td><td><input type="text" name="device"          size="12" value="" title="full or partial device name"     ></input></td>
                  <td><b>Origin</b></td> <td><input type="text" name="origin_loc"           size="12" value="" title="full or partial origin name"     ></input></td></tr>
              <tr><td><b>Job #</b>       </td><td><input type="text" name="job"             size="6"  value="" title="full or partial job number"      ></input></td>
                  <td><b>Routing</b>     </td><td><input type="text" name="routing"         size="6"  value="" title="full or partial routing"         ></input></td>
                  <td><b>Function</b>    </td><td><input type="text" name="func"            size="12" value="" title="full or partial function name"   ></input></td>
                  <td><b>Destination</b> </td><td><input type="text" name="destination_loc" size="12" value="" title="full or partial destination name"></input></td></tr>
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
          <div style="float:right;" id="search-cables-info">&nbsp;</div>
          <div style="clear:both;"></div>
          <div id="search-cables-display">
            <div style=font-size:80%;">
              <table style="font-size:120%;"><tbody>
                <tr>
                  <td rowspan=2><button class="export" name="excel" title="Export into Microsoft Excel 2007 File"><img src="img/EXCEL_icon.gif" /></button></td>
                  <td><div style="width:20px;"></div></td>
                  <td><input type="checkbox" name="status"   checked="checked"></input>status</td>
                  <td><input type="checkbox" name="project"  checked="checked"></input>project</td>
                  <td><input type="checkbox" name="job"      checked="checked"></input>job #</td>
                  <td><input type="checkbox" name="cable"    checked="checked"></input>cable #</td>
                  <td><input type="checkbox" name="device"   checked="checked"></input>device</td>
                  <td><input type="checkbox" name="func"     checked="checked"></input>function</td>
                  <td rowspan=2><div style="width:20px;"></div></td>
                  <td rowspan=2><b>Sort by:</b></td>
                  <td rowspan=2><select name="sort" style="padding:1px;">
                        <option>status</option>
                        <option>project</option>
                        <option>job</option>
                        <option>cable</option>
                        <option>device</option>
                        <option>function</option>
                        <option>origin</option>
                        <option>destination</option>
                        <option>modified</option>
                      </select></td>
                  <td rowspan=2><div style="width:20px;"></div></td>
                  <td rowspan=2><button name="reverse">Show in Reverse Order</button></td>
                </tr>
                <tr>
                  <td colspan=1></td>
                  <td          ><input type="checkbox" name="length"   checked="checked"></input>length</td>
                  <td          ><input type="checkbox" name="routing"  checked="checked"></input>routing</td>
                  <td colspan=3><input type="checkbox" name="sd"                        ></input>expanded ORIGIN/DESTINATION</td>
                  <td          ><input type="checkbox" name="modified"                  ></input>modified</td>
                </tr>
              </tbody></table>
            </div>
          </div>
        </div>
        <div id="search-cables-result"></div>
      </div>
    </div>

    <div id="admin-access" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-access-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>

      <div style="margin-top:20px; margin-bottom:20px; width:720px;">
        <p>This section allows to assign user accounts to various roles defined in a context of the application.
        See a detailed description of each role in the corresponding subsection below.</p>
      </div>

      <div id="tabs" style="font-size:12px;">
        <ul>
          <li><a href="#administrators">Administrators</a></li>
          <li><a href="#projmanagers">Project Managers</a></li>
          <li><a href="#others">Other Users</a></li>
        </ul>

        <div id="administrators" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>Administrators posses highest level privileges in the application as they're allowed
              to perform any operation on projects, cables and other users. The only restriction is that
              an administrator is not allowed to remove their own account from the list of administrators.</p>
            </div>
            <div style="float:left; "><input type="text" size="8" name="administrator2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left; padding-top: 4px; color:maroon; "> &larr; add new user here</div>
            <div style="clear:both; "></div>
            <div id="admin-access-ADMINISTRATOR"></div>
          </div>
        </div>

        <div id="projmanagers" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>Project managers can create new projects, and, delete or edit cables, and also manage certain
              aspects of the cables life-cycle.</p>
            </div>
            <div style="float:left; "><input type="text" size="8" name="projmanager2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left; padding-top: 4px; color:maroon; "> &larr; add new user here</div>
            <div style="clear:both; "></div>
            <div id="admin-access-PROJMANAGER"></div>
          </div>
        </div>

        <div id="others" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>Other users may be allowed some limited access to manage certain aspects of the cables life-cycle.</p>
            </div>
            <div style="float:left; "><input type="text" size="8" name="other2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left; padding-top: 4px; color:maroon; "> &larr; add new user here</div>
            <div style="clear:both; "></div>
            <div id="admin-access-OTHER"></div>
          </div>
        </div>
      </div>
    </div>

    <div id="admin-notifications" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-notifications-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>

      <div style="margin-top:20px; margin-bottom:20px; width:720px;">
        <p>In order to avoid an excessive e-mail traffic the notification system
        will send just one message for any modification made in a specific context. For the very same
        reason the default behavior of the system is to send a summary daily message with all changes
        made before a time specified below, unless this site administrators choose a different policy
        (such as instantaneous notification).</p>
       </div>

      <div id="tabs" style="font-size:12px;">
        <ul>
          <li><a href="#myself">On my project(s)</a></li>
          <li><a href="#administrators">Sent to administrators</a></li>
          <li><a href="#others">Sent to other users</a></li>
          <li><a href="#pending">Pending</a></li>
        </ul>

        <div id="myself" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>This section is aiming at project managers who might be interested to track changes
              made to their projects by other people involved into various stages
              of the project workflow. Note that project managers will not get notifications
              on changes made by themselves.</p>
              <p>Notification settings found in this section can only be managed by project managers themselves
              or by administrators of the application.</p>
            </div>
            <div style="margin-bottom:20px;">
              <select name="policy4PROJMANAGER" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-PROJMANAGER"></div>
          </div>
        </div>

        <div id="administrators" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>This section is aiming at administrators of this software who might be interested to track major changes
              made to the projects, user accounts or software configuration. Note that administrators will not get notifications
              on changes made by themselves.</p>
              <p>Notification settings found in this section can only be managed by any administrator of the software.</p>
            </div>
            <div style="margin-bottom:20px;">
              <select name="policy4ADMINISTRATOR" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-ADMINISTRATOR"></div>
          </div>
        </div>

        <div id="others" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>This section is aiming at users (not necessarily project managers) who are involved
              into various stages of the project workflow.</p>
              <p>Only administrators of this application are
              allowed to modify notification settings found on this page.</p>
            </div>
            <div style="margin-bottom:20px;">
              <select name="policy4OTHER" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-OTHER"></div>
          </div>
        </div>

        <div id="pending" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>Pending/scheduled notifications (if any found below) can be submitted for instant delivery by pressing a group 'Submit' button or individually if needed.
              Notifications can also be deleted if needed. An additional dialog will be initiated to confirm group operations.</p>
              <p>Only administrators of this application are authorized for these operations.</p>
            </div>
            <div style="margin-bottom:20px;"">
              <button name="submit_all" title="Submit all pending notifications to be instantly delivered to their recipient">submit</button>
              <button name="delete_all" title="Delete all pending notifications">delete</button>
            </div>
            <div id="admin-notifications-pending"></div>
          </div>
        </div>

      </div>
    </div>

    <div id="admin-cablenumbers" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-cablenumbers-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div style="margin-top:20px; margin-bottom:20px; ">
        <p>PCDS is allocated a set of "official" cable identifiers (so called "cable numbers") which are managed by
        this application. A particular cable number begins with a two-letter prefix corresponding to a building where
        the cable "originates" from and it's followed by 4 digits. A unique cable number is generated each time
        a cable gets "registered" in the cable editor of the projects management tab.
        The current section is designed for configuring a generator of cable numbers and monitoring
        the allocation of the numbers.</p>
      </div>

      <div id="tabs" style="font-size:12px;">
        <ul>
          <li><a href="#cablenumbers">Prefixes</a></li>
          <li><a href="#orphan">Orphan Numbers</a></li>
          <li><a href="#reserved">Reserved Numbers</a></li>
        </ul>

        <div id="cablenumbers" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; ">
              <p>Each PCDS location (a building or an instrument) is associated with a set of cable numbers. The set represents
                 a family of cable numbers starting with so called 'prefix' (an upper case character string of the length of 2 or 3).
                 Multiple locations can share the same set (prefix). Each set is composed of one or many subranges of cable numbers.
                 Different prefixes have independent sets of numbers. Individual cable numbers are allocated from (any range) of a set
                 when a new cable is being registered in the database. This page is meant to be used by administrators to configure
                 the cable number generator.</p>
            </div>
            <div id="prefixes" style="float:left; margin-right:20px;">
              <div style="margin-bottom:10px;">
                <button name="edit">Edit</button>
                <button name="save">Save</button>
                <button name="cancel">Cancel</button>
              </div>
              <div id="admin-cablenumbers-prefixes-table"></div>
            </div>
            <div id="ranges" style="float:left; margin-right:20px;">
              <div style="margin-bottom:10px;">
                <button name="edit">Edit</button>
                <button name="save">Save</button>
                <button name="cancel">Cancel</button>
              </div>
              <div id="admin-cablenumbers-ranges-table"></div>
            </div>
            <div style="clear:both;"></div>
          </div>
        </div>
        <div id="orphan" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; ">
              <p>This page is meant for discovering cable numbers which are in use but which are not yet associated with any
                 allocation range known to this application. The numbers will be grouped according to their prefixes
                 and potential eligibility to be synchronized with allocation ranges. Administrators of this applications
                 will be allowed to synchronize the later numbers with the managed allocation ranges.
                 Note that some cables which were imported into this database from the "big" CAPTOR may not be synchronized.</p>
            </div>
            <div style="margin-bottom:10px;">
              <button name="scan">Scan</button>
              <button name="synchronize">Synchronize</button>
            </div>
            <div id="admin-orphan-table"></div>
          </div>
        </div>
        <div id="reserved" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; ">
              <p>This page will display cable numbers which are reserved but not used for specific cables.
                 And here follows an explanation of what it means. When a user "registers" a cable
                 for the first time (either on purpose or by a mistake) the cable gets some number
                 from the allocation pool. This will establish a permanent database associated between a cable
                 and its number. "Unregistering" the cable won't break this association, hence next time a user
                 "Registers" the same cable the application will always use the previously allocated number.
                 This page is meant to monitor cable numbers (and also find the corresponding cables) which  
                 are found in this intermediate state (of reserved by not really used).
                 Administrators of this applications will also be allowed to clean the permanent associations
                 for those cable and cable number pairs. Another possibility is to delete the corresponding cables
                 from their projects if they are not needed. This will free cable numbers allocated for those
                 (deleted) cables.
              </p>
            </div>
            <div style="margin-bottom:10px;">
              <button name="scan">Scan</button>
              <button name="free">Free</button>
            </div>
            <div id="admin-reserved-table"></div>
          </div>
        </div>
      </div>
    </div>

    <div id="admin-jobnumbers" class="application-workarea hidden">
      <div style="float:left;" ><button id="admin-jobnumbers-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both; "></div>
      <div style="margin-top:20px; margin-bottom:20px; width:720px;">
        <p>PCDS is allocated a set of "official" job identifiers (so called "job numbers") which are managed by
        this application. A particular job number begins with a three-letter prefix corresponding to a user who's
        responsible for the job and it's followed by 3 digits. A unique job number is generated each time
        a new project is created in the projects management tab.
        The current section is designed for configuring a generator of job numbers and monitoring
        the allocation of the numbers.</p>
      </div>

      <div id="tabs" style="font-size:12px;">
        <ul>
          <li><a href="#ranges">Ranges</a></li>
          <li><a href="#allocations">Allocated Numbers</a></li>
        </ul>

        <div id="ranges" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>Each user (including administrators and project managers) who's authorized to create new projects
              is assigned a range of cable numbers. The range is configured below.
              Only administrators of this application are allowed to modify the ranges.</p>
            </div>
            <div id="admin-jobnumbers-jobnumbers"></div>
          </div>
        </div>
      
        <div id="allocations" >
          <div style="font-size:11px; border:solid 1px #b0b0b0; padding:10px; padding-left:20px; padding-bottom:20px;" >
            <div style="margin-bottom:10px; width:720px;">
              <p>This section shows all job numbers allocated by the application.</p>
            </div>
            <div id="admin-jobnumbers-allocations"></div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <div id="popupdialogs" ></div>
  <div id="popupdialogs-varable-size" ></div>
  <div id="infodialogs" ></div>
  <div id="editdialogs" ></div>

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
