<?php

require_once 'authdb/authdb.inc.php' ;
require_once 'irep/irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use AuthDB\AuthDB ;
use AuthDB\AuthDBException ;

use Irep\Irep ;
use Irep\IrepException ;

use LusiTime\LusiTimeException ;


$document_title = 'PCDS Inventory And Repar Database:' ;
$document_subtitle = 'Electronic Equipment' ;

$required_field_html = '<span style="color:red ; font-size:110% ; font-weight:bold ;"> * </span>' ;

try {

    $authdb = AuthDB::instance() ;
    $authdb->begin() ;

    $irep = Irep::instance() ;
    $irep->begin() ;

?>


<!-- Document Begins Here -->


<!DOCTYPE html>
<html>

<head>
<title><?php echo $document_title ?></title>
<meta http-equiv="Content-Type" content="text/html ; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="css/common.css" rel="Stylesheet" />
<link type="text/css" href="css/irep.css" rel="Stylesheet" />
<link type="text/css" href="css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="js/Utilities.js"></script>
<script type="text/javascript" src="js/equipment.js"></script>
<script type="text/javascript" src="js/dictionary.js"></script>
<script type="text/javascript" src="js/admin.js"></script>
<script type="text/javascript" src="js/Table.js"></script>


<!-- Window layout styles and support actions -->

<style type="text/css">

body {
  margin: 0 ;
  padding: 0 ;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif ;
  font-size: 11px ;
}
#p-top {
  position: absolute ;
  top: 0 ;
  left: 0 ;
  width: 100% ;
  height: 129px ;
  background-color: #e0e0e0 ;
}
#p-top-header {
  position: absolute ;
  top: 0 ;
  left: 0 ;
  width: 100% ;
  height: 92px ;
  background-color: #ffffff ;
}
#p-top-title {
  width: 100% ;
  height: 61px ;
}
#p-context-header {
  width: 100% ;
  height: 35px ;
  background-color: #E0E0E0 ;
  border-bottom: 2px solid #a0a0a0 ;
}
#p-title,
#p-subtitle {
  font-family: "Times", serif ;
  font-size: 32px ;
  font-weight: bold ;
  text-align: left ;
}
#p-subtitle {
  margin-left: 10px ;
  color: #0071bc ;
}
#p-login {
  padding-top:   15px ;
  padding-right: 10px ;
  font-size:     11px ;
  font-family:   Arial, Helvetica, Verdana, Sans-Serif ;
}

a, a.link {
  text-decoration: none ;
  font-weight: bold ;
  color: #0071bc ;
}
a:hover, a.link:hover {
  color: red ;
}
#p-left {
  position: absolute ;
  left: 0 ;
  top: 130px ;
  width: 200px ;
  overflow: auto ;
}
#p-splitter {
  position: absolute ;
  left: 200px ;
  top: 130px ;
  width: 1px ;
  overflow: none ;
  cursor: e-resize ;
  border-left: 1px solid #a0a0a0 ;
  border-right: 1px solid #a0a0a0 ;
}
#p-center {
  position: relative ;
  top:130px ;
  margin: 0px 0px 20px 203px ;
  overflow: auto ;
  background-color: #ffffff ;
  border-left: 1px solid #a0a0a0 ;
}

#p-menu {
  font-family: Arial, sans-serif ;
  font-size: 14px ;
  height: 32px ;
  width: 100% ;
  border: 0 ;
  padding: 0 ;
}

#p-context {
  margin-left: 0px ;
  padding-top: 10px ;
  padding-left: 10px ;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif ;
  font-size: 12px ;
  font-weight: bold ;
}
#p-search {
  padding-top: 2px ;
  padding-right: 10px ;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif ;
  font-size: 11px ;
}

div.m-item {

  margin-left: 3px ;
  margin-top: 5px ;

  padding: 5px ;
  padding-left: 10px ;
  padding-right: 10px ;

  background: #DFEFFC url(/jquery/css/custom-theme/images/ui-bg_glass_85_dfeffc_1x400.png) 50% 50% repeat-x ;

  color: #0071BC ;

  border-right: 2px solid #a0a0a0 ;

  border-radius: 5px ;
  border-bottom-left-radius: 0 ;
  border-bottom-right-radius: 0 ;

  -moz-border-radius: 5px ;
  -moz-border-radius-bottomleft: 0 ;
  -moz-border-radius-bottomright: 0 ;

  cursor: pointer ;
}

div.m-item:hover {
  background: #d0e5f5 url(/jquery/css/custom-theme/images/ui-bg_glass_75_d0e5f5_1x400.png) 50% 50% repeat-x ;
}
div.m-item-first {
  margin-left: 0px ;
  float: left ;

  border-top-left-radius: 0 ;

  -moz-border-radius-topleft: 0 ;
}
.m-item-next {
  float: left ;
}
.m-item-last {
  float: left ;
}
.m-item-end {
  clear: both ;
}
div.m-select {
  font-weight: bold ;
  background: #e0e0e0 ;
}

#v-menu {
  width: 100% ;
  height: 100% ;
  background: url('img/menu-bg-gradient-4.png') repeat ;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif ;
  font-size: 12px ;
}
#menu-title {
  height: 10px ;
}
div.v-item {
  padding: 4px ;
  padding-left: 10px ;
  cursor: pointer ;
}
div.v-item:hover {
  background:#f0f0f0 ;
}
.v-select {
  font-weight: bold ;
}
.application-workarea {
  overflow: auto ;
  padding: 20px ;
  /*font-family: Lucida Grande, Lucida Sans, Arial, sans-serif ;*/
  /*font-size: 75% ;*/
}
.section1,
.section2,
.section3 {
  margin: 0.25em 0 0.25em 0 ;
  padding: 0.25em ;
  padding-left: 0.5em ;

  /*
  background-color: #DEF0CD ;
  */
  border: 2px solid #a0a0a0 ;

  border-left:0 ;
  border-top:0 ;
  border-right:0 ;
  /*
  border-radius: 5px ;
  -moz-border-radius: 5px ;
  */
  font-family: "Times", serif ;
  font-size: 36px ;
  font-weight: bold ;
  text-align:left ;
}
.section2 {
  font-size: 28px ;
}
.section3 {
  font-size: 18px ;
}
.hidden  { display: none ; }
.visible { display: block ; }

#popupdialogs {
  display: none ;
  font-size: 11px ;
}
#popupdialogs-varable-size {
  display: none ;
  font-size: 11px ;
}

#infodialogs {
  display: none ;
  padding: 20px ;
  font-size: 11px ;
}
#editdialogs {
  display: none ;
  padding: 20px ;
  font-size: 11px ;
}

</style>


<script type="text/javascript">


/* ------------------------------------------------
 *          VERTICAL SPLITTER MANAGEMENT
 * ------------------------------------------------
 */
function resize () {
    $('#p-left').height($(window).height()-125-5) ;
    $('#p-splitter').height($(window).height()-125-5) ;
    $('#p-center').height($(window).height()-125-5) ;
}

/* Get mouse position relative to the document.
 */
function getMousePosition (e) {

    var posx = 0 ;
    var posy = 0 ;
    if (!e) var e = window.event ;
    if (e.pageX || e.pageY)     {
        posx = e.pageX ;
        posy = e.pageY ;
    }
    else if (e.clientX || e.clientY)     {
        posx = e.clientX + document.body.scrollLeft
            + document.documentElement.scrollLeft ;
        posy = e.clientY + document.body.scrollTop
            + document.documentElement.scrollTop ;
    }
    return {'x': posx, 'y': posy } ;
}

function move_split (e) {
    var pos = getMousePosition(e) ;
    $('#p-left').css('width', pos['x']) ;
    $('#p-splitter').css('left', pos['x']) ;
    $('#p-center').css('margin-left', pos['x']+3) ;
}

$(function () {

    resize() ;

    var mouse_down = false ;

    $('#p-splitter').mousedown (function(e) { mouse_down = true ; return false ; }) ;

    $('#p-left'    ).mousemove(function(e) { if (mouse_down) move_split(e) ; }) ;
    $('#p-center'  ).mousemove(function(e) { if (mouse_down) move_split(e) ; }) ;

    $('#p-left'    ).mouseup   (function(e) { mouse_down = false ; }) ;
    $('#p-splitter').mouseup   (function(e) { mouse_down = false ; }) ;
    $('#p-center'  ).mouseup   (function(e) { mouse_down = false ; }) ;
}) ;

/* ---------------------------------------------
 *          AUTHENTICATION MANAGEMENT
 * ---------------------------------------------
 */
var auth_is_authenticated="<?php echo $authdb->isAuthenticated()?>" ;
var auth_type="<?php echo $authdb->authType()?>" ;
var auth_remote_user="<?php echo $authdb->authName()?>" ;

var auth_webauth_token_creation="<?php echo $_SERVER['WEBAUTH_TOKEN_CREATION']?>" ;
var auth_webauth_token_expiration="<?php echo $_SERVER['WEBAUTH_TOKEN_EXPIRATION']?>" ;

function refresh_page() {
    window.location = "<?php echo $_SERVER['REQUEST_URI']?>" ;
}

var auth_timer = null ;
function auth_timer_restart() {
    if (auth_is_authenticated && (auth_type == 'WebAuth'))
        auth_timer = window.setTimeout('auth_timer_event()', 1000) ;
}

var auth_last_secs = null  ;
function auth_timer_event() {

    var auth_expiration_info = document.getElementById('auth_expiration_info') ;
    var now = mktime() ;
    var seconds = auth_webauth_token_expiration - now ;
    if (seconds <= 0) {
        $('#popupdialogs').html(
            '<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+
            'Your WebAuth session has expired. Press <b>Ok</b> or use <b>Refresh</b> button'+
            'of the browser to renew your credentials.</p>'
        ) ;
        $('#popupdialogs').dialog({
            resizable: false,
            modal: true,
            buttons: {
                'Ok': function() {
                    $(this).dialog('close') ;
                    refresh_page() ;
                }
            },
            title: 'Session Expiration Notification'
        }) ;
        return ;
    }
    var hours_left   = Math.floor(seconds / 3600) ;
    var minutes_left = Math.floor((seconds % 3600) / 60) ;
    var seconds_left = Math.floor((seconds % 3600) % 60) ;

    var hours_left_str = hours_left ;
    if (hours_left < 10) hours_left_str = '0'+hours_left_str ;
    var minutes_left_str = minutes_left ;
    if (minutes_left < 10) minutes_left_str = '0'+minutes_left_str ;
    var seconds_left_str = seconds_left ;
    if (seconds_left < 10) seconds_left_str = '0'+seconds_left_str ;

    auth_expiration_info.innerHTML=
        '<b>'+hours_left_str+':'+minutes_left_str+'.'+seconds_left_str+'</b>' ;

    auth_timer_restart() ;
}

function logout () {
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+
        'This will log yout out from the current WebAuth session. Are you sure?</p>'
    ) ;
    $('#popupdialogs').dialog({
        resizable: false,
        modal: true,
        buttons: {
            "Yes": function() {
                $(this).dialog('close') ;
                document.cookie = 'webauth_wpt_krb5= ; expires=Fri, 27 Jul 2001 02:47:11 UTC ; path=/' ;
                document.cookie = 'webauth_at= ; expires=Fri, 27 Jul 2001 02:47:11 UTC ; path=/' ;
                refresh_page() ;
            },
            Cancel: function() {
                $(this).dialog('close') ;
            }
        },
        title: 'Session Logout Warning'
    }) ;
}

$(function () {
    auth_timer_restart() ;
}) ;

/* ----------------------------------------------
 *             CONTEXT MANAGEMENT
 * ----------------------------------------------
 */
var current_tab = '' ;

function set_current_tab (tab) {
    current_tab = tab ;
}

function set_context (app) {
    var ctx = app.full_name+' :' ;
    if (app.context) ctx += ' '+app.context ;
    $('#p-context').html(ctx) ;
}

/* ----------------------------------------------
 *             UTILITY FUNCTIONS
 * ----------------------------------------------
 */
function show_email (user, addr) {
    $('#popupdialogs').html('<p>'+addr+'</p>') ;
    $('#popupdialogs').dialog({
        modal: true,
        title: 'e-mail: '+user
    }) ;
}

function printer_friendly () {
    if (current_application != null) {
        var wa_id = current_application.name ;
        if (current_application.context != '') wa_id += '-'+current_application.context ;
        $('#p-center .application-workarea#'+wa_id).printElement({
            leaveOpen: true,
            printMode: 'popup',
            printBodyOptions: {
                styleToAdd:'font-size:10px ;'
            }
        }) ;
    }    
}

function ask_yes_no (title, msg, on_yes, on_cancel) {
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+msg+'</p>'
    ) ;
    $('#popupdialogs').dialog({
        resizable: false,
        modal: true,
        buttons: {
            "Yes": function() {
                $(this).dialog('close') ;
                if (on_yes) on_yes() ;
            },
            Cancel: function() {
                $(this).dialog('close') ;
                if (on_cancel) on_cancel() ;
            }
        },
        title: title
    }) ;
}

function ask_for_input (title, msg, on_ok, on_cancel) {
    $('#popupdialogs-varable-size').html(
'<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+msg+'</p>'+
'<div><textarea rows=4 cols=60></textarea/>'
    ) ;
    $('#popupdialogs-varable-size').dialog({
        resizable: true,
        modal: true,
        width:  470,
        height: 300,
        buttons: {
            "Ok": function() {
                var user_input = $('#popupdialogs-varable-size').find('textarea').val() ;
                $(this).dialog('close') ;
                if (on_ok) on_ok(user_input) ;
            },
            Cancel: function() {
                $(this).dialog('close') ;
                if (on_cancel) on_cancel() ;
            }
        },
        title: title
    }) ;
}

function report_error (msg, on_cancel) {
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+msg+'</p>'
    ) ;
    $('#popupdialogs').dialog({
        resizable: false,
        modal: true,
        buttons: {
            Cancel: function() {
                $(this).dialog('close') ;
                if (on_cancel) on_cancel() ;
            }
        },
        title: 'Error'
    }) ;
}

function report_info (title, msg) {
    $('#infodialogs').html(msg) ;
    $('#infodialogs').dialog({
        resizable: true,
        modal: true,
        title: title
    }) ;
}
function report_info_table (title, hdr, rows) {
    var table = new Table('infodialogs', hdr, rows) ;
    table.display() ;
    $('#infodialogs').dialog({
        width: 720,
        height: 800,
        resizable: true,
        modal: true,
        title: title
    }) ;
}
function report_action (title, msg) {
    return $('#infodialogs').
        html(msg).
        dialog({
            resizable: true,
            modal: true,
            buttons: {
                Cancel: function() {
                    $(this).dialog('close') ;
                }
            },
            title: title
        }) ;
}

function edit_dialog (title, msg, on_save, on_cancel) {
    $('#editdialogs').html(msg) ;
    $('#editdialogs').dialog({
        resizable: true,
        width: 640,
        modal: true,
        buttons: {
            Save: function() {
                $(this).dialog('close') ;
                if (on_save != null) on_save() ;
            },
            Cancel: function() {
                $(this).dialog('close') ;
                if (on_cancel != null) on_cancel() ;
            }
        },
        title: title
    }) ;
}

/* ------------------------------------------------------
 *             APPLICATION INITIALIZATION
 * ------------------------------------------------------
 */
var global_current_user = {
    uid:               '<?php echo $authdb->authName        () ;         ?>' ,
    is_other:           <?php echo $irep->is_other          ()?'1':'0' ; ?>  ,
    is_administrator:   <?php echo $irep->is_administrator  ()?'1':'0' ; ?>  ,
    can_edit_inventory: <?php echo $irep->can_edit_inventory()?'1':'0' ; ?>  ,
    has_dict_priv:      <?php echo $irep->has_dict_priv     ()?'1':'0' ; ?>
} ;

var global_users   = [] ;
var global_editors = [] ;
<?php
    foreach ($irep->users() as $user) {
        echo "global_users.push('{$user->uid()}') ;\n" ;
        if ($user->is_administrator() || $user->is_editor()) echo "global_editors.push('{$user->uid()}') ;\n" ;
    }
?>
function global_get_editors() {
    var editors = admin.editors() ;
    if (editors) return editors ;
    return global_editors ;
}
var applications = {
    'p-appl-equipment'  : equipment,
    'p-appl-dictionary' : dict,
    'p-appl-admin'      : admin
} ;

var current_application = null ;

var select_app         = 'equipment' ;
var select_app_context = 'inventory' ;
<?php
$known_apps = array(
    'equipment'  => True,
    'dictionary' => True,
    'admin'      => True
) ;
if (isset($_GET['app'])) {
    $app_path = explode(':', strtolower(trim($_GET['app']))) ;
    $app = $app_path[0] ;
    if (array_key_exists($app, $known_apps)) {
        echo "select_app = '{$app}' ;" ;
        echo "select_app_context = '".(count($app_path) > 1 ? $app_path[1] : "")."' ;" ;
    }
}
?>
var select_params = {
    equipment_id: null
} ;
<?php
if (isset($_GET['equipment_id'])) {
    $equipment_id = intval(trim($_GET['equipment_id'])) ;
    if ($equipment_id) echo "select_params.equipment_id = {$equipment_id} ;" ;
}
?>
/* Event handler for application selections from the top-level menu bar:
 * - fill set the current application context.
 */
function m_item_selected(item) {

    if (current_application == applications[item.id]) return ;
    if ((current_application != null) && (current_application != applications[item.id])) {
        current_application.if_ready2giveup(function() {
            m_item_selected_impl(item) ;
        }) ;
        return ;
    }
    m_item_selected_impl(item) ;
}

function m_item_selected_impl(item) {

    current_application = applications[item.id] ;

    $('.m-select').removeClass('m-select') ;
    $(item).addClass('m-select') ;
    $('#p-left > #v-menu .visible').removeClass('visible').addClass('hidden') ;
    $('#p-left > #v-menu > #'+current_application.name).removeClass('hidden').addClass('visible') ;

    $('#p-center .application-workarea.visible').removeClass('visible').addClass('hidden') ;
    var wa_id = current_application.name ;
    if (current_application.context != '') wa_id += '-'+current_application.context ;
    $('#p-center .application-workarea#'+wa_id).removeClass('hidden').addClass('visible') ;

    current_application.select_default() ;
    v_item_selected($('#v-menu > #'+current_application.name).children('.v-item#'+current_application.context)) ;
    
    set_context(current_application) ;
}

/* Event handler for vertical menu item (actual commands) selections:
 * - dim the poreviously active item
 * - hightlight the new item
 * - change the current context
 * - execute the commands
 * - switch the work area (make the old one invisible, and the new one visible)
 */
function v_item_selected(item) {

     var item = $(item) ;
    if ($(item).hasClass('v-select')) return ;

    if (current_application.context != item.attr('id')) {
        current_application.if_ready2giveup(function() {
            v_item_selected_impl(item) ;
        }) ;
        return ;
    }
    v_item_selected_impl(item) ;
}

function v_item_selected_impl(item) {

    $('#'+current_application.name).find('.v-item.v-select').each(function(){
        $(this).children('.ui-icon').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
        $(this).removeClass('v-select') ;
    }) ;

    $(item).children('.ui-icon').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
    $(item).addClass('v-select') ;

    /* Hide the older work area
     */
    var wa_id = current_application.name ;
    if (current_application.context != '') wa_id += '-'+current_application.context ;
    $('#p-center > #application-workarea > #'+wa_id).removeClass('visible').addClass('hidden') ;

    current_application.select(item.attr('id')) ;

    /* display the new work area
     */
    wa_id = current_application.name ;
    if (current_application.context != '') wa_id += '-'+current_application.context ;
    $('#p-center > #application-workarea > #'+wa_id).removeClass('hidden').addClass('visible') ;

    set_context(current_application) ;
}

$(function() {

    $('.m-item').click(function() { m_item_selected (this) ; }) ;
    $('.v-item').click(function() { v_item_selected (this) ; }) ;

    $('#p-search-text').keyup(function(e) { if (($(this).val() != '') && (e.keyCode == 13)) global_simple_search() ; }) ;

    // Make sure the dictionaries are loaded
    //
    dict.init() ;
    admin.init() ;

    // Finally, activate the selected application.
    //
    for (var id in applications) {
        var application = applications[id] ;
        if (application.name == select_app) {
            $('#p-menu').children('#p-appl-'+select_app).each(function() { m_item_selected(this) ; }) ;
            if ('' == select_app_context) {
                v_item_selected($('#v-menu > #'+select_app+' > #'+application.default_context)) ;
                application.select_default() ;
            } else {
                v_item_selected($('#v-menu > #'+select_app+' > #'+select_app_context)) ;
                application.select(select_app_context) ;            
            }
            switch(application.name) {
            case 'equipment':
                switch(application.context) {
                case 'inventory':
                    if (select_params.equipment_id) global_search_equipment_by_id(select_params.equipment_id) ;
                    break ;
                }
                break ;
            }
        }
    }
}) ;

/* ------------------------------------------------------
 *             CROSS_APPLICATION EVENT HANDLERS
 * ------------------------------------------------------
 */
function global_switch_context(application_name, context_name) {
    for (var id in applications) {
        var application = applications[id] ;
        if (application.name == application_name) {
            $('#p-menu').children('#'+id).each(function() {    m_item_selected(this) ; }) ;
            v_item_selected($('#v-menu > #'+application_name).children('.v-item#'+context_name)) ;
            if (context_name != null) application.select(context_name) ;
            else application.select_default() ;
            return application ;
        }
    }
    return null ;
}
function global_simple_search                            ()   { global_switch_context('equipment', 'inventory').simple_search($('#p-search-text').val()) ; }
function global_search_equipment_by_id                   (id) { global_switch_context('equipment', 'inventory').search_equipment_by_id(id) ; }
function global_search_equipment_by_dict_location_id     (id) { global_switch_context('equipment', 'inventory').search_equipment_by_dict_location_id(id) ; }
function global_search_equipment_by_dict_manifacturer_id (id) { global_switch_context('equipment', 'inventory').search_equipment_by_dict_manifacturer_id(id) ; }
function global_search_equipment_by_dict_model           (id) { global_switch_context('equipment', 'inventory').search_equipment_by_dict_model(id) ; }

function global_export_equipment(search_params,outformat) {
    search_params.format = outformat ;
    var html = '<img src="../logbook/images/ajaxloader.gif" />' ;
    var dialog = report_action('Generating Document: '+outformat,html) ;
    var jqXHR = $.get(
        '../irep/ws/equipment_inventory_search.php', search_params,
        function(data) {
            if (data.status != 'success') {
                report_error(data.message) ;
                dialog.dialog('close') ;
                return ;
            }
            var html = 'Document is ready to be downloaded from this location: <a class="link" href="'+data.url+'" target="_blank" >'+data.name+'</a>' ;
            dialog.html(html) ;
        },
        'JSON'
    ).error(
        function () {
            report_error('failed because of: '+jqXHR.statusText) ;
            dialog.dialog('close') ;
        }
    ).complete(
        function () {
        }
    ) ;
}

function global_equipment_status2rank(status) {
    switch(status) {
        case 'Unknown': return 0 ;
    }
    return -1 ;
}
function global_equipment_sorter_by_status       (a,b) { return global_equipment_status2rank(a.status) - global_equipment_status2rank(b.status) ; }
function sort_as_text                            (a,b) { return a == b ? 0 : (a < b ? -1 : 1) ; }
function global_equipment_sorter_by_manufacturer (a,b) { return sort_as_text(a.manufacturer, b.manufacturer) ; }
function global_equipment_sorter_by_model        (a,b) { return sort_as_text(a.model,        b.model) ; }
function global_equipment_sorter_by_location     (a,b) { return sort_as_text(a.location,     b.location) ; }
function global_equipment_sorter_by_modified     (a,b) { return a.modified.time_64 - b.modified.time_64 ; }

</script>

</head>

<body onresize="resize()">

<div id="p-top">
  <div id="p-top-header">
    <div id="p-top-title">
      <div style="float:left ; padding-left:15px ; padding-top:10px ;">
        <span id="p-title"><?php echo $document_title?></span>
        <span id="p-subtitle"><?php echo $document_subtitle?></span>
      </div>
      <div id="p-login" style="float:right ;" >
        <div style="float:left ; padding-top:20px ;" class="not4print" >
          <a href="javascript:printer_friendly()" title="Printer friendly version of this page"><img src="img/PRINTER_icon.gif" style="border-radius: 5px ;" /></a>
        </div>
        <div style="float:left ; margin-left:10px ;" >
          <table><tbody>
            <tr>
              <td>&nbsp ;</td>
              <td>[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]</td></tr>
            <tr>
              <td>User:&nbsp ;</td>
              <td><b><?php echo $authdb->authName()?></b></td></tr>
            <tr>
              <td>Session expires in:&nbsp ;</td>
              <td id="auth_expiration_info"><b>00:00.00</b></td></tr>
          </tbody></table>
        </div>
        <div style="clear:both ;" class="not4print"></div>
      </div>
      <div style="clear:both ;"></div>
    </div>
    <div id="p-menu">
      <div class="m-item m-item-first m-select" id="p-appl-equipment" >Equipment</div>
      <div class="m-item m-item-next"           id="p-appl-dictionary">Dictionary</div>
      <div class="m-item m-item-last"           id="p-appl-admin"     >Admin</div>
      <div class="m-item-end"></div>
    </div>
    <div id="p-context-header">
      <div id="p-context" style="float:left"></div>
      <div id="p-search" style="float:right">
        quick search: <input type="text" id="p-search-text" value="" size=16 title="enter full or partial attribute of equipment to search, then press RETURN to proceed"  style="font-size:80% ; padding:1px ; margin-top:6px ;" />
      </div>
      <div style="clear:both ;"></div>
    </div>
  </div>
</div>

<div id="p-left">

<div id="v-menu">

    <div id="menu-title"></div>

    <div id="equipment" class="visible">
      <div class="v-item" id="inventory">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left ;"></div>
        <div style="float:left ;" >Inventory</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="add">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div class="link" style="float:left ;" >Add New Equipment</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

    <div id="dictionary" class="hidden">
      <div class="v-item" id="manifacturers">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Manufacturers and Models</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="locations">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Locations</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

    <div id="admin" class="hidden">
      <div class="v-item" id="access">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Access Control</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="notifications">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >E-mail Notifications</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="slacid">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >SLACid Numbers</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

  </div>
</div>

<div id="p-splitter"></div>

<div id="p-center">
  <div id="application-workarea">

    <!-- An interface for displaying an inventory of all known equipment -->
    <div id="equipment-inventory" class="application-workarea hidden">

      <!-- Controls for selecting equipment for display and updating the list of
        -- the selected equipment.
        -->
      <div id="equipment-inventory-controls">
        <div style="float:left ;">
          Here be the controls.
        </div>
        <div style="float:left ; margin-left:20px ;">
          <button name="search" title="refresh the list">Search</button>
          <button name="reset"  title="reset the search form to the default state">Reset Form</button>
        </div>
      </div>
      <div style="clear:both ;"></div>
      <div style="float:right ;" id="equipment-inventory-info">&nbsp ;</div>
      <div style="clear:both ;"></div>

      <!-- The table to display equipment selection -->
      <div id="equipment-inventory-display">
        Here be the table with a list of equipment. And perhaps sorting controls.
      </div>

    </div>

    <div id="equipment-add" class="application-workarea hidden">
<?php
    if ($irep->can_edit_inventory())
        print <<<HERE
      <div style="margin-bottom:20px ; border-bottom:1px dashed #c0c0c0 ;">
        <div style="float:left ;">
          <div style="margin-bottom:10px ; width:480px ;">
            When making a clone of an existing equipment record make sure the SLAC Property Control Number (PC)
            of the new equipment differs from the original one. All other attributes of the original equipment
            will be copied into the new one. The copied equipment will all be put into the 'Unknown' state.
          </div>
          <form id="equipment-add-form">
            <table><tbody>
              <tr><td><b>Manifacturer:</b></td>
                  <td><input type="text" name="manifacturer" size="16" class="equipment-add-form-element" style="padding:2px ;" value="" /></td></tr>
              <tr><td>&nbsp ;</td></tr>
              <tr><td><b>Model:{$required_field_html}</b></td>
                  <td><input type="text" name="model" size="5" class="equipment-add-form-element" style="padding:2px ;" value="" /></td></tr>
              <tr><td><b>Property Control #:{$required_field_html}</b></td>
                  <td><input type="text" name="property_control_number"  size="50" class="equipment-add-form-element" style="padding:2px ;" value="" /></td></tr>
              <tr><td><b>Descr: </b></td>
                  <td colspan="4"><textarea cols=54 rows=4 name="description" class="equipment-add-form-element" style="padding:4px ;" title="Here be an arbitrary description"></textarea></td></tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left ; padding:5px ;">
          <div>
            <button id="equipment-add-save">Create</button>
            <button id="equipment-add-reset">Reset Form</button>
          </div>
          <div style="margin-top:5px ;" id="equipment-add-info" >&nbsp ;</div>
        </div>
        <div style="clear:both ;"></div>
      </div>
      {$required_field_html} required feild
HERE;
      else {
          $admin_access_href = "javascript:global_switch_context('admin','access')" ;
        print <<<HERE
<br><br>
<center>
  <span style="color: red ; font-size: 175% ; font-weight: bold ; font-family: Times, sans-serif ;">
    A c c e s s &nbsp ; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10% ; padding: 10px ; font-size: 125% ; font-family: Times, sans-serif ; border-top: 1px solid #b0b0b0 ;">
  We're sorry! Your SLAC UNIX account <b>{$authdb->authName()}</b> has no sufficient permissions for this operation.
  Normally we assign this task to authorized <a href="{$admin_access_href}">database editors</a>.
  Please contact administrators of this application if you think you need to add/edit equipment records.
  A list of administrators can be found in the <a href="{$admin_access_href}">Access Control</a> section of the <a href="{$admin_access_href}">Admin</a> tab of this application.
</div>
HERE;
      }
?>
    </div>

    <div id="dictionary-manifacturers" class="application-workarea hidden">
      <div><button id="dictionary-manifacturers-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="manifacturer2add" title="fill in new manifacturer name, then press RETURN to save" /></div>
              <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new manifacturer here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-manifacturers-manifacturers"></div>
      </div>
      <div style="float:left ; margin-left:20px ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="model2add" title="fill in new model name, then press RETURN to save" /></div>
          <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new rack here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-manifacturers-models"></div>
      </div>
      <div style="clear:both ; "></div>
    </div>

    <div id="dictionary-locations" class="application-workarea hidden">
      <div><button id="dictionary-locations-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="location2add" title="fill in new location name, then press RETURN to save" /></div>
              <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new location here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-locations-locations"></div>
      </div>
      <div style="clear:both ; "></div>
    </div>

    <div id="admin-access" class="application-workarea hidden">
      <div style="float:left ;" ><button id="admin-access-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both ; "></div>

      <div style="margin-top:20px ; margin-bottom:20px ; width:720px ;">
        <p>This section allows to assign user accounts to various roles defined in a context of the application.
        See a detailed description of each role in the corresponding subsection below.</p>
      </div>

      <div id="tabs" style="font-size:12px ;">
        <ul>
          <li><a href="#administrators">Administrators</a></li>
          <li><a href="#editors">Editors</a></li>
          <li><a href="#others">Other Users</a></li>
        </ul>

        <div id="administrators" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Administrators posses highest level privileges in the application as they're allowed
              to perform any operation on the inventory and other users. The only restriction is that
              an administrator is not allowed to remove their own account from the list of administrators.</p>
            </div>
            <div style="float:left ; "><input type="text" size="8" name="administrator2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left ; padding-top: 4px ; color:maroon ; "> &larr; add new user here</div>
            <div style="clear:both ; "></div>
            <div id="admin-access-ADMINISTRATOR"></div>
          </div>
        </div>

        <div id="editors" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Editors can add new equipment to the inventory, delete or edit existing records of the equipment
              and also manage certain aspects of the equipment life-cycle.</p>
            </div>
            <div style="float:left ; "><input type="text" size="8" name="editor2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left ; padding-top: 4px ; color:maroon ; "> &larr; add new user here</div>
            <div style="clear:both ; "></div>
            <div id="admin-access-EDITOR"></div>
          </div>
        </div>

        <div id="others" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Other users may be allowed some limited access to manage certain aspects of the equipment life-cycle.</p>
            </div>
            <div style="float:left ; "><input type="text" size="8" name="other2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left ; padding-top: 4px ; color:maroon ; "> &larr; add new user here</div>
            <div style="clear:both ; "></div>
            <div id="admin-access-OTHER"></div>
          </div>
        </div>
      </div>
    </div>

    <div id="admin-notifications" class="application-workarea hidden">
      <div style="float:left ;" ><button id="admin-notifications-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both ; "></div>

      <div style="margin-top:20px ; margin-bottom:20px ; width:720px ;">
        <p>In order to avoid an excessive e-mail traffic the notification system
        will send just one message for any modification made in a specific context. For the very same
        reason the default behavior of the system is to send a summary daily message with all changes
        made before a time specified below, unless this site administrators choose a different policy
        (such as instantaneous notification).</p>
       </div>

      <div id="tabs" style="font-size:12px ;">
        <ul>
          <li><a href="#myself">On my equipment</a></li>
          <li><a href="#administrators">Sent to administrators</a></li>
          <li><a href="#others">Sent to other users</a></li>
          <li><a href="#pending">Pending</a></li>
        </ul>

        <div id="myself" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>This section is aiming at editors who might be interested to track changes
              made to their equipment by other people involved into various stages
              of the workflow. Note that editors will not get notifications
              on changes made by themselves.</p>
              <p>Notification settings found in this section can only be managed by editors themselves
              or by administrators of the application.</p>
            </div>
            <div style="margin-bottom:20px ;">
              <select name="policy4EDITOR" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-EDITOR"></div>
          </div>
        </div>

        <div id="administrators" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>This section is aiming at administrators of this software who might be interested to track major changes
              made to the equipment, user accounts or software configuration. Note that administrators will not get notifications
              on changes made by themselves.</p>
              <p>Notification settings found in this section can only be managed by any administrator of the software.</p>
            </div>
            <div style="margin-bottom:20px ;">
              <select name="policy4ADMINISTRATOR" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-ADMINISTRATOR"></div>
          </div>
        </div>

        <div id="others" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>This section is aiming at users (not necessarily editors) who are involved
              into various stages of the equipment workflow.</p>
              <p>Only administrators of this application are allowed to modify notification settings found on this page.</p>
            </div>
            <div style="margin-bottom:20px ;">
              <select name="policy4OTHER" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-OTHER"></div>
          </div>
        </div>

        <div id="pending" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Pending/scheduled notifications (if any found below) can be submitted for instant delivery by pressing a group 'Submit' button or individually if needed.
              Notifications can also be deleted if needed. An additional dialog will be initiated to confirm group operations.</p>
              <p>Only administrators of this application are authorized for these operations.</p>
            </div>
            <div style="margin-bottom:20px ;"">
              <button name="submit_all" title="Submit all pending notifications to be instantly delivered to their recipient">submit</button>
              <button name="delete_all" title="Delete all pending notifications">delete</button>
            </div>
            <div id="admin-notifications-pending"></div>
          </div>
        </div>

      </div>
    </div>

    <div id="admin-slacidnumbers" class="application-workarea hidden">
      <div style="float:left ;" ><button id="admin-slacidnumbers-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both ; "></div>
      <div style="margin-top:20px ; margin-bottom:20px ; ">
        <p>PCDS is allocated a set of "official" SLACid numbers which are managed by
        this application. A unique number is generated each time
        a new equipment gets registered in the Inventory database.
        The current section is designed for configuring a generator of numbers and monitoring
        the allocation of the numbers.</p>
      </div>

      <div id="tabs" style="font-size:12px ;">
        <ul>
          <li><a href="#slacidnumbers">Ranges</a></li>
        </ul>

        <div id="slacidnumbers" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; ">
              <p>This page is meant to be used by administrators to configure the generator/allocator of numbers.</p>
            </div>
            <div id="ranges">
              <div style="margin-bottom:10px ;">
                <button name="edit">Edit</button>
                <button name="save">Save</button>
                <button name="cancel">Cancel</button>
              </div>
              <div id="admin-slacidnumbers-ranges-table"></div>
            </div>
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

    $authdb->commit() ;
    $irep->commit() ;

} catch(AuthDBException   $e) { print $e->toHtml() ; }
  catch(LusiTimeException $e) { print $e->toHtml() ; }
  catch(IrepException     $e) { print $e->toHtml() ; }

?>
