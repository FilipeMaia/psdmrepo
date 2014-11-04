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
        resizable: true,
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


