
/**
 * Utility class for creating tab-based applications.
 *
 * 1. This class requires that Fwk.css was loaded before calling configure()
 *
 * 2. No actions or DOM modifications are taken upon a construction of the
 *    framework object. The real stuff starts happening when the method configure()
 *    is called.
 */
function Fwk () {

    var that = this;

    this.title = null ;
    this.subtitle = null ;
    this.auth = null ;
    this.url = null ;

    this.is_configured = false ;

    this.configure = function (title, subtitle, auth, url) {

        if (this.is_configured) return ;
        this.is_configured = true ;

        this.title = title ;
        this.subtitle = subtitle ;
        this.auth = auth ;
        this.url = url ;

        this.start_vsplitter_manager() ;
        this.auth_timer_restart() ;

    } ;


    /* --------------------------------
     *   VERTICAL SPLITTER MANAGEMENT
     * --------------------------------
     */
 
    this.mouse_down = false ;

    this.resize = function () {
	$('#p-left'    ).height($(window).height()-125-20) ;
	$('#p-splitter').height($(window).height()-125-20) ;
	$('#p-center'  ).height($(window).height()-125-20) ;
    } ;

    /* Get mouse position relative to the document.
     */
    this.getMousePosition = function (e) {

        var posx = 0 ;
        var posy = 0 ;

        if (!e) var e = window.event ;
        if (e.pageX || e.pageY) {
            posx = e.pageX ;
            posy = e.pageY ;
        } else if (e.clientX || e.clientY) {
            posx = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft ;
            posy = e.clientY + document.body.scrollTop  + document.documentElement.scrollTop ;
        }
        return {'x': posx, 'y': posy } ;
    } ;

    this.move_split = function  (e) {
        var pos = this.getMousePosition(e) ;
        $('#p-left'    ).css('width',       pos['x']) ;
        $('#p-splitter').css('left',        pos['x']) ;
        $('#p-center'  ).css('margin-left', pos['x']+3) ;
    }

    this.start_vsplitter_manager = function() {

        $('body').attr('onresize', 'fwk.resize()') ;

        this.resize() ;

        $('#p-splitter').mousedown (function(e) { that.mouse_down = true ; return false ; }) ;

        $('#p-left'    ).mousemove(function(e) { if (that.mouse_down) that.move_split(e) ; });
        $('#p-center'  ).mousemove(function(e) { if (that.mouse_down) that.move_split(e) ; });

        $('#p-left'    ).mouseup  (function(e) { that.mouse_down = false ; });
        $('#p-splitter').mouseup  (function(e) { that.mouse_down = false ; });
        $('#p-center'  ).mouseup  (function(e) { that.mouse_down = false ; });
    } ;


    /* ---------------------------------------------
     *          AUTHENTICATION MANAGEMENT
     * ---------------------------------------------
     */
    var auth_is_authenticated="<?php echo $authdb->isAuthenticated()?>";
    var auth_type="<?php echo $authdb->authType()?>";
    var auth_remote_user="<?php echo $authdb->authName()?>";

    var auth_webauth_token_creation="<?php echo $_SERVER['WEBAUTH_TOKEN_CREATION']?>";
    var auth_webauth_token_expiration="<?php echo $_SERVER['WEBAUTH_TOKEN_EXPIRATION']?>";

    this.refresh_page = function () {
        window.location = this.url ;
    } ;

    this.auth_timer = null;

    function auth_timer_restart() {
        if( auth_is_authenticated && ( auth_type == 'WebAuth' ))
            auth_timer = window.setTimeout('fwk.auth_timer_event()', 1000 );
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





}

/* ATTENTION: This will only create an instance of the framework. No actions
 * or DOM modifications will be taken until it's configured and activated.
 */
var fwk = new Fwk() ;
