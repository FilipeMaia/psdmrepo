/*
 * An interface providing persistent support for run-time configurations
 * of Web applications.
 */
function FwkConfigHandlerCreator (application_config, scope, parameter) {

    this.application_config = application_config ;

    this.scope     = scope ;
    this.parameter = parameter ;

    this.cached_value = null ;

    this.set_cached_value = function (value) {
        this.cached_value = value ;
    } ;
    this.load = function (on_found, on_not_found) {
        if (this.cached_value !== null) {
            return this.cached_value ;
        }
        this.application_config.load(this.scope, this.parameter, on_found, on_not_found, this) ;
        return null ;
    } ;
    this.save = function (value) {
        this.cached_value = value ;
        this.application_config.save(this.scope, this.parameter, value) ;
    } ;
}

function FwkConfigCreator (application) {

    this.application = application ;

    this.cached_handlers = {} ;
    
    this.handler = function (scope, parameter) {
        if (!(scope     in this.cached_handlers))        this.cached_handlers[scope] = {} ;
        if (!(parameter in this.cached_handlers[scope])) this.cached_handlers[scope][parameter] = new FwkConfigHandlerCreator(this, scope, parameter) ;
        return this.cached_handlers[scope][parameter] ;
    } ;

    this.load = function (scope, parameter, on_found, on_not_found, handler2update_on_found) {
        var url = '../portal/ws/config_load.php' ;
        var params = {
            application: application ,
            scope      : scope ,
            parameter  : parameter
        } ;
        var jqXHR = $.get(url, params, function(data) {
            var result = eval(data) ;
            if (result.status != 'success') { Fwk.report_error(result.message, null) ; return ; }
            if (result.found) {
                var value = eval('('+result.value.value+')') ;
                on_found(value) ;
                if (handler2update_on_found) handler2update_on_found.set_cached_value(value) ;
            } else {
                on_not_found() ;
            }
        } ,
        'JSON').error(function () {
            Fwk.report_error('configuration loading failed because of: '+jqXHR.statusText) ;
        }) ;
    } ;
    
    this.save = function (scope, parameter, value) {
        var url = '../portal/ws/config_save.php' ;
        var params = {
            application: application ,
            scope      : scope ,
            parameter  : parameter ,
            value      : $.toJSON(value)
        } ;
        var jqXHR = $.post(url, params, function(data) {
            var result = eval(data) ;
            if (result.status != 'success') { Fwk.report_error(result.message, null) ; return ; }
        } ,
        'JSON').error(function () {
            Fwk.report_error('configuration saving failed because of: '+jqXHR.statusText) ;
        }) ;
    } ;
}

/*
 * The base class for user-defined dispatchers.
 *
 * @returns {FwkDispatcherBase}
 */
function FwkDispatcherBase () {

    this.container = null ;
    this.active = false ;

    this.set_container = function (container) { if (!this.container) this.container = container ; }
    this.activate = function (container) {
        this.set_container(container) ;
        this.active = true ;
        this.on_activate() ;
    } ;
    this.deactivate = function (container) {
        this.set_container(container) ;
        this.active = false ;
        this.on_deactivate() ;
    } ;
    this.update = function (container) {
        this.set_container(container) ;
        this.on_update() ;
    } ;
}

/*
 * The template constructor for applications.
 * 
 * @param {String} name
 * @param {String} full_name
 * @returns {FwkApplicationTemplate}
 */
function FwkApplicationTemplate (name, full_name) {

    var that = this ;

    this.name = name ;
    this.full_name = full_name ;

    this.context1 = '' ;
    this.context2 = '' ;

    this.context = {} ;

    this.wa_id = null;
    this.dispatcher = null ;
    this.html = null ;
    this.load_html = null ;

    this.is_initialized = false ;

    this.select = function (ctx1, ctx2) {
        that.context1 = ctx1 ;
        that.context2 = ctx2 ? ctx2 : '' ;
    } ;
    this.context1_to_name = function () {
        if (this.context[this.context1] !== undefined)
            return this.context[this.context1].name ;
        return '' ;
    } ;
    this.name_to_context1 = function (name) {
        for (var context1 in this.context)
            if (this.context[context1].name === name)
                return context1 ;
        return undefined ;
    } ;
    this.context2_to_name = function () {
        if ((this.context[this.context1] !== undefined) &&
            (this.context[this.context1].context[this.context2] !== undefined))
            return this.context[this.context1].context[this.context2].name ;
        return '' ;
    } ;
    this.get_any_wa_id = function (context1, context2) {
        if (this.context[context1] === undefined) return this.wa_id ;
        if (this.context[context1].context[context2] === undefined) return this.context[context1].wa_id ;
        return this.context[context1].context[context2].wa_id ;
    } ;
    this.get_wa_id = function () {
        return this.get_any_wa_id (this.context1, this.context2) ;
    } ;
    this.get_dispatcher = function () {
        if ((this.context1 !== '') && (this.context2 !== '')  && (this.context[this.context1].context[this.context2].dispatcher))
            return this.context[this.context1].context[this.context2].dispatcher ;
        if ((this.context1 !== '') && this.context[this.context1].dispatcher)
            return this.context[this.context1].dispatcher ;
        if (this.dispatcher)
            return this.dispatcher ;
        return null ;
    } ;

    this.init = function () {
        if (this.is_initialized) return ;
        this.is_initialized = true ;

        if (this.html)
            $('#fwk-applications > #'+this.get_any_wa_id()+' > div').html(this.html) ;

        if (this.load_html)
            Fwk.loader_GET (
                this.load_html.url ,
                this.load_html.params ,
                $('#fwk-applications > #'+this.get_any_wa_id()+' > div')) ;

        if (this.context)
            for (var i in this.context) {
                var context1_name = ''+i ;
                var context1 = this.context[context1_name] ;

                if (context1.html)
                    $('#fwk-applications > #'+this.get_any_wa_id(context1_name)+' > div').html(context1.html) ;

                if (context1.load_html)
                    Fwk.loader_GET (
                        context1.load_html.url ,
                        context1.load_html.params ,
                        $('#fwk-applications > #'+this.get_any_wa_id(context1_name)+' > div')) ;

                if (context1.context)
                    for (var j in context1.context) {
                        var context2_name = ''+j ;
                        var context2 = context1.context[context2_name] ;

                        if (context2.html)
                            $('#fwk-applications > #'+this.get_any_wa_id(context1_name, context2_name)+' > div').html(context2.html) ;

                        if (context2.load_html)
                            Fwk.loader_GET (
                                context2.load_html.url ,
                                context2.load_html.params ,
                                $('#fwk-applications > #'+this.get_any_wa_id(context1_name, context2_name)+' > div')) ;
                    }
        }
    } ;
    this.activate = function () {
        this.init() ;
        var container = $('#fwk-applications > #'+this.get_wa_id()+' > div') ;
        var dispatcher = this.get_dispatcher() ;
        if (dispatcher) dispatcher.activate(container) ;
    } ;
    this.deactivate = function () {
        this.init() ;
        var container = $('#fwk-applications > #'+this.get_wa_id()+' > div') ;
        var dispatcher = this.get_dispatcher() ;
        if (dispatcher) dispatcher.deactivate(container) ;
    } ;
    
    /*
     * Ping all subscribers (for whatever reason)
     *
     * @returns {undefined}
     */
    this.update = function () {
        this.init() ;

        if (this.dispatcher)
            dispatcher.update($('#fwk-applications > #'+this.get_any_wa_id()+' > div')) ;

        if (this.context)
            for (var i in this.context) {
                var context1_name = ''+i ;
                var context1 = this.context[context1_name] ;

                if (context1.dispatcher)
                     context1.dispatcher.update($('#fwk-applications > #'+this.get_any_wa_id(context1_name)+' > div')) ;

                if (context1.context)
                    for (var j in context1.context) {
                        var context2_name = ''+j ;
                        var context2 = context1.context[context2_name] ;

                        if (context2.dispatcher)
                            context1.dispatcher.update($('#fwk-applications > #'+this.get_any_wa_id(context1_name, context2_name)+' > div')) ;
                    }
            }
    } ;
}

/**
 * Utility class for creating tab-based applications.
 *
 * Note that no actions or DOM modifications are taken upon a construction of
 * the framework object. The real stuff starts happening build() is called.
 */
function FwkCreator () {

    var that = this;

    this.title = null ;
    this.subtitle = null ;
    this.auth = null ;
    this.url = document.URL ;
    this.applications = {} ;
    this.select_app = null ;
    this.select_app_context1 = null ;
    this.on_activate = null ;
    this.on_deactivate = null ;
    this.on_quick_search = null ;
    this.on_update = null ;
    this.config_svc = null ;
    this.is_built = false ;
    this.prev_app = null ;
    this.current_application = null ;

    /*
     * The UI builder
     * 
     * @param {string} title
     * @param {string} subtitle
     * @param {object} ui_config
     * @param {function} on_quick_search
     * @returns {unresolved}
     */
    this.build = function (title, subtitle, ui_config, on_quick_search) {

        if (this.is_built) return ;

        // Process and store UI configuration parameters

        this.title = title ;
        this.subtitle = subtitle ;

        this.config_svc = new FwkConfigCreator(this.title+':'+this.subtitle) ;

        for (var i in ui_config) {
            var menu1 = ui_config[i] ;
            var application = new FwkApplicationTemplate(''+i, menu1.name) ;
            application.wa_id = null ;
            if (menu1.menu) {
                var first_menu2 = true ;
                for (var j in menu1.menu) {
                    var menu2 = menu1.menu[j] ;
                    if (first_menu2) application.context1 = ''+j ;
                    var descr2 = {
                        'name': ''+menu2.name ,
                         context: {} ,
                         wa_id: null
                    } ;
                    application.context[''+j] = descr2 ;
                    var first_menu3 = true ;
                    if (menu2.menu) {
                        for (var k in menu2.menu) {
                            var menu3 = menu2.menu[k] ;
                            if (first_menu2 && first_menu3) application.context2 = ''+k ;
                            var descr3 = {
                                'name': ''+menu3.name ,
                                context: {} ,
                                wa_id: null
                            } ;
                            application.context[''+j].context[''+k] = descr3 ;
                            if (menu3.dispatcher)     descr3.dispatcher = menu3.dispatcher ;
                            if (menu3.html_container) descr3.html       = $('#'+menu3.html_container).html() ;
                            if (menu3.html)           descr3.html       = menu3.html ;
                            if (menu3.load_html)      descr3.load_html  = {
                                url:    menu3.load_html.url ,
                                params: menu3.load_html.params ? menu3.load_html.params : {}} ;
                            first_menu3 = false ;
                        }
                    } else {
                        if (menu2.dispatcher)     descr2.dispatcher = menu2.dispatcher ;
                        if (menu2.html_container) descr2.html       = $('#'+menu2.html_container).html() ;
                        if (menu2.html)           descr2.html       = menu2.html ;
                        if (menu2.load_html)      descr2.load_html  = {
                            url:    menu2.load_html.url ,
                            params: menu2.load_html.params ? menu2.load_html.params : {}} ;
                    }
                    first_menu2 = false ;
                }
            } else {
                if (menu1.dispatcher)     application.dispatcher = menu1.dispatcher ;
                if (menu1.html_container) application.html       = $('#'+menu1.html_container).html() ;
                if (menu1.html)           application.html       = menu1.html ;
                if (menu1.load_html)      application.load_html  = {
                    url:    menu1.load_html.url ,
                    params: menu1.load_html.params ? menu1.load_html.params : {}} ;
            }
            this.applications[''+i] = application ;
        }
        if (on_quick_search) {            
            this.on_quick_search = on_quick_search ? on_quick_search : null ;
        }
        this.on_update = function () {            
            for (var i in this.applications) {
                var app = this.applications[i] ;
                app.update() ;
            }
        } ;

        // Delayed initialization after we get the authentication context
        // of the logged user.

        that.web_service_GET (
            '../authdb/ws/AuthenticationProfile.php' ,
            {} ,
            function (data) {

                // Finalize the UI builfing process

                that.auth = data.profile ;
                that.is_built = true ;

                // Build the UI and start the framework services

                that.init_html () ;
                that.start_vsplitter_manager() ;
                that.auth_timer_restart() ;
                that.init_tabs_menus () ;
                that.update_timer_restart() ;
            }
        ) ;
    } ;

    /*
     * The method will return an API object for retreiving and saving states
     * of configuration paremeters.
     *
     * @param {string} scope
     * @param {string} parameter
     * @returns {object}
     */
    this.config_handler = function (scope, parameter) {
        return this.config_svc ?
             this.config_svc.handler(scope, parameter) :
             null ;
    } ;

    /* --------------------------------
     *   VERTICAL SPLITTER MANAGEMENT
     * --------------------------------
     */
 
    this.mouse_down = false ;

    this.resize = function () {
        var    top_height = 132 ;
        var bottom_height = 0 ;
        var center_height = $(window).height() - top_height - bottom_height ;
        $('#fwk-left'    ).height(center_height + 2);
        $('#fwk-splitter').height(center_height);
        $('#fwk-center'  ).height(center_height);
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
        $('#fwk-left'    ).css('width',       pos['x']) ;
        $('#fwk-splitter').css('left',        pos['x']) ;
        $('#fwk-center'  ).css('margin-left', pos['x']+1) ;
    }

    this.start_vsplitter_manager = function() {

        $('body').attr('onresize', 'Fwk.resize()') ;

        this.resize() ;

        $('#fwk-splitter').mousedown (function(e) { that.mouse_down = true ; return false ; }) ;

        $('#fwk-left'    ).mousemove(function(e) { if (that.mouse_down) that.move_split(e) ; });
        $('#fwk-center'  ).mousemove(function(e) { if (that.mouse_down) that.move_split(e) ; });

        $('#fwk-left'    ).mouseup  (function(e) { that.mouse_down = false ; });
        $('#fwk-splitter').mouseup  (function(e) { that.mouse_down = false ; });
        $('#fwk-center'  ).mouseup  (function(e) { that.mouse_down = false ; });
    } ;

    /* -----------------------------
     *   AUTHENTICATION MANAGEMENT
     * -----------------------------
     */

    this.auth_timer = null ;

    this.auth_timer_restart = function () {
        if (this.auth.is_authenticated && (this.auth.type === 'WebAuth'))
            this.auth_timer = window.setTimeout('Fwk.auth_timer_event()', 1000 ) ;
    } ;

    this.auth_last_secs = null ;

    this.auth_timer_event = function () {
        var now_sec = this.now().sec ;
        var seconds = this.auth.webauth_token_expiration - now_sec ;
        if (seconds <= 0) {
            $('#fwk-popupdialogs').html (
                '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>' +
                'Your WebAuth session has expired. Press <b>Ok</b> or use <b>Refresh</b> button' +
                'of the browser to renew your credentials.</p>'
            ) ;
            $('#fwk-popupdialogs').dialog ({
                resizable: false ,
                modal: true ,
                buttons: {
                    'Ok': function () {
                        $(this).dialog('close') ;
                        that.refresh_page() ;
                    }
                } ,
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

        $('#auth_expiration_info').html (
            '<b>'+hours_left_str+':'+minutes_left_str+'.'+seconds_left_str+'</b>'
        ) ;

        this.auth_timer_restart() ;
    } ;

    this.logout = function () {
        $('#fwk-popupdialogs').html (
            '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
            'This will log yout out from the current WebAuth session. Are you sure?</p>'
        ) ;
        $('#fwk-popupdialogs').dialog ({
            resizable: false ,
            modal: true ,
            buttons: {
                "Yes": function () {
                    $(this).dialog('close') ;
                    document.cookie = 'webauth_wpt_krb5=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/' ;
                    document.cookie = 'webauth_at=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/' ;
                    that.refresh_page() ;
                } ,
                Cancel: function () {
                    $(this).dialog('close') ;
                }
            } ,
            title: 'Session Logout Warning'
        }) ;
    } ;

    this.refresh_page = function () {
        window.location = this.url ;
    } ;

    /* ---------------------
     *   UTILITY FUNCTIONS
     * ---------------------
     */
    this.zeroPad = function (num, base) {
        var len = base - String(num).length + 1 ;
        return (len ? new Array(len).join('0') : '') + num ;
    } ;

    this.now = function () {
        var date = new Date() ;
        var msec = date.getTime() ;
        return { date: date, sec: Math.floor(msec/1000), msec: msec % 1000 } ; 
    }

    this.printer_friendly = function () {
        if (this.current_application) {
            $('#fwk-applications > #'+this.current_application.get_wa_id()).printElement ({
                leaveOpen: true ,
                printMode: 'popup' ,
                printBodyOptions: {
                    styleToAdd: 'font-size:10px ;'
                }
            }) ;
        }
    } ;

    this.show_email = function (user, addr) {
        var container = $('#fwk-popupdialogs') ;
        container.html('<p>'+addr+'</p>') ;
        container.dialog ({
            modal:  true ,
            title:  'e-mail: '+user
        }) ;
    } ;
    this.ask_yes_no = function (title, msg, on_yes, on_cancel) {
        var container = $('#fwk-popupdialogs') ;
        container.html (
            '<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+msg+'</p>'
        ) ;
        container.dialog ({
            resizable: false ,
            modal: true ,
            buttons: {
                "Yes": function() {
                    $( this ).dialog('close') ;
                    if(on_yes) on_yes() ;
                } ,
                Cancel: function() {
                    $(this).dialog('close') ;
                    if(on_cancel) on_cancel() ;
                }
            } ,
            title: title
        }) ;
    } ;
    this.ask_for_input = function (title, msg, on_ok, on_cancel) {
        var container = $('#fwk-popupdialogs-varable-size') ;
        container.html (
'<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+msg+'</p>'+
'<div><textarea rows=4 cols=60></textarea/>'
        ) ;
        container.dialog ({
            resizable: true ,
            modal: true ,
            width:  470 ,
            height: 300 ,
            buttons: {
                "Ok": function () {
                    var user_input = container.find('textarea').val() ;
                    $(this).dialog('close') ;
                    if (on_ok) on_ok(user_input) ;
                },
                Cancel: function () {
                    $(this).dialog('close') ;
                    if (on_cancel) on_cancel() ;
                }
            } ,
            title: title
        }) ;
    } ;
    this.edit_dialog = function (title, msg, on_save, on_cancel) {
        var container = $('#fwk-editdialogs') ;
        container.html(msg) ;
        container.dialog ({
            resizable: true ,
            width: 640 ,
            modal: true ,
            buttons: {
                Save: function () {
                    $(this).dialog('close') ;
                    if (on_save != null) on_save() ;
                } ,
                Cancel: function() {
                    $(this).dialog('close') ;
                    if (on_cancel != null) on_cancel() ;
                }
            } ,
            title: title
        }) ;
    } ;
    this.report_error = function (msg) {
        var container = $('#fwk-popupdialogs') ;
        container.html('<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+msg+'</p>') ;
        container.dialog ({
            resizable: true ,
            modal: true ,
            buttons: {
                'Ok': function() { $(this).dialog('close') ; }
            } ,
            title: 'Error'
        }) ;
    } ;
    this.report_info = function (title, msg) {
        var container = $('#fwk-infodialogs') ;
        container.html(msg) ;
        container.dialog({
            resizable: true ,
            modal: true ,
            title: title
        }) ;
    } ;
    this.report_info_table = function (title, hdr, rows) {
        var container = $('#fwk-infodialogs') ;
        var table = new Table(container, hdr, rows) ;
        table.display() ;
        container.dialog ({
            width: 720 ,
            height: 800 ,
            resizable: true ,
            modal: true ,
            title: title
        }) ;
    } ;

    /* -----------------------------------------------
     *   SIMPLE API TO WEB SERVICES AND HTML LOADERS
     * -----------------------------------------------
     */
    this.web_service_GET = function (url, params, on_success, on_failure) {
        var jqXHR = $.get(url, params, function (data) {
            if (data.status != 'success') {
                if (on_failure) on_failure(data.message) ;
                else            Fwk.report_error(data.message, null) ;
                return ;
            }
            if (on_success) on_success(data) ;
        },
        'JSON').error(function () {
            var message = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
            if (on_failure) on_failure(message) ;
            else            Fwk.report_error(message, null) ;
        }) ;
    } ;
    this.web_service_POST = function (url, params, on_success, on_failure) {
        var jqXHR = $.post(url, params, function (data) {
            if (data.status != 'success') {
                if (on_failure) on_failure(data.message) ;
                else            Fwk.report_error(data.message, null) ;
                return ;
            }
            if (on_success) on_success(data) ;
        },
        'JSON').error(function () {
            var message = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
            if (on_failure) on_failure(message) ;
            else            Fwk.report_error(message, null) ;
        }) ;
    } ;
    this.loader_GET = function (url, params, container, on_failure) {
        var jqXHR = $.get(url, params, function (data) {
            container.html(data) ;
        },
        'HTML').error(function () {
            var message = 'Document loading request to '+url+' failed because of: '+jqXHR.statusText ;
            if (on_failure) on_failure(message) ;
            else            Fwk.report_error(message, null) ;
        }) ;
    } ;

    /* -------------------------------------------
     *   OPERATIONS WITH TABS AND VERTICAL MENUS
     * -------------------------------------------
     */
    this.set_context = function (app) {
        if (this.prev_app && (this.prev_app != app)) this.prev_app.deactivate() ;
        app.activate() ;
        this.prev_app = app ;
    } ;

    this.v_item_group = function (item) {
        var item = $(item) ;
        var parent = item.parent() ;
        if (parent.hasClass('fwk-menu-group-members')) return parent.prev() ;
        return null ;
    } ;

    /* Event handler for application selections from the top-level menu bar
     * to fill current application context.
     */
    this.m_item_selected = function (item) {

        var item = $(item) ;

        if(item.hasClass('fwk-tab-select')) return ;

        var previous_application = this.current_application ;
        this.current_application = this.applications[item.attr('id')] ;

        $('#fwk-tabs > .fwk-tab-select').removeClass('fwk-tab-select') ;
        item.addClass('fwk-tab-select') ;

        if (previous_application) $('#fwk-menu > #'+previous_application.name).removeClass('fwk-visible').addClass('fwk-hidden') ;
        $('#fwk-menu > #'+this.current_application.name).removeClass('fwk-hidden').addClass('fwk-visible') ;

        if (this.current_application.context2)
            this.v_item_selected($('#fwk-menu > #'+this.current_application.name+' > #'+this.current_application.context1).next().children('.fwk-menu-item#'+this.current_application.context2)) ;
        else
            this.v_item_selected($('#fwk-menu > #'+this.current_application.name).children('.fwk-menu-item#'+this.current_application.context1)) ;

        this.set_context(this.current_application) ;
    } ;

    /* Event handler for vertical menu group selections will only
     * show/hide children (if any).
     */
    this.v_group_selected = function (group) {
        var group = $(group) ;
        var toggler = group.children('.ui-icon') ;
        if (toggler.hasClass('ui-icon-triangle-1-s')) {
            toggler.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
            group.next().removeClass('fwk-menu-group-members-visible').addClass('fwk-menu-group-members-hidden') ;
        } else {
            toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
            group.next().removeClass('fwk-menu-group-members-hidden').addClass('fwk-menu-group-members-visible') ;
        }
    } ;

    /* Event handler for vertical menu item (actual commands) selections:
     * - dim the poreviously active item (and if applies - its group)
     * - hightlight the new item (and if applies - its group)
     * - change the current context
     * - execute the commands
     * - switch the work area (make the old one invisible, and the new one visible)
     */
    this.v_item_selected = function (item) {

        var item = $(item) ;

        $('#fwk-menu > #'+this.current_application.name).find('.fwk-menu-item.fwk-menu-select').each(function () {
            $(this).children('.ui-icon').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
            $(this).removeClass('fwk-menu-select') ;
            var this_group = that.v_item_group(this) ;
            if (this_group != null) this_group.removeClass('fwk-menu-select') ;
        });

        item.children('.ui-icon').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
        item.addClass('fwk-menu-select') ;

        var group = this.v_item_group(item) ;
        if (group) {

            /* Force the group to unwrap
             *
             * NOTE: This migth be needed of the current method is called out of
             *       normal sequence.
             *
             * TODO: Do it "right" when refactoring the menu classes.
             */
            var toggler = $(group).children('.ui-icon') ;
            if (!toggler.hasClass('ui-icon-triangle-1-s')) {
                toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
                $(group).next().removeClass('fwk-menu-group-members-hidden').addClass('fwk-menu-group-members-visible') ;
            }
            group.addClass('fwk-menu-select') ;

            this.current_application.select(group.attr('id'), item.attr('id')) ;

        } else {

            this.current_application.select(item.attr('id')) ;
        }
        this.set_context(this.current_application) ;

        // Hide the older work area and display the new one

        $('#fwk-applications > .fwk-appl-wa.fwk-visible').removeClass('fwk-visible').addClass('fwk-hidden') ;
        $('#fwk-applications > #'+this.current_application.get_wa_id()).addClass('fwk-visible') ;
    } ;

    this.init_tabs_menus = function () {

	$('.fwk-tab' ).click(function() { that.m_item_selected (this) ; }) ;
	$('.fwk-menu-group').click(function() { that.v_group_selected(this) ; }) ;
	$('.fwk-menu-item' ).click(function() { that.v_item_selected (this) ; }) ;

        if (this.on_quick_search) {
            $('#fwk-search-text').keyup(function (e) {
                var text2search = $('#fwk-search-text').val() ;
                if ((text2search != '') && (e.keyCode == 13)) {
                    that.on_quick_search(text2search) ;
                }
            }) ;
        }

	// Finally, activate the selected application (if any provided). Otherwise
        // pick the first one.
	//
	for (var id in this.applications) {
            var application = this.applications[id] ;
            if (this.select_app) {
                if (application.name === this.select_app) {
                    $('#fwk-tabs').children('#'+application.name).each(function() { that.m_item_selected(this) ; }) ;
                    if (this.select_app_context1) {
                        this.v_item_selected($('#fwk-menu > #'+application.name+' > #'+this.select_app_context1)) ;
                        application.select(this.select_app_context1) ;
                    }
                    break ;
                }
            } else {
                this.m_item_selected($('#fwk-tabs > #'+application.name)) ;
                break ;
            }
	}
    } ;

    /**
     * (Forced) Activate the specified application similar to clicking
     * UI active elements of the corresponding tab and vertical menu
     * item(S).
     * 
     * @param {string} application_name - the tab name
     * @param {string} context1_name - the vertical menu name from teh left
     * @returns {undefined
     */
    this.activate = function (application_name, context1_name) {
	for (var i in this.applications) {
            var application = this.applications[i] ;
            if (application.full_name === application_name) {
                $('#fwk-tabs').children('#'+application.name).each(function() { that.m_item_selected(this) ; }) ;
                if (context1_name) {
                    this.v_item_selected($('#fwk-menu > #'+application.name+' > #'+application.name_to_context1(context1_name))) ;
                    application.select(application.name_to_context1(context1_name)) ;
                }
                return application.get_dispatcher() ;
            }
	}
    } ;

    /* ------------------------------------------------------------
     *   POPULATE HTML ACCORDING TO THE APPLICATION CONFIGURATION
     * ------------------------------------------------------------
     */
    function num_keys_in_dict (obj) {
        var result = 0 ;
        for (var key in obj) result++ ;
        return result ;
    }

    this.init_html = function () {
        var html =
'<div id="fwk-top">' +
'  <div id="fwk-top-header">' +
'    <div id="fwk-top-title">' +
'      <div style="float:left; padding-left:15px; padding-top:10px;">' +
'        <span id="fwk-title">'+this.title+' : </span>' +
'        <span id="fwk-subtitle">'+this.subtitle+'</span>' +
'      </div>' +
'      <div style="float:right;" id="fwk-login" class="not4print">' +
'        <table><tbody>' +
'          <tr>' +
'            <td rowspan="4" valign="bottom"><a href="javascript:Fwk.printer_friendly()" title="Printer friendly version of this page"><img src="../portal/img/PRINTER_icon.gif" style="border-radius: 5px;" /></a></td>' +
'          </tr>' +
'          <tr>' +
'            <td>&nbsp;</td>' +
'            <td>[<a href="javascript:Fwk.logout()" title="close the current WebAuth session">logout</a>]</td>' +
'          </tr>' +
'          <tr>' +
'            <td>Logged as:&nbsp;</td>' +
'            <td><b>'+this.auth.user+'</b></td>' +
'          </tr>' +
'          <tr>' +
'            <td>Session expires in:&nbsp;</td>' +
'            <td><span id="auth_expiration_info"><b>00:00.00</b></span></td>' +
'          </tr>' +
'        </tbody></table>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'    <div id="fwk-tabs">' ;

        var first_application = true ;
        for (var i in this.applications) {
            var application = this.applications[i] ;
            html +=
'      <div class="fwk-tab '+(first_application ? 'fwk-tab-first' : 'fwk-tab-next')+'" id="'+application.name+'">'+application.full_name+'</div>' ;
            first_application = false ;
        }
        html +=
'      <div class="fwk-tab-end"></div>' +
'    </div>' +
'    <div id="fwk-context-header">' +
'      <div id="fwk-context" style="float:left"></div>' +
'      <div id="fwk-search" style="float:right">' ;
        if (this.on_quick_search) {
            html +=
'        <b>quick search:</b> <input type="text" id="fwk-search-text" value="" size=16 title="enter text to search in the application, then press ENTER to proceed" />' ;
        }
        html +=
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'  </div>' +
'</div>' +
'' +
'<div id="fwk-left">' +
'' +
'<div id="fwk-menu">' +
'' +
'    <div id="fwk-menu-title"></div>' +
'' ;
        first_application = true ;
        for (var i in this.applications) {
            var application = this.applications[i] ;
            html +=
'    <div id="'+application.name+'" class="'+(first_application ? 'fwk-visible' : 'fwk-hidden')+'">' ;
            for (var j in application.context) {
                var context1 = application.context[j] ;
                var context1_empty = !num_keys_in_dict(context1.context) ;
                html += context1_empty ?
'      <div class="fwk-menu-item" id="'+j+'">' :
'      <div class="fwk-menu-group" id="'+j+'">' ;
                html +=
'        <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>' +
'        <div style="float:left;" >'+context1.name+'</div>' +
'        <div style="clear:both;"></div>' +
'      </div>' ;
                if (!context1_empty) {
                    html +=
'      <div class="fwk-menu-group-members fwk-menu-group-members-hidden">' ;
                    for (var k in context1.context) {
                        var context2 = context1.context[k] ;
                        html +=
'        <div class="fwk-menu-item" id="'+k+'">' +
'          <div class="ui-icon ui-icon-triangle-1-e" style="float:left;"></div>' +
'          <div class="link" style="float:left;" >'+context2.name+'</div>' +
'          <div style="clear:both;"></div>' +
'        </div>' ;
                    }
                    html +=
'      </div>' ;
                }
            }
            html +=
'    </div>' ;
            first_application = false ;
        }

        html +=
'' +
'  </div>' +
'</div>' +
'' +
'<div id="fwk-splitter"></div>' +
'' +
'<div id="fwk-center">' +
'  <div id="fwk-applications">' ;
        for (var i in this.applications) {
            var application = this.applications[i] ;
            var application_empty = true ;
            for (var j in application.context) {
                application_empty = false ;
                var context1 = application.context[j] ;
                var context1_empty = true ;
                if (context1.context) {
                    for (var k in context1.context) {
                        context1_empty = false ;
                        var context2 = context1.context[k] ;
                        var wa_id = application.name + '-' + j + '-' + k ;
                        context2.wa_id = wa_id ;
                        html += this.init_wa_html(wa_id) ;
                    }
                }
                if (context1_empty) {
                    var wa_id = application.name + '-' + j ;
                    context1.wa_id = wa_id ;
                    html += this.init_wa_html(wa_id) ;
                }
            }
            if (application_empty) {
                var wa_id = application.name ;
                application.wa_id = wa_id ;
                html += this.init_wa_html(wa_id) ;
            }
        }
        html +=
'  </div>' +
'' +
'  <div id="fwk-popupdialogs" style="display:none;"></div>' +
'  <div id="fwk-popupdialogs-varable-size" style="display:none;"></div>' +
'  <div id="fwk-infodialogs" style="display:none;"></div>' +
'  <div id="fwk-editdialogs" style="display:none;"></div>' +
'</div>';
        $('body').html(html);
    } ;

    /*
     * Return an HTML code for the workarea container.
     *
     * @param {String} id
     * @returns {String}
     */
    this.init_wa_html = function (id) {
        var html =
'    <div id="'+id+'" class="fwk-appl-wa fwk-hidden">' +
'      <div class="fwk-appl-wa-cont"></div>' +
'    </div>' ;
        return html ;
    } ;

    /* -----------------------------
     *   APPLICATIONS UPDATE TIMER
     * -----------------------------
     */
    this.update_timer = null ;

    this.update_timer_restart = function () {
        if (this.on_update) this.update_timer = window.setTimeout('Fwk.update_timer_event()', 1000 ) ;
    } ;

    /*
     * Process an event. Do not notify subscribers at the first invocation.
     *
     * @returns {undefined}
     */
    this.update_timer_event = function () {
        this.on_update() ;
        this.update_timer_restart() ;
    } ;
}

/* ATTENTION: This will only create an instance of the framework. No actions
 * or DOM modifications will be taken until it's finally built and activated.
 */
var Fwk = new FwkCreator() ;
