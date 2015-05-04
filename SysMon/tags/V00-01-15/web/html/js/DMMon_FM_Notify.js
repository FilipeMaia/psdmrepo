define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../sysmon/css/DMMon_FM_Notify.css') ;

    /**
     * The application for managing e-mail notifications for delayed migrations
     *
     * @returns {DMMon_History}
     */
    function DMMon_FM_Notify (app_config) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this.on_update() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 60 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._app_config = app_config ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;
        this._is_loading = false ;

        var _MIN_ALLOWED_DELAY = 10 ;

        var _DELAYS_DEF = [
            {sec:    2*3600, name:  '2 hours'} ,
            {sec:    4*3600, name:  '4 hours'} ,
            {sec:    8*3600, name:  '8 hours'} ,
            {sec:   12*3600, name: '12 hours'} ,
            {sec:   18*3600, name: '18 hours'} ,
            {sec:   24*3600, name:      'day'} ,
            {sec: 2*24*3600, name:   '2 days'} ,
            {sec: 7*24*3600, name:     'week'}
        ] ;
        this._wa = function (html) {
            if (!this._wa_elem) {
                var title_subscription =
                    'Check to subscribe a search for specified search criteria.' ;
                var title_instr =
                    'Narrow a search down to the specified instrument. \n' +
                    'Otherwise all instruments will be assumed.' ;
                var title_last =
                    'Specify how far to look back into the history of \n' +
                    'past transfers.' ;
                var title_delay =
                    'Specify a threshold for delayed transfers to be ignored. \n' +
                    'Note that the minimum value of this parameter is set to '+_MIN_ALLOWED_DELAY+' seconds. \n' +
                    'This is done to prevent an excess load onto the Data Management system.' ;
                var html = html ||
'<div id="dmmon-fm-notify" >' +

  '<div class="info" id="updated" style="float:right;" ></div> ' +
  '<div style="clear:both;" ></div> ' +

  '<div id="notes" style="float:left;" > ' +
    '<h2>Instructions:</h2>' +
    '<p>This application allows to subscribe (or update an existing subscription) ' +
       'for automatic e-mail notifications when certain delays occur in migrating ' +
       'experimental files through various stages of the LCLS Data ' +
       'Management system. The present status of your subscription' +
       'is sown below. You may change parameters of the subscription ' +
       'or subscribe/unsubscribe at any time. You shoudl receive a confirmation e-mail ' +
       'at your address <b>'+this._app_config.uid+'(@slac.stanford.edu</b> shortly' +
       'after the change.' +
    '</p>' +
    '<p>A table at the bottom of the page also presents a list of other ussers ' +
       'who are regestered to receive similar notifications. Note that each subscriber ' +
       'is free to choose their own criteria.' +
    '</p>' +
  '</div> ' +
  '<div id="buttons" style="float:left;" > ' +
    '<button class="control-button" name="update" title="click to update the information from the database" ><img src="../webfwk/img/Update.png" /></button> ' +
  '</div> ' +
  '<div style="clear:both;" ></div> ' +

  '<h2>Manage my subscription:</h2>' +
  
  '<div id="ctrl" style="float:left;" > ' +

    '<div class="control-group" data="'+title_subscription+'"> ' +
      '<div class="control-group-title" >Subscribed</div> ' +
      '<div class="control-group-selector" > ' +
        '<input type="checkbox" name="subscription" / > ' +
      '</div> ' +
    '</div> ' +

    '<div class="control-group" data="'+title_instr+'" > ' +
      '<div class="control-group-title" >Instr.</div> ' +
      '<div class="control-group-selector" > ' +
        '<select name="instr" > ' +
          '<option value="" ></option> ' + _.reduce(this._app_config.instruments, function (html, instr) { return html +=
          '<option value="'+instr+'" >'+instr+'</option> ' ; }, '') +
        '</select> ' +
      '</div> ' +
    '</div> ' +

    '<div class="control-group" data="'+title_last+'" > ' +
      '<div class="control-group-title" >Search last</div> ' +
      '<div class="control-group-selector" > ' +
        '<select name="last" > ' + _.reduce(_DELAYS_DEF, function (html, d) { return html +=
          '<option value="'+d.sec+'" >'+d.name+'</option> ' ; }, '') +
        '</select> ' +
      '</div> ' +
    '</div> ' +
  
    '<div class="control-group" data="'+title_delay+'" > ' +
      '<div class="control-group-title" >Delayed by [sec]</div> ' +
      '<div class="control-group-selector" > ' +
        '<input type="text" size="1" name="delay" value="0" / > ' +
      '</div> ' +
    '</div> ' +

    '<div class="control-group control-group-buttons" > ' +
      '<button class="control-button" name="reset"  title="click to reset the to recommended state" >SET TO RECOMMENDED</button> ' +
    '</div> ' +
    
    '<div class="control-group-end" ></div> ' +
    
  '</div>' +
  '<div style="clear:both;" ></div> ' +

  '<h2>All subscriptions:</h2>' +
  '<div id="users" >' +
  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dmmon-fm-notify') ;
            }
            return this._wa_elem ;
        } ;
        this._subscription_selector = function () {
            if (!this._subscription_selector_elem) {
                this._subscription_selector_elem = this._wa().find('div.control-group-selector').children('input[name="subscription"]') ;
            }
            return this._subscription_selector_elem ;
        } ;
        this._instr_selector = function () {
            if (!this._instr_selector_elem) {
                this._instr_selector_elem = this._wa().find('div.control-group-selector').children('select[name="instr"]') ;
            }
            return this._instr_selector_elem ;
        } ;
        this._last_selector = function () {
            if (!this._last_selector_elem) {
                this._last_selector_elem = this._wa().find('div.control-group-selector').children('select[name="last"]') ;
            }
            return this._last_selector_elem ;
        } ;
        this._delay_selector = function () {
            if (!this._delay_selector_elem) {
                this._delay_selector_elem = this._wa().find('div.control-group-selector').children('input[name="delay"]') ;
            }
            return this._delay_selector_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;

        this._button_reset = function () {
            if (!this._button_reset_elem) {
                this._button_reset_elem = this._wa().find('.control-button[name="reset"]').button() ;
            }
            return this._button_reset_elem ;
        } ;
        this._button_load = function () {
            if (!this._button_load_elem) {
                this._button_load_elem = this._wa().find('.control-button[name="update"]').button() ;
            }
            return this._button_load_elem ;
        } ;
        this._table = function () {

            if (!this._table_obj) {
                var rows = [] ;
                var hdr = [
                    {   name: 'Name', align: 'right'} ,
                    {   name: 'Instrument'} ,
                    {   name: 'Search Last', align: 'right' ,
                        type: {
                            to_string: function (a)   {
                                for(var i in _DELAYS_DEF) {
                                    var d = _DELAYS_DEF[i] ;
                                    if (d.sec == a) return d.name ;
                                }
                                return a + ' s' ;
                            } ,
                            compare_values: function (a,b) { return a - b ; }
                        }
                    } ,
                    {   name: 'Delay [s]'} ,
                    {   name: 'Subscribed' ,
                        type: {
                            to_string:      function (a)   { return a.day  + '&nbsp;&nbsp;' + a.hms+ '</span>' ; } ,
                            compare_values: function (a,b) { return a.sec - b.sec ; }
                        }
                    } ,
                    {   name: 'By'} ,
                    {   name: 'From'}
                ] ;
                this._table_obj = new SimpleTable.constructor (
                    this._wa().find('#users') ,
                    hdr ,
                    rows) ;
                this._table_obj.display() ;
            }
            return this._table_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._subscription_selector().change(function () { _that._save() ; }) ;
            this._instr_selector()       .change(function () { _that._save() ; }) ;
            this._last_selector()        .change(function () { _that._save() ; }) ;
            this._delay_selector()       .change(function () {
                // Always make sure it's a positive numeric value.
                // Otherwise force it to be the one before reloading the table.
                var obj = $(this) ;
                var delay = parseInt(obj.val()) ;
                obj.val(delay >= _MIN_ALLOWED_DELAY ? delay : _MIN_ALLOWED_DELAY) ;
                _that._save() ;
            }) ;
            this._button_reset().click(function () { _that._reset() ; _that._save() ; }) ;
            this._button_load() .click(function () { _that._load() ;  }) ;

            this._table() ;     // just to display an empty table
            this._load() ;
        } ;
        this._reset = function () {
            this._instr_selector().val(0) ;
            this._last_selector() .val(2*3600) ;
            this._delay_selector().val(  3600) ;
        } ;
        this._load = function () {
            this._action (
                '../sysmon/ws/dmmon_fm_notify_get.php' ,
                {}
            ) ;
        } ;
        this._save = function () {
            var params = {
                uid: this._app_config.uid ,
                is_subscribed: this._subscription_selector().attr('checked') ? 1 : 0
            } ;
            if (params.is_subscribed) {
                params.instr     = this._instr_selector().val() ;
                params.last_sec  = this._last_selector() .val() ;
                params.delay_sec = this._delay_selector().val() ;
            }
            this._action (
                '../sysmon/ws/dmmon_fm_notify_save.php' ,
                params
            ) ;
        } ;
        this._action = function (url, params) {

            if (this._is_loading) return ;
            this._is_loading = true ;

            Fwk.web_service_GET (
                url ,
                params ,
                function (data) {
                    _that._users = data.users ;
                    _that._display() ;
                    _that._set_updated('Last updated: <b>'+data.updated+'</b>') ;
                    _that._button_reset().button('enable') ;
                    _that._button_load() .button('enable') ;
                    _that._is_loading = false ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._button_reset().button('enable') ;
                    _that._button_load().button('enable') ;
                    _that._is_loading = false ;
                }
            ) ;
        } ;
        this._display = function () {
            var my_subscription = null ;    // An object describing the current user's
                                            // subscription (if any)
            var rows = [] ;
            for (var i in this._users) {
                var user = this._users[i] ;
                if (user.uid === this._app_config.uid) {
                    my_subscription = user ;
                }
                rows.push([
                    user.gecos ,
                    user.instr ,
                    user.last_sec ,
                    user.delay_sec ,
                    user.subscribed_time ,
                    user.subscribed_gecos ,
                    user.subscribed_host
                ]) ;
            }
            this._display_my_subscription(my_subscription) ;
            this._table().load(rows) ;
        } ;
        this._display_my_subscription = function (user) {
            if (user) {
                this._subscription_selector().attr('checked', 'checked') ;
                this._instr_selector().val(user.instr) ;
                this._last_selector() .val(user.last_sec) ;
                this._delay_selector().val(user.delay_sec) ;
                this._instr_selector().removeAttr('disabled') ;
                this._last_selector() .removeAttr('disabled') ;
                this._delay_selector().removeAttr('disabled') ;
            } else {
                this._subscription_selector().removeAttr('checked') ;
                this._reset() ;
                this._instr_selector().attr('disabled', 'disabled') ;
                this._last_selector() .attr('disabled', 'disabled') ;
                this._delay_selector().attr('disabled', 'disabled') ;
            }
        } ;
    }
    Class.define_class (DMMon_FM_Notify, FwkApplication, {}, {}) ;
    
    return DMMon_FM_Notify ;
}) ;

