define ([    
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/CheckTable', 'webfwk/PropList', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, CheckTable, PropList, FwkApplication, Fwk) {

    cssloader.load('../portal/css/ExpSwitch_Station.css') ;

    /**
     * @brief The application for activating/deactivating experiments
     *
     * @returns {ExpSwitch_Station}
     */
    function ExpSwitch_Station (instrument, station, access_list) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this._init() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.instrument  = instrument ;
        this.station     = station ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this._wa = null ;    // work area container

        this._is_initialized = false ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._activate_button(this.access_list.can_manage).click(function () { _that._activate       () ; }) ;
            this._save_button    (false)                      .click(function () { _that._activate_save  () ; }) ;
            this._cancel_button  (false)                      .click(function () { _that._activate_cancel() ; }) ;
            this._refresh_button (true)                       .click(function () { _that._load           () ; }) ;

            this._current_proplist().load('loading...') ;
            this._load() ;
        } ;
        this._wa = function (html) {
            if (this._wa_elem) {
                if (html !== undefined) {
                    this._wa_elem.html(html) ;
                }
            } else {
                this.container.html('<div id="expswitch-station"></div>') ;
                this._wa_elem = this.container.find('div#expswitch-station') ;
                if (html === undefined) {
                    html =
'<div id="ctrl">' +
'  <button class="control-button"' +
'          name="activate"' +
'          style="color:red;"' +
'          title="activate another experiment" >ACTIVATE ANOTHER EXPERIMENT</button>' +
'  <button class="control-button"' +
'          name="save"' +
'          title="save modifications" >SAVE</button>' +
'  <button class="control-button"' +
'          name="cancel"' +
'          title="cancel modification and roll back to the previous state" >CANCEL</button>' +
'  <button class="control-button"' +
'          name="refresh"' +
'          title="refresh the status of teh current experiment" >REFRESH</button>' +
'</div>' +
'<div id="body">' +
'  <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div id="current" class="expswitch-enabled"  ></div>' +
'  <div id="next"    class="expswitch-disabled" ></div>' +
'</div>' ;
                }
                this._wa_elem.html(html) ;
            }
            return this._wa_elem ;
        } ;

        this._ctrl = function () {
            if (!this._ctrl_elem) {
                this._ctrl_elem = this._wa().children('#ctrl') ;
            }
            return this._ctrl_elem ;
        } ;
        this._ctrl_button = function (name, enable) {
            if (!this._ctrl_buttons_elem) this._ctrl_buttons_elem = {} ;
            if (!this._ctrl_buttons_elem[name]) {
                this._ctrl_buttons_elem[name] = this._ctrl().children('button[name="'+name+'"]').button() ;
            }
            if (enable !== undefined) {
                this._ctrl_buttons_elem[name].button(enable ? 'enable' : 'disable') ;
            }
            return this._ctrl_buttons_elem[name] ;
        } ;
        this._activate_button = function (enable) {  return this._ctrl_button('activate', enable) ; } ;
        this._save_button     = function (enable) {  return this._ctrl_button('save',     enable) ; } ;
        this._cancel_button   = function (enable) {  return this._ctrl_button('cancel',   enable) ; } ;
        this._refresh_button  = function (enable) {  return this._ctrl_button('refresh',  enable) ; } ;

        this._body = function () {
            if (!this._body_elem) this._body_elem = this._wa().children('#body') ;
            return this._body_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._body().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._current = function (enable) {
            if (!this._current_elem) this._current_elem = this._body().children('#current') ;
            if (enable !== undefined) {
                if (enable) this._current_elem.removeClass('expswitch-disabled').addClass('expswitch-enabled') ;
                else        this._current_elem.removeClass('expswitch-enabled') .addClass('expswitch-disabled') ;
            }
            return this._current_elem ;
        } ;
        this._current_proplist = function () {
            if (!this._current_proplist_obj) {
                this._current_proplist_obj = new PropList ([
                    {name: 'name',             text: 'Name',            type: 'html'} ,
                    {name: 'id',               text: 'Id'} ,
                    {name: 'first_run',        text: 'First run',       type: 'html'} ,
                    {name: 'last_run',         text: 'Last Run' ,       type: 'html'} ,
                    {name: 'descr',            text: 'Description'} ,
                    {name: 'contact',          text: 'Contact person',  type: 'html'} ,
                    {name: 'leader',           text: 'UNIX account of the PI'} ,
                    {name: 'posix_group',      text: 'POSIX group'} ,
                    {name: 'switch_time',      text: 'Switch time'} ,
                    {name: 'switch_requestor', text: 'Switch made by'}]) ;
                this._current_proplist_obj.display(this._current()) ;
            }
            return this._current_proplist_obj ;
        } ;

        this._next = function (enable) {
            if (!this._next_elem) {
                this._next_elem = this._body().children('#next') ;
                var html =
'<div class="group" >' +
'  <span   class="label" >Select experiment:</span>' +
'  <select class="value" name="experiment" ></select>' +
'</div>' +
'<div class="group" >' +
'  <textarea name="message" rows="8" cols="76" ' +
'            title="additional info to be send to a list of recepients \nafter switching to the new experiment" ></textarea>' +
'</div>' +
'<div class="group" >' +
'  <button class="control-button" name="check"   >CHECK ALL</button>' +
'  <button class="control-button" name="uncheck" >UNCHECK ALL</button>' +
'  <div id="checktable" ></div>' +
'</div>' ;
                this._next_elem.html(html) ;
            }
            if (enable !== undefined) {
                if (enable) this._next_elem.removeClass('expswitch-disabled').addClass('expswitch-enabled') ;
                else        this._next_elem.removeClass('expswitch-enabled') .addClass('expswitch-disabled') ;
            }
            return this._next_elem ;
        } ;
        this._next_experiment_select = function () {
            if (!this._next_experiment_elem) {
                this._next_experiment_elem = this._next().find('select[name="experiment"]') ;
            }
            return this._next_experiment_elem ;
        } ;
        this._next_message = function () {
            if (!this._next_message_elem) {
                this._next_message_elem = this._next().find('textarea[name="message"]') ;
            }
            return this._next_message_elem ;
        } ;
        this._next_checktable = function () {
            if (!this._next_checktable_obj) {
                var coldef = [
                    {name: 'notify', text:   'Notify'} ,
                    {name: 'gecos',   text:   'User'} ,
                    {name: 'email',  text:   'E-mail address'} ,
                    {name: 'role',   text:   'Role'} ,
                    {name: 'rank',   hidden: true} ,
                    {name: 'uid',    hidden: true}
                ] ;
                this._next_checktable_obj = new CheckTable (coldef) ;
                this._next_checktable_obj.display(this._next().find('#checktable')) ;
                this._next().find('button').button().click(function () {
                    switch(this.name) {
                        case 'check'   : _that._next_checktable().check_all() ; break ;
                        case 'uncheck' : _that._next_checktable().uncheck_all() ; break ;
                    }
                }) ;
            }
            return this._next_checktable_obj ;
        } ;

        this._load = function () {

            this._activate_button(false) ;
            this._refresh_button (false) ;

            this._set_updated('Updating...') ;

            Fwk.web_service_GET (
                '../portal/ws/expswitch_station.php' ,
                {instr_name: this.instrument, station: this.station} ,
                function (data) {

                    _that._activate_button(_that.access_list.can_manage) ;
                    _that._refresh_button (true) ;

                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data.current) ;
                }   
            ) ;
        } ;
        this._display = function (data) {
            var portal_url = '<a href="index.php?exper_id='+data.id+'" target="_blank" class="link" title="go to the Web Portal of the experiment">'+data.name+'</a>' ;
            this._current_proplist().load({
                'name'             : portal_url ,
                'id'               : data.id ,
                'first_run'        : data.first_run.begin_time + ' (<b>run:</b> '+data.first_run.num+')' ,
                'last_run'         : data.last_run.begin_time  + ' (<b>run:</b> '+data.last_run.num +')' ,
                'descr'            : data.descr ,
                'contact'          : data.decorated_contact ,
                'leader'           : data.leader ,
                'posix_group'      : data.posix_group ,
                'switch_time'      : data.switch_time ,
                'switch_requestor' : data.switch_requestor
            }) ;
        } ;
        this._load_next_experiments = function () {

            this._next_message().val('') ;
            this._next_checktable().remove_all() ;

            Fwk.web_service_GET (
                '../portal/ws/expswitch_next.php' ,
                {instr_name: this.instrument, station: this.station} ,
                function (data) {

                    // Initialize the experiment selector

                    var html =
                        _.reduce(data.experiments, function (html, e) {
                            html +=
'<option value="'+e.id+'">'+_.escape(e.name)+'</option>' ;
                            return html ;
                        } , '') ;

                    _that._next_experiment_select().html(html) ;
                    _that._next_experiment_select().val(data.current.id) ;
                    _that._next_experiment_select().change(function () {
                        var exper_id = parseInt($(this).val()) ;
                        _that._next_checktable().remove(function (row) { return row.rank == 'PI' ; }) ;
                        var exper = _.find(data.experiments, function (e) { return e.id == exper_id ; }) ;
                        _.each(exper.contact, function (u) {
                            _that._next_checktable().insert_front({
                                'notify' : true ,
                                'gecos'  : u.gecos ,
                                'email'  : u.email ,
                                'role'   : '<b>Experiment PI or contact</b>' ,
                                'rank'   : 'PI' ,
                                'uid'    : u.uid
                            }) ;
                        }) ;
                    }) ;

                    // Populate the table of people to be notified

                    var rows = [].concat(
                        _.map(data.current.contact, function (u) {
                            return {
                                'notify' : true ,
                                'gecos'  : u.gecos ,
                                'email'  : u.email ,
                                'role'   : '<b>Experiment PI or contact</b>' ,
                                'rank'   : 'PI' ,
                                'uid'    : u.uid
                            } ;
                        }) ,
                        _.map(data.data_managers, function (u) {
                            return {
                                'notify' : true ,
                                'gecos'  : u.gecos ,
                                'email'  : u.email ,
                                'role'   : 'LCLS data administrators' ,
                                'rank'   : 'ADMIN' ,
                                'uid'    : u.uid
                            } ;
                        }) ,
                        _.map(data.instr_group_members, function (u) {
                            return {
                                'notify' : true ,
                                'gecos'  : u.gecos ,
                                'email'  : u.email ,
                                'role'   : 'Instrument support group <b>'+data.instr_group+'</b>' ,
                                'rank'   : 'IS' ,
                                'uid'    : u.uid
                            } ;
                        })
                    ) ;
                    _that._next_checktable().append(rows) ;

                    // Update the current experimen info too

                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data.current) ;
                }
            ) ;
        } ;

        this._turn_editor_on = function (on) {

            this._activate_button(!on) ;
            this._save_button    ( on) ;
            this._cancel_button  ( on) ;
            this._refresh_button (!on) ;

            this._current(!on) ;
            this._next   (on) ;
        } ;

        this._activate = function () {
            this._turn_editor_on(true) ;
            this._load_next_experiments() ;
        } ;
        this._activate_cancel = function () {
            this._turn_editor_on(false) ;
            this._load() ;
        } ;
        this._activate_save = function () {

            this._set_updated('Activating...') ;

            Fwk.web_service_POST (
                '../portal/ws/expswitch_save.php' ,
                {   instr_name : this.instrument ,
                    station    : this.station ,
                    exper_id   : parseInt(this._next_experiment_select().val()) ,
                    message    : this._next_message().val() ,
                    notify     :
                        JSON.stringify (
                            _.map (
                                this._next_checktable().find_checked() ,
                                function (row) { return {
                                    uid      : row.uid,
                                    gecos    : row.gecos ,
                                    email    : row.email ,
                                    rank     : row.rank ,
                                    notified : 'YES' } ; }))
                } ,
                function (data) {

                    _that._turn_editor_on(false) ;

                    // Update the current experimen info too

                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data.current) ;
                }
            ) ;
        } ;
    }
    Class.define_class (ExpSwitch_Station, FwkApplication, {}, {}) ;

    return ExpSwitch_Station ;
}) ;
