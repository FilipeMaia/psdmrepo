define ([    
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/CheckTable', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, CheckTable, FwkApplication, Fwk) {

    cssloader.load('../portal/css/ExpSwitch_ELogAccess.css') ;

    /**
     * The application for managing access to the Electronic LogBook
     * of the instrument's experiments by the corresponing operator.
     *
     * @returns {ExpSwitch_ELogAccess}
     */
    function ExpSwitch_ELogAccess (instrument, operator_uid, access_list) {

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
            this._update() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                //this._update() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.instrument   = instrument ;
        this.operator_uid = operator_uid ;
        this.access_list  = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._wa = function (html) {
            if (this._wa_elem) {
                if (html !== undefined) {
                    this._wa_elem.html(html) ;
                }
            } else {
                this.container.html('<div id="expswitch-elog"></div>') ;
                this._wa_elem = this.container.find('div#expswitch-elog') ;
                if (html === undefined) {
                    html =
'<div id="ctrl">' +
  '<div style="float:left;" >' +
    '<button class="control-button" ' +
            'name="refresh" ' +
            'title="refresh the page" >REFRESH</button>' +
  '</div>' +
  '<div style="clear:both;" ></div>' +
'</div>' +
'<div id="body">' +
  '<div class="info" id="info"    style="float:left;">&nbsp;</div>' +
  '<div class="info" id="updated" style="float:right;">&nbsp;</div>' +
  '<div style="clear:both;"></div>' +
  '<div id="operator">' +
    '<h2>Access privileges for operator account \''+this.operator_uid+'\':</h2>' +
    '<div>' +
      '<p align="justify" >' +
      ' This section displays current e-Log authorizations of the instrument operator\'s account.' +
      ' Unnecessary authorizations can be revoked by selecting the corresponding table rows' +
      ' and then clicking on the <span>REVOKE SELECTED</span> button. Note, that authorizations' +
      ' of the active experiment can\'t be revoked because this will interfere with proper functioning' +
      ' of the <span>e-Log Grabber</span> application in the experiment\s hutch. To grant authorizations' +
      ' for additional experiments one should go directly to the <a href="select_experiment" target="_blank" >Data Manager</a>' +
      ' of a desired experiment and use the <span>e-Log Access</span> tool of the experiment.</p>' +
      '<button class="control-button" ' +
              'name="select" ' +
              'title="select all but the active experiment\s authorizations" >SELECT ALL</button> ' +
      '<button class="control-button" ' +
              'name="unselect" ' +
              'title="un-select all authorizations" >UN-SELECT ALL</button> ' + (this.access_list.can_manage ?
      '<button class="control-button" ' +
              'name="remove" ' +
              'title="revoke selected authorizations" >REVOKE SELECTED</button> ' : '') +
      '<div id="checktable" ></div>' +
    '</div>' +
  '</div>' +
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
        this._body = function () {
            if (!this._body_elem) {
                this._body_elem = this._wa().children('#body') ;
            }
            return this._body_elem ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) {
                this._info_elem = this._body().children('#info') ;
            }
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) {
                this._updated_elem = this._body().children('#updated') ;
            }
            this._updated_elem.html(html) ;
        } ;
        this._operator_checktable = function () {
            if (!this._operator_checktable_obj) {
                var coldef = [
                    {name: 'selected',   text:   ''} ,
                    {name: 'experiment', text:   'Experiment', align: 'right'} ,
                    {name: 'role',       text:   'Access',     align: 'right'} ,
                    {name: 'notes',      text:   'Notes'} ,
                    {name: 'exper_id',   hidden: true} ,
                    {name: 'uid',        hidden: true}
                ] ;
                this._operator_checktable_obj = new CheckTable (coldef) ;
                this._operator_checktable_obj.display(this._body().children('#operator').find('#checktable')) ;
            }
            return this._operator_checktable_obj ;
        } ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // -- no further initialization beyond this point if not authorized

            if (!this.access_list.can_read) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }
            if (!this.operator_uid) {
                this._wa("This instrument doesn't seem to have an operator account to be managed") ;
                return ;
            }

            // -- set up event handlers

            this._ctrl().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'refresh': _that._update() ; break ;
                }
            }) ;

            this._body().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'select' :
                        _that._operator_checktable().check(true, function (row) {
                            return !row._locked ;
                        }) ;
                        break ;
                    case 'unselect':
                        _that._operator_checktable().uncheck_all() ;
                        break ;
                    case 'remove' :
                        _that._remove() ;
                        break ;
                }
            }) ;

            this._operator_checktable() ;
            this._update() ;
        } ;

        this._update = function () {
            if (!this.access_list.can_read) return ;
            this._load() ;
        } ;

        this._load = function () {
            _that._set_updated('Loading...') ;
            Fwk.web_service_GET (
                '../portal/ws/expswitch_elog_access_get.php',
                {   instr_name : this.instrument
                } ,
                function (data) {
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data) ;
                }
            ) ;
        } ;
        this._display = function (data) {
            this._operator_checktable().remove_all() ;
            for (var i in data.roles) {
                var r = data.roles[i] ;
                this._operator_checktable().append({
                    selected   : false ,
                    experiment : r.exper.name ,
                    role       : r.role ,
                    notes      : r.exper.is_active ? 'active experiment' : '' ,
                    exper_id   : ''+r.exper.id ,
                    uid        : r.uid ,
                    _locked    : r.exper.is_active ? true : false
                }) ;
            }
        } ;
        this._remove = function () {

            var auth2remove = [] ;
            var rows = this._operator_checktable().find_checked() ;
            for (var i in rows) {
                var r = rows[i] ;
                auth2remove.push({
                    exper_id: r.exper_id ,
                    uid:      r.uid ,
                    role:     r.role
                }) ;
            }
            Fwk.web_service_POST (
                '../portal/ws/expswitch_elog_access_set.php',
                {   instr_name  : this.instrument ,
                    auth2remove : JSON.stringify(auth2remove)
                } ,
                function (data) {
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._load() ;
                }
            ) ;
        } ;
    }
    Class.define_class (ExpSwitch_ELogAccess, FwkApplication, {}, {}) ;

    return ExpSwitch_ELogAccess ;
}) ;