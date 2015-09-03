define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/RadioBox', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, RadioBox, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Experiment_ELogAccess.css') ;

    /**
     * The application for managing access to the Electronic Logbook of teh application
     *
     * @returns {Experiment_ELogAccess}
     */
    function Experiment_ELogAccess (experiment, access_list) {

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

        this._update_interval_sec = 30 ;
        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > this._update_interval_sec) {
                    this._prev_update_sec = now_sec ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment  = experiment ;
        this.access_list = access_list ;

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
                this.container.html('<div id="exp-elog"></div>') ;
                this._wa_elem = this.container.find('div#exp-elog') ;
                if (html === undefined) {
                    html =
'<div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'<div style="clear:both;"></div>' +

'<div id="body" style="float:left;">' +
'  <div id="operator">' +
'    <h2>Access privileges for the operator account \''+this.experiment.operator_uid+'\': </h2>' +
'    <div>' +
'      <p align="justify">' +
'        This section displays and manages access privileges of the instrument operator\'s account' +
'        to the Electronic LogBook of the experiment. The access level is selected through the' +
'        radio-buttons box show below. Changes would get into effect immediatelly after pressing' +
'        the corresponding button.</p>' +
'      <div id="access" ></div>' +
'      <p><b>ATTENTION</b>: Disabling the WRITE access to e-Log of the on-going experiment' +
'         is not allowed because this will prevent e-Log <b>Grabber</b> from posting images into e-Log.</p>' +
'    </div>' +
'  </div>' +
'</div>' +

'<div id="buttons" style="float:left;" >' +
'  <button class="control-button"' +
'          name="update"' +
'          title="update the page" ><img src="../webfwk/img/Update.png" /></button>' +
'</div>' +

'<div style="clear:both;" ></div>' ;
                }
                this._wa_elem.html(html) ;
            }
            return this._wa_elem ;
        } ;
        this._body = function () {
            if (!this._body_elem) {
                this._body_elem = this._wa().children('#body') ;
            }
            return this._body_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) {
                this._updated_elem = this._wa().children('#updated') ;
            }
            this._updated_elem.html(html) ;
        } ;
        this._operator_access = function () {
            if (!this._operator_access_obj) {
                this._operator_access_obj = new RadioBox (
                    [   {name: "NoAccess", text: "NO ACCESS", title: "No access allowed"} ,
                        {name: "Reader",   text: "READER",    title: "Can read the contents"} ,
                        {name: "Writer",   text: "WRITER",    title: "Can read, post and extend messages"} ,
                        {name: "Editor",   text: "EDITOR",    title: "Can edit and delete the contents"}
                    ] ,
                    function (name) { _that._set(name) ; }
                ) ;
                this._operator_access_obj.display(this._body().children('#operator').find('#access')) ;
            }
            return this._operator_access_obj ;
        } ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // -- no further initialization beyond this point if not authorized

            if (!this.access_list.experiment.view_info) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }
            if (!this.experiment.operator_uid) {
                this._wa("This experiment doesn't seem to have an operator account to be managed") ;
                return ;
            }

            // -- set up event handlers

            this._wa().find('button[name="update"]').button().click(function () { _that._load() ; }) ;

            this._operator_access().activate('NoAccess') ;

            this._load() ;
        } ;

        this._load = function () {
            if (!this.access_list.experiment.view_info) return ;
            this._load() ;
        } ;

        this._load = function () {
            Fwk.web_service_GET (
                '../portal/ws/experiment_elog_access_get.php',
                {   exper_id: this.experiment.id ,
                    uid:      this.experiment.operator_uid} ,
                function (data) {
                    _that._set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._display(data) ;
                }
            ) ;
        } ;
        this._display = function (data) {
            this._operator_access().activate(data.role) ;
        } ;
        this._set = function (role) {
            Fwk.web_service_GET (
                '../portal/ws/experiment_elog_access_set.php',
                {   exper_id: this.experiment.id ,
                    uid:      this.experiment.operator_uid ,
                    role:     role
                } ,
                function (data) {
                    _that._set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._display(data) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._load() ;
                }
            ) ;
        } ;
    }
    Class.define_class (Experiment_ELogAccess, FwkApplication, {}, {}) ;

    return Experiment_ELogAccess ;
}) ;

