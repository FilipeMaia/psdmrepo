define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class' ,
    'webfwk/FwkApplication' ,
    'webfwk/Fwk' ,
    'webfwk/SimpleTable' ,
    'regdb/DRPMgr_Defs'] ,

function (
    cssloader ,
    Class ,
    FwkApplication ,
    Fwk ,
    SimpleTable ,
    DRPMgr_Defs) {

    cssloader.load('../regdb/css/DRPMgr_Exceptions.css') ;

    /**
     * The application for displaying and managin experiment-specific
     * policy exceptions.
     *
     * @returns {DRPMgr_Exceptions}
     */
    function DRPMgr_Exceptions (app_config) {

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
        this._loading_is_allowed = true ;

        this._allow_loading = function (yes_or_no) {
            this._loading_is_allowed = yes_or_no ? true : false ;
            this._button_add ().button(this._loading_is_allowed ? 'enable' : 'disable') ;
            this._button_load().button(this._loading_is_allowed ? 'enable' : 'disable') ;
            this._table().enable_sort(this._loading_is_allowed) ;
            this._table().get_container().find('.control-button').button(this._loading_is_allowed ? 'enable' : 'disable') ;
            if (this._loading_is_allowed)
                this._table().get_container().find('.control-input').removeAttr('disabled') ;
            else
                this._table().get_container().find('.control-input').attr('disabled', 'disabled') ;
        } ;

        this._experiments = [] ;
        this._last_updated = '' ;

        var _DOCUMENT = {
            add:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Add another experiment to the table for managing its exceptions.') ,
            update:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Update exceptions from the database.') ,
            remove:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Remove all exceptions for this experiment.') ,
        } ;
        var _DATA_RETENTION_POLICY_URL =
            "https://confluence.slac.stanford.edu/display/PCDS/Data+Retention+Policy" ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ;
                if (!this_html) {
                    this_html =
'<div id="drpmgr-exceptions" >' +

    '<div class="info" id="updated" style="float:right;" >Loading...</div> ' +
    '<div style="clear:both;" ></div> ' +

    '<div id="notes" style="float:left;" > ' +
      '<p>This application allows to view and manage experiment-specific exceptions to general the ' +
         '<a class="link" target="_blank" href="'+_DATA_RETENTION_POLICY_URL+'" >LCLS Data Retention Policy</a>. ' +
         'Use the <b>ADD EXPERIMENT</b> button to add ' +
         'a new experiment which is not listed in the table below. Use the <b>REMOVE</b> ' +
         'button to remove an experiment in the corresponidng row of the table.</p> ' +
      '<p>Editing instructions: <b>ENTER</b> - save the value in the database, <b>ESC</b> - discared ' +
         'modifications. Put an empty string to remove an exception for ' +
         'the corresponding parameter. If all exceptions for an experiment ' +
         'are removed then the corresponding experiment as well would be completelly from the table.</p> ' +
      '<p><b>NOTE</b>:all active elements of the application except ones of an experiment ' +
         'which is being edited will remain locked while editing any parameter of ' +
         'the experiment. And no automatic updates to the table will be made until ' +
         'the editing is over.</p> ' +
    '</div> ' +

    '<div id="buttons" style="float:left;" > ' +
      '<button name="add"    class="control-button control-button-important" '+_DOCUMENT.add   +' >ADD EXPERIMENT</button> ' +
      '<button name="update" class="control-button" '                         +_DOCUMENT.update+' ><img src="../webfwk/img/Update.png" /></button> ' +
    '</div> ' +
    '<div style="clear:both;" ></div> ' +

    '<div id="experiments" ></div> ' +

'</div>' ;
                }
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('#drpmgr-exceptions') ;
            }
            return this._wa_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) {
                this._updated_elem = this._wa().children('#updated') ;
            }
            this._updated_elem.html(html) ;
        } ;
        this._button_add = function () {
            if (!this._button_add_elem) {
                this._button_add_elem = this._wa().children('#buttons').children('.control-button[name="add"]').button() ;
            }
            return this._button_add_elem ;
        } ;
        this._button_load = function () {
            if (!this._button_load_elem) {
                this._button_load_elem = this._wa().children('#buttons').children('.control-button[name="update"]').button() ;
            }
            return this._button_load_elem ;
        } ;
        this._table = function () {
            if (!this._table_obj) {
                var rows = [] ;
                var hdr = [{
                    name: 'instr'}, {
                    name: 'exper', align: 'right'}, {
                    name: 'SHORT-TERM', coldef: [{
                        name: 'ctime', type: {
                            to_string:
                                function (a) {
                                    return SimpleTable.html.TextInput ({
                                        value: a.policy['SHORT-TERM'].ctime ,
                                        name: 'SHORT-TERM:ctime' ,
                                        id: a.id ,
                                        size: 6 ,
                                        classes: 'control-input'}) ;
                                } ,
                            compare_values:
                                function (a,b) {
                                    return this.compare_strings (
                                        a.policy['SHORT-TERM'].ctime ,
                                        b.policy['SHORT-TERM'].ctime) ;
                                }}}, {
                        name: 'retention', type: {
                            to_string:
                                function (a) {
                                    return SimpleTable.html.TextInput ({
                                        value: a.policy['SHORT-TERM'].retention ,
                                        name: 'SHORT-TERM:retention' ,
                                        id: a.id ,
                                        size: 2 ,
                                        classes: 'control-input'}) ;
                                } ,
                            compare_values:
                                function (a,b) {
                                    return parseInt(a.policy['SHORT-TERM'].retention) -
                                           parseInt(b.policy['SHORT-TERM'].retention) ;
                                }}}]}, {
                    name: 'MEDIUM-TERM', coldef: [{
                        name: 'ctime', type: {
                            to_string:
                                function (a) {
                                    return SimpleTable.html.TextInput ({
                                        value: a.policy['MEDIUM-TERM'].ctime ,
                                        name: 'MEDIUM-TERM:ctime' ,
                                        id: a.id ,
                                        size: 6 ,
                                        classes: 'control-input'}) ;
                                } ,
                            compare_values:
                                function (a,b) {
                                    return this.compare_strings (
                                        a.policy['MEDIUM-TERM'].ctime ,
                                        b.policy['MEDIUM-TERM'].ctime) ;
                                }}}, {
                        name: 'retention', type: {
                            to_string:
                                function (a) {
                                    return SimpleTable.html.TextInput ({
                                        value: a.policy['MEDIUM-TERM'].retention ,
                                        name: 'MEDIUM-TERM:retention' ,
                                        id: a.id ,
                                        size: 2 ,
                                        classes: 'control-input'}) ;
                                } ,
                            compare_values:
                                function (a,b) {
                                    return parseInt(a.policy['MEDIUM-TERM'].retention) -
                                           parseInt(b.policy['MEDIUM-TERM'].retention) ;
                                }}}, {
                        name: 'quota', type: {
                            to_string:
                                function (a) {
                                    return SimpleTable.html.TextInput ({
                                        value: a.policy['MEDIUM-TERM'].quota ,
                                        name: 'MEDIUM-TERM:quota' ,
                                        id: a.id ,
                                        size: 6 ,
                                        classes: 'control-input'}) ;
                                } ,
                            compare_values:
                                function (a,b) {
                                    return parseInt(a.policy['MEDIUM-TERM'].quota) -
                                           parseInt(b.policy['MEDIUM-TERM'].quota) ;
                                }}}]}, {
                    name: 'ACTIONS', sorted: false, type: {
                            to_string:
                                function (exper_id) {
                                    return SimpleTable.html.Button ('REMOVE', {
                                        id: exper_id ,
                                        classes: 'control-button control-button-important' ,
                                        extra: _DOCUMENT.remove}) ;
                                }}}
                ] ;
                this._table_obj = new SimpleTable.constructor (
                    this._wa().children('#experiments') ,
                    hdr ,
                    rows ,
                    {   text_when_empty: null ,
                        common_after_sort: function () {
                            _that._table().get_container().find('.control-input').keyup(function (e) {
                                var exper_id = this.id ;
                                if (e.keyCode === 13) {
                                    _that._save_experiment(exper_id) ;
                                } else if (e.keyCode === 27) {
                                    _that._display() ;
                                    _that._allow_loading(true) ;
                                } else {
                                    _that._set_updated('Editing...') ;
                                    _that._allow_loading(false) ;
                                    _that._table().get_container().find('.control-input#'+exper_id).removeAttr('disabled') ;
                                    _that._table().get_container().find('.control-button#'+exper_id).button('enable') ;
                                }
                            }) ;
                            _that._table().get_container().find('.control-button').button().click(function () {
                                var exper_id = this.id ;
                                var remove = true ;
                                _that._save_experiment(exper_id, remove) ;
                            }) ;
                        }
                    }
                ) ;
                this._table_obj.display(this._wa().find('#experiment')) ;
            }
            return this._table_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this._app_config.access_control.can_edit) {
                this._wa(this._app_config.access_control.no_page_access_html) ;
                return ;
            }
            
            this._button_add ().click(function () { _that._add_experiment() ; }) ;
            this._button_load().click(function () { _that._load() ; }) ;

            // Make sure the table is shown before the loading takes place.
            this._table() ;

            // Proceed to the first loading
            this._load() ;
        } ;
        this._display = function () {
            var rows = [] ;
            for (var i in _that._experiments) {
                var exper = _that._experiments[i] ;
                rows.push([
                    exper.instr_name ,
                    exper.name ,
                    exper ,     // SHORT-TERM:ctime
                    exper ,     // SHORT-TERM:retention
                    exper ,     // MEDIUM-TERM:ctime
                    exper ,     // MEDIUM-TERM:retention
                    exper ,     // MEDIUM-TERM:quota
                    exper.id    // ACTIONS: REMOVE, etc.
                ]) ;
            }
            this._table().load(rows) ;
            this._set_updated('Last updated: <b>'+this._last_updated+'</b>') ;
        } ;
        this._action = function (msg, url, params, on_success) {

            this._set_updated(msg) ;
            this._allow_loading(false) ;

            Fwk.web_service_GET (
                url ,
                params ,
                function (data) {
                    _that._experiments  = data.experiments ;
                    _that._last_updated = data.updated ;
                    _that._display() ;
                    _that._allow_loading(true) ;
                    if (on_success) on_success() ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._set_updated('Last updated: <b>'+_that._last_updated+'</b>') ;
                    _that._allow_loading(true) ;
                }
            ) ;
        } ;
        this._load = function () {
            if (!this._is_initialized) return ;
            if (!this._loading_is_allowed) return ;
            this._action (
                'Loading...' ,
                '../regdb/ws/drp_exceptions_get.php' ,
                {}
            ) ;
        } ;
        this._focus_at_experiment = function (exper_id) {

            var focus_at = this._table().get_container().find('.control-input#'+exper_id) ;
            var focus_relative_to = $('#fwk-center') ;
            var offset_top = focus_relative_to.scrollTop() + focus_at.position().top - 24;
            focus_relative_to.animate({scrollTop: offset_top}, 'slow') ;
            focus_at.focus() ;
            this._set_updated('Editing...') ;
            this._allow_loading(false) ;
            this._table().get_container().find('.control-input#'+exper_id).removeAttr('disabled') ;
            this._table().get_container().find('.control-button#'+exper_id).button('enable') ;
        } ;
        this._add_experiment = function () {
            Fwk.ask_for_line (
                'Adding Experiment Exception' ,
                'What is the name of an experiment which needs to be added?' ,
                function (exper_name) {
                    var exper = _.find(_that._experiments, function (e) { return e.name === exper_name ; }) ;
                    if (exper) {
                        _that._focus_at_experiment(exper.id) ;
                    } else {
                        _that._action (
                            'Adding...' ,
                            '../regdb/ws/drp_exceptions_add.php' ,
                            {   exper_name: exper_name
                            } ,
                            function () {
                                var exper = _.find(_that._experiments, function (e) { return e.name === exper_name ; }) ;
                                if (exper)
                                    _that._focus_at_experiment(exper.id) ;
                            }
                        ) ;
                    }
                }
            ) ;
        } ;
        /**
         * Save Policy exceptions for an experiment. The experiment will be
         * removed (for the white list of exceptions) if the 'remove'
         * is set to true.
         *
         * @param {integer} exper_id
         * @param {boolean} remove
         * @returns {undefined}
         */
        this._save_experiment = function (exper_id, remove) {
            this._set_updated(remove ? 'Removing...' : 'Saving...') ;
            var table_cont = this._table().get_container() ;
            var params = {} ;
            table_cont.find('.control-input#'+exper_id).each(function () {
                var storage_class_param = this.name.split(':') ;
                var storage_class = storage_class_param[0] ,
                            param = storage_class_param[1] ;
                if (_.isUndefined(params[storage_class])) params[storage_class] = {} ;
                params[storage_class][param] = remove ? '' : $(this).val() ;
            }) ;
            this._action (
                'Saving...' ,
                '../regdb/ws/drp_exceptions_save.php' ,
                {   exper_id: exper_id ,
                    policy: JSON.stringify(params)
                }
            ) ;
        } ;
    }
    Class.define_class (DRPMgr_Exceptions, FwkApplication, {}, {}) ;
    
    return DRPMgr_Exceptions ;
}) ;