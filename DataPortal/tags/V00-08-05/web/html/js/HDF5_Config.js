define ([
    'webfwk/CSSLoader', 'webfwk/PropList' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader, PropList ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/HDF5_Config.css') ;

    /**
     * The application for configuring the HDF5 translation service for the experiment
     *
     * @returns {HDF5_Config}
     */
    function HDF5_Config (experiment, access_list) {

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

        this._update_interval_sec = 300 ;
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

        this._can_manage = function () { return this.access_list.hdf5.is_data_administrator ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="hdf5-config" >' +

  '<div style="float:left;" class="hdf5-notes" >' +
    '<p>This application manages various configuration parameters of the translation service' +
    '   set up for the experiment.</p>' +
  '</div>' +
  '<div style="float:right;" >' +
    '<button name="update" class="control-button" title="update from the database" >UPDATE</button>' +
  '</div>' +
  '<div style="clear:both;" ></div>' +

  '<div class="info" id="info"    style="float:left;"  >&nbsp;</div>' +
  '<div class="info" id="updated" style="float:right;" >&nbsp;</div>' +
  '<div style="clear:both;"></div>' +

  '<div id="tabs" >' +

    '<ul>' +
      '<li><a href="#translator" >Translator</a></li>' +
    '</ul>' +

    '<div id="translator" >' +
      '<div class="tab-body" >' +
        '<div class="table-ctrl" >' +
          '<div style="float:left;" class="hdf5-notes" >' +
            '<p>By default the translation service would be launching the <b>Translator</b> from' +
            '   the latest analysis release. This would work for most experiments at LCLS. Experiments' +
            '   may override this behavior by providing an absolute path to a top level directory of' +
            '   a specific release, or by providing a different configuration file for the <b>Translator</b> job.' +
            '   To switch back to the default configuration one should just clear both fields and' +
            '   save results to the database.</p>' +
          '</div>' +
          '<div style="float:right;" class="buttons" >' +
            '<button name="edit"    class="control-button control-button-important" title="edit configuration options"         >EDIT  </button>' +
            '<button name="save"    class="control-button"                          title="save modifications to the database" >SAVE  </button>' +
            '<button name="cancel"  class="control-button"                          title="cancel the editing session"         >CANCEL</button>' +
          '</div>' +
          '<div style="clear:both;" ></div>' +
      '</div>' +
        '<div id="config" ></div>' +
      '</div>' +
    '</div>' +

  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#hdf5-config') ;
            }
            return this._wa_elem ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().children('#info') ;
            this._info_elem.html(html) ;
        } ;
        
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._button_update = function () {
            if (!this._button_update_elem) {
                this._button_update_elem = this._wa().find('button[name="update"]').button() ;
            }
            return this._button_update_elem ;
        } ;
        this._button_edit = function () {
            if (!this._button_edit_elem) this._button_edit_elem = this._wa().find('button[name="edit"]').button() ;
            return this._button_edit_elem ;
        } ;
        this._button_save = function () {
            if (!this._button_save_elem) this._button_save_elem = this._wa().find('button[name="save"]').button() ;
            return this._button_save_elem ;
        } ;
        this._button_cancel = function () {
            if (!this._button_cancel_elem) this._button_cancel_elem = this._wa().find('button[name="cancel"]').button() ;
            return this._button_cancel_elem ;
        } ;
        this._translator_config = function () {
            if (!this._translator_config_obj) {
                this._translator_config_obj = new PropList ([
                    {name: "release_dir", text: "Release directory",       edit_mode: true, editor: 'text' ,   edit_size: 64} ,
                    {name: "config_file", text: "Configuration file",      edit_mode: true, editor: 'text',    edit_size: 64} ,
                    {name: "auto",        text: "Enable Auto-Translation", edit_mode: true, editor: 'checkbox'} ,
                    {name: "ffb",         text: "Input from FFB",          edit_mode: true, editor: 'checkbox'}
                ]) ;
                this._translator_config_obj.display(this._wa().find('div#translator').find('div#config')) ;
            }
            return this._translator_config_obj ;
        } ;

        /**
         * Returns and (optionally) set the editing mode. Changes control
         * buttons accordingly.
         *
         * @param {Boolean} editing
         * @returns {Boolean}
         */
        this._edit_mode = function (editing) {
            if (this._editing === undefined) this._editing = false ;
            if (editing !== undefined) {
                if (this._can_manage()) {
                    this._button_update().button(editing ? 'disable' : 'enable') ;
                    this._button_edit  ().button(editing ? 'disable' : 'enable') ;
                    this._button_save  ().button(editing ? 'enable'  : 'disable') ;
                    this._button_cancel().button(editing ? 'enable'  : 'disable');
                    this._editing = editing ;
                } else {
                    this._button_update().button('enable') ;
                    this._button_edit  ().button('disable') ;
                    this._button_save  ().button('disable') ;
                    this._button_cancel().button('disable');
                }
            }
            return this._editing ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this.access_list.hdf5.read) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }
            this._wa().children('#tabs').tabs() ;

            this._button_update().click(function () { _that._load() ; }) ;
            this._button_edit  ().click(function () { _that._edit() ; }) ;
            this._button_save  ().click(function () { _that._edit_save() ; }) ;
            this._button_cancel().click(function () { _that._edit_cancel() ; }) ;

            this._translator_config().set_value('release_dir', 'Loading...') ;
            this._translator_config().set_value('config_file', 'Loading...') ;
            this._translator_config().set_value('auto',        'Loading...') ;
            this._translator_config().set_value('ffb',         'Loading...') ;

            this._edit_mode(false) ;
            this._load() ;
        } ;

        this._load = function () {

            if (!this._edit_mode(false)) {
                var params = {
                    exper_id: this.experiment.id
                } ;
                this._set_updated('Updating...') ;

                Fwk.web_service_GET (
                    '../portal/ws/hdf5_config_get.php' ,
                    params ,
                    function (data) {
                        _that._config = data.config ;
                        _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                        _that._display() ;
                    }
                ) ;
            }
        } ;
        this._edit = function () {
            this._edit_mode(true) ;
            this._translator_config().edit_value('release_dir') ;
            this._translator_config().edit_value('config_file') ;
            this._translator_config().edit_value('auto') ;
            this._translator_config().edit_value('ffb') ;
        } ;
        this._edit_save = function () {
            this._edit_mode(false) ;
            var params = {
                exper_id:    this.experiment.id ,
                release_dir: this._translator_config().get_value('release_dir') ,
                config_file: this._translator_config().get_value('config_file') ,
                auto:        this._translator_config().get_value('auto') ? 1 : 0 ,
                ffb:         this._translator_config().get_value('ffb')  ? 1 : 0
            } ;
            this._set_updated('Saving...') ;

            Fwk.web_service_POST (
                '../portal/ws/hdf5_config_set.php' ,
                params ,
                function (data) {
                    _that._config = data.config ;
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._translator_config().view_value('release_dir') ;
                    _that._translator_config().view_value('config_file') ;
                    _that._translator_config().view_value('auto') ;
                    _that._translator_config().view_value('ffb') ;
                    _that._display() ;
                }
            ) ;
        } ;
        this._edit_cancel = function () {
            this._edit_mode(false) ;
            this._translator_config().view_value('release_dir') ;
            this._translator_config().view_value('config_file') ;
            this._translator_config().view_value('auto') ;
            this._translator_config().view_value('ffb') ;
            this._display() ;
        } ;
        this._display = function () {
            this._translator_config().set_value('release_dir', this._config.release_dir) ;
            this._translator_config().set_value('config_file', this._config.config_file) ;
            this._translator_config().set_value('auto',        this._config.auto) ;
            this._translator_config().set_value('ffb',         this._config.ffb) ;
        } ;
    }
    Class.define_class (HDF5_Config, FwkApplication, {}, {}) ;

    return HDF5_Config ;
}) ;
