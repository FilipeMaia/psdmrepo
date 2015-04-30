define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class' ,
    'webfwk/FwkApplication' ,
    'webfwk/Fwk' ,
    'webfwk/TextInput' ,
    'regdb/DRPMgr_Defs'] ,

function (
    cssloader ,
    Class ,
    FwkApplication ,
    Fwk ,
    TextInput ,
    DRPMgr_Defs ) {

    cssloader.load('../regdb/css/DRPMgr_Policy.css') ;
    cssloader.load('../webfwk/css/Table.css') ;

    /**
     * The application for displaying and managing general policies
     *
     * @returns {DRPMgr_Policy}
     */
    function DRPMgr_Policy (app_config) {

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

        this._policy = null ;

        var _DOCUMENT = {
            input:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Press ENTER to save the new value of the parameter in the database.') ,
            update:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Click this button to update the Policy parameters \n' +
                    'from the database.')
        } ;
        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ;
                if (!this_html) {
                    this_html =
'<div id="drpmgr-policy" >' +
  '<div id="tabs"> ' +
    '<ul> ' ;
                    for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                        var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                        this_html +=
      '<li><a href="#'+storage_class.name+'" >'+storage_class.name+'</a></li> ' ;
                    }
                    this_html +=
    '</ul> ' ;
                    for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                        var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                        this_html +=
    '<div id="'+storage_class.name+'" class="panel" > ' +

      '<div class="info" id="updated" style="float:right;" >Loading...</div> ' +
      '<div style="clear:both;" ></div> ' +

      '<table style="float:left;" > ' +
        '<thead>' +
          '<tr>' +
            '<td class="table_hdr" >Parameter</td> ' +
            '<td class="table_hdr" >Definition</td> ' +
            '<td class="table_hdr" >Units</td> ' +
            '<td class="table_hdr" >Value</td> ' +
            '<td class="table_hdr" >If not set then</td> ' +
          '</tr> ' +
        '</thead>' +
        '<tbody>' ;
                        for (var j in storage_class.parameters) {
                            var param = storage_class.parameters[j] ;
                            this_html +=                        
          '<tr> ' +
            '<td class="table_cell"            valign="top" >'+param.title+'</td> ' +
            '<td class="table_cell definition" valign="top" >'+param.definition+'</td> ' +
            '<td class="table_cell"            valign="top" >'+param.units+'</td> ' +
            '<td class="table_cell"            valign="top" ><input type="text" ' +
                                                                   'name="'+param.name+'" ' +
                                                                   'size="'+param.input.size+'" /></td> ' +
            '<td class="table_cell"            valign="top" >'+param.if_not_set_then+'</td> ' +
          '</tr> ' ;
                        }
                        this_html +=
        '</tbody>' +
      '</table> ' +

      '<div class="buttons" style="float:left;" > ' +
        '<button name="update" class="control-button" '+_DOCUMENT.update+' ><img src="../webfwk/img/Update.png" /></button> ' +
      '</div> ' +
      '<div style="clear:both;" ></div> ' +
    '</div> ' ;
                    }
                    this_html +=
  '</div> ' +
'</div>' ;
                }
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('#drpmgr-policy') ;
            }
            return this._wa_elem ;
        } ;
        this._panel = function (storage_class_name) {
            if (!this._tabs_obj) {
                this._tabs_obj = this._wa().children('#tabs').tabs() ;
                this._panel_elem = {} ;
            }
            if (!this._panel_elem[storage_class_name]) {
                this._panel_elem[storage_class_name] = this._tabs_obj.children('div#'+storage_class_name) ;
            }
            return this._panel_elem[storage_class_name] ;
        } ;
        this._parameter = function (storage_class_name, param_name) {
            if (!this._parameter_obj) { this._parameter_obj = {} ; }
            if (!this._parameter_obj[storage_class_name]) {
                this._parameter_obj[storage_class_name] = {} ;
            }
            if (!this._parameter_obj[storage_class_name][param_name]) {
                for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                    var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                    if (storage_class_name === storage_class.name) { 
                        for (var j in storage_class.parameters) {
                            var param = storage_class.parameters[j] ;
                            if (param_name === param.name) {
                                this._parameter_obj[storage_class_name][param_name] = new TextInput (
                                    this._panel(storage_class_name).find('input[name="'+param_name+'"]') ,
                                    {   disabled:      true ,
                                        default_value: param.input.value ,
                                        on_validate:   param.input.on_validate ,
                                        on_change:     function () { _that._save() ; }
                                    }
                                ) ;
                                break ;
                            }
                        }
                        break ;
                    }
                }
            }
            return this._parameter_obj[storage_class_name][param_name] ;
        }
        this._set_updated = function (storage_class_name, html) {
            if (!this._updated_elem) { this._updated_elem = {} ; }
            if (!this._updated_elem[storage_class_name]) {
                this._updated_elem[storage_class_name] = this._panel(storage_class_name).children('#updated') ;
            }
            this._updated_elem[storage_class_name].html(html) ;
        } ;
        this._button_load = function (storage_class_name) {
            if (!this._button_load_elem) { this._button_load_elem = {} ; }
            if (!this._button_load_elem[storage_class_name]) {
                this._button_load_elem[storage_class_name] = this._panel(storage_class_name).find('.control-button[name="update"]').button() ;
            }
            return this._button_load_elem[storage_class_name] ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // Touch visual objects to make sure they're displayed before
            // going for any loading of data
            for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                for (var j in storage_class.parameters) {
                    var param = storage_class.parameters[j] ;
                    this._parameter(storage_class.name, param.name) ;
                }
                this._button_load(storage_class.name) .click(function () { _that._load() ; }) ;
            }

            // Proceed to the first loading
            this._load() ;
        } ;
        this._load = function () {
            if (!this._is_initialized) return ;
            this._action (
                'Saving...' ,
                '../regdb/ws/drp_policy_get.php' ,
                {}
            ) ;
        } ;
        this._save = function () {
            if (!this._is_initialized) return ;
            var params = {} ;
            for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                for (var j in storage_class.parameters) {
                    var param = storage_class.parameters[j] ;
                    params[storage_class.name+':'+param.name.toUpperCase()] = this._parameter(storage_class.name, param.name).value() ;
                }
            }
            this._action (
                'Saving...' ,
                '../regdb/ws/drp_policy_save.php' ,
                params
            ) ;
        } ;
        this._action = function (operation, url, params) {
            for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                this._set_updated(storage_class.name, operation) ;
                this._button_load(storage_class.name).button('disable') ;
            }
            Fwk.web_service_GET (
                url ,
                params ,
                function (data) {
                    _that._policy = data.policy ;
                    _that._display() ;
                    for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                        var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                        _that._set_updated(storage_class.name, 'Last updated: <b>'+data.updated+'</b>') ;
                        _that._button_load(storage_class.name).button('enable') ;
                    }
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                        var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                        _that._button_load(storage_class.name).button('enable') ;
                    }
                }
            ) ;
        } ;
        this._display = function () {
            for (var i in DRPMgr_Defs.STORAGE_CLASS) {
                var storage_class = DRPMgr_Defs.STORAGE_CLASS[i] ;
                for (var j in storage_class.parameters) {
                    var param = storage_class.parameters[j] ;
                    this._parameter(storage_class.name, param.name).set_value (
                        this._policy[storage_class.name][param.name]) ;
                }
            }
        } ;
    }
    Class.define_class (DRPMgr_Policy, FwkApplication, {}, {}) ;
    
    return DRPMgr_Policy ;
}) ;



