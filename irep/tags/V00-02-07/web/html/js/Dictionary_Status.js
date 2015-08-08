define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../irep/css/Dictionary_Status.css') ;

    /**
     * The application for browsing and managing a dictionary of statuses
     *
     * @returns {Dictionary_Status}
     */
    function Dictionary_Status (app_config) {

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

        this._update_ival_sec = 120 ;
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

        /**
         * Preload the dictionary w/o displaying it
         *
         * @returns {undefined}
         */
        this.init = function () {
            this._preload() ;
        } ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._status = null ;

        this.statuses = function () {
            return this._status ;
        } ;

        this._can_manage = function () { return this._app_config.current_user.has_dict_priv ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dictionary-status" >' +

  '<div style="float:right;" >' +
    '<button name="update" class="control-button" title="update from the database" >UPDATE</button>' +
  '</div>' +
  '<div style="clear:both;" ></div>' +

  '<div id="updated-info" >' +
    '<div class="info" id="info"    style="float:left;"  >&nbsp;</div>' +
    '<div class="info" id="updated" style="float:right;" >&nbsp;</div>' +
    '<div style="clear:both;" ></div>' +
  '</div>' +

  '<div id="tables" >' +

    '<div class="table-cont" style="float:left;" >' +
      '<div style="float:left; "><input type="text" size="12" name="status" title="fill in new status name, then press RETURN to save" /></div>' +
      '<div style="float:left; padding-top:4px; color:maroon;" >  &larr; add new status here</div>' +
      '<div style="clear:both; "></div>' +
      '<div id="table-status" class="table" ></div>' +
    '</div>' +

    '<div class="table-cont" style="float:left;" >' +
      '<div style="float:left; "><input type="text" size="12" name="status2" title="fill in new sub-status name, then press RETURN to save" /></div>' +
      '<div style="float:left; padding-top:4px; color:maroon;" >  &larr; add new sub-status here</div>' +
      '<div style="clear:both; "></div>' +
      '<div id="table-status2" class="table" ></div>' +
    '</div>' +

    '<div style="clear:both;" ></div>' +

  '</div>' +

'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dictionary-status') ;
            }
            return this._wa_elem ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().find('#info') ;
            this._info_elem.html(html) ;
        } ;
        
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().find('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._table_status = function () {
            if (!this._table_status_obj) {
                this._table_status_elem = this._wa().find('#table-status') ;
                var hdr =  [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_status_elem.find('.status-delete').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        _that._status_delete(id) ;
                                    }) ;
                            }
                        }
                    }) ;
                hdr.push (
                    {   name: 'status', selectable: true ,
                        type: {
                            select_action : function (status_name) {
                                _that._status2_display() ;
                            }
                        }
                    } ,
                    {   name: 'created', hideable: true } ,
                    {   name: 'by user', hideable: true } ,
                    {   name: 'in use',  hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_status_elem.find('.status-search').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        global_search_equipment_by_status(id) ;
                                    }) ;
                            }
                        }
                    }
                ) ;
                var rows = [] ;
                this._table_status_obj = new SimpleTable.constructor (
                    this._table_status_elem ,
                    hdr ,
                    rows ,
                    {selected_col: this._can_manage() ? 1 : 0} ,
                    Fwk.config_handler('dict', 'table_status')
                ) ;
                this._table_status_obj.display() ;
            }
            return this._table_status_obj ;
        } ;
        this._table_status2 = function () {
            if (!this._table_status2_obj) {
                this._table_status2_elem = this._wa().find('#table-status2') ;
                var hdr =  [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_status2_elem.find('.status2-delete').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        _that._status2_delete(id) ; }) ;
                                _that._table_status2_elem.find('.status2-search').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        global_search_equipment_by_status2(id) ;
                                    }) ;
                            }
                        }
                    }) ;
                hdr.push (
                    {   name: 'sub-status', style: 'font-weight:bold;' } ,
                    {   name: 'created',    hideable: true } ,
                    {   name: 'by user',    hideable: true } ,
                    {   name: 'in use',     hideable: true, sorted: false }
                ) ;
                var rows = [] ;
                this._table_status2_obj = new SimpleTable.constructor (
                    this._table_status2_elem ,
                    hdr ,
                    rows ,
                    {} ,
                    Fwk.config_handler('dict', 'table_status2')
                ) ;
                this._table_status2_obj.display() ;
            }
            return this._table_status2_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().find('button[name="update"]').button().click(function () {
                _that._load() ;
            }) ;
            this._wa().find('input[name="status"]').keyup(function (e) {
                var name = $(this).val() ;
                if (name === '') { return ; }
                if (e.keyCode === 13) { _that._status_create(name) ; $(this).val('') ; return ; }
            }) ;
            this._wa().find('input[name="status2"]').keyup(function (e) {
                var name = $(this).val() ;
                if (name === '') { return ; }
                if (e.keyCode === 13) { _that._status2_create(name) ; $(this).val('') ; return ; }
            }) ;

            this._load() ;
        } ;
        this._load = function () {
            if (!this._is_initialized) return ;
            this._status_action (
                'Loading...' ,
                '../irep/ws/status_get.php' ,
                {}
            ) ;
        } ;
        this._preload = function () {
            var dont_display = true ;
            this._status_action (
                'Loading...' ,
                '../irep/ws/status_get.php' ,
                {} ,
                dont_display
            ) ;
        } ;
        this._status_display = function () {
            var rows = [] ;
            for (var i in this._status) {
                var status = this._status[i] ;
                var row = [] ;
                if (this._can_manage()) row.push (
                    status.is_locked ? '&nbsp;' :
                    SimpleTable.html.Button ('X', {
                        name:    status.id,
                        classes: 'control-button control-button-small control-button-important status-delete',
                        title:   'delete this status from the list' })) ;
                row.push (
                    status.name ,
                    status.created_time ,
                    status.created_uid ,
                    SimpleTable.html.Button ('search', {
                        name:    status.id,
                        classes: 'status-search',
                        title:   'search for all equipment of this status' })
                ) ;
                rows.push(row) ;
            }
            this._table_status().load(rows) ;

            this._status2_display() ;

            if (this._can_manage()) {
                var input = this._wa().find('input[name="status2"]') ;
                if (this._table_status().selected_object() === null) input.attr('disabled', 'disabled') ;
                else                                                 input.removeAttr('disabled') ;
            }
        } ;
        this._status2_display = function () {
            var rows = [] ;
            var status_name = this._table_status().selected_object() ;
            if (status_name !== null) {
                for (var i in this._status) {
                    var status = this._status[i] ;
                    if (status.name === status_name) {
                        for (var j in status.status2) {
                            var status2 = status.status2[j] ;
                            var row = [] ;
                            if (this._can_manage()) row.push (
                                status2.is_locked ? '&nbsp;' :
                                SimpleTable.html.Button ('X', {
                                    name:    status2.id,
                                    classes: 'control-button control-button-small control-button-important status2-delete',
                                    title:   'delete this sub-status from the list' })) ;
                            row.push (
                                status2.name ,
                                status2.created_time ,
                                status2.created_uid ,
                                SimpleTable.html.Button ('search', {
                                    name:    status2.id,
                                    classes: 'status2-search',
                                    title:   'search for all equipment of this status and sub-status' })
                            ) ;
                            rows.push(row) ;
                        }
                        break ;
                    }
                }
            }
            this._table_status2().load(rows) ;
        } ;
        this._status_delete = function (id) {
            Fwk.ask_yes_no (
                'Confirm Status Deletion' ,
                'Are you sure you want to delete this status and all its sub-statuses from the Dictionary?' ,
                function () {
                    _that._status_action (
                        'Deleting...' ,
                        '../irep/ws/status_delete.php' ,
                        {scope:'status', id: id}
                    ) ;
                }
            ) ;
        } ;
        this._status2_delete = function (id) {
            Fwk.ask_yes_no (
                'Confirm Sub-Status Deletion' ,
                'Are you sure you want to delete this sub-status from the Dictionary?' ,
                function () {
                    _that._status2_action (
                        'Deleting...' ,
                        '../irep/ws/status_delete.php' ,
                        {scope:'status2', id: id}
                    ) ;
                }
            ) ;
        } ;
        this._status_create = function (name) {
            this._status_action (
                'Creating...' ,
                '../irep/ws/status_new.php' ,
                {status: name}
            ) ;
        } ;
        this._status2_create = function (name) {
            this._status2_action (
                'Creating...' ,
                '../irep/ws/status_new.php' ,
                {status: this._table_status().selected_object(), status2: name}
            ) ;
        } ;
        this._status_action = function (name, url, params, dont_display) {
            if (dont_display) {
                Fwk.web_service_GET (url, params, function (data) {
                    _that._status = data.cable_status ;
                }) ;
            } else {
                this._set_updated(name) ;
                Fwk.web_service_GET (url, params, function (data) {
                    _that._status = data.cable_status ;
                    _that._status_display() ;
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                }) ;
            }
        } ;
        this._status2_action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET(url, params, function (data) {
                _that._status = data.cable_status ;
                _that._status2_display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
    }
    Class.define_class (Dictionary_Status, FwkApplication, {}, {}) ;
    
    return Dictionary_Status ;
}) ;

