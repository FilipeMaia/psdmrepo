define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../irep/css/Admin_SLACid.css') ;

    /**
     * The application for managing ranges of SLAC IDs of equipment in the Inventory
     *
     * @returns {Admin_SLACid}
     */
    function Admin_SLACid (app_config) {

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

        this._update_ival_sec = 10 ;
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

        this._range = null ;

        this._can_manage = function () { return this._app_config.current_user.is_administrator ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="admin-slacid" >' +

  '<div style="float:left;" class="notes" >' +
    '<p>This application manages a set of "official" SLACid numbers allocated to PCDS/LCLS.' +
    '   Each time a new equipment is being registered in the Inventory Databaseits proposed' +
    '   SLAC ID number will be validated to make sure it falls into one of the ranges' +
    '   known to the application.</p>' +
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
      '<li><a href="#ranges" >Ranges</a></li>' +
    '</ul>' +

    '<div id="ranges" >' +
      '<div class="tab-body" >' +
        '<div class="table-ctrl" >' +
          '<div style="float:left;" class="notes" >' +
            '<p>This page is meant to manage ranges. Note that Ranges can\'t overlap and they shouldn\'t' +
            '   be empty.</p>' +
          '</div>' +
          '<div style="float:right;" class="buttons" >' +
            '<button name="edit"    class="control-button control-button-important" title="edit ranges"                        >EDIT  </button>' +
            '<button name="save"    class="control-button"                          title="save modifications to the database" >SAVE  </button>' +
            '<button name="cancel"  class="control-button"                          title="cancel the editing session"         >CANCEL</button>' +
          '</div>' +
          '<div style="clear:both;" ></div>' +
      '</div>' +
        '<div id="table" ></div>' +
      '</div>' +
    '</div>' +

  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#admin-slacid') ;
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
        this._table = function () {
            if (!this._table_obj) {
                this._table_elem = this._wa().find('#table') ;
                var hdr = [
                    {   name: 'DELETE', sorted: false, hideable: true ,
                        type: { after_sort: function () {
                            _that._table_elem.find('.range-delete').button().click(function () {
                                var id = this.name ;
                                _that._range_delete(id) ;
                            }) ;
                            for (var i in _that._range) {
                                var range = _that._range[i] ;
                                _that._table_elem.find('.description[name="'+range.id+'"]').val(range.description) ;
                            }
                            _that._table_elem.find('.range-search').button().click(function () {
                                var id = this.name ;
                                global_search_equipment_by_slacid_range(id) ;
                            }) ; }}} ,
                    {   name: 'first',       align: 'right', type: SimpleTable.Types.Number} ,
                    {   name: 'last',        align: 'right', type: SimpleTable.Types.Number} ,
                    {   name: '# total',     align: 'right', type: SimpleTable.Types.Number} ,
                    {   name: '# available', align: 'right', type: SimpleTable.Types.Number} ,
                    {   name: 'description', sorted: false,  hideable: true} ,
                    {   name: 'in use',      sorted: false}
                ] ;
                var rows = [] ;
                this._table_obj = new SimpleTable.constructor (
                    this._table_elem ,
                    hdr ,
                    rows ,
                    {default_sort_column: 1} ,
                    Fwk.config_handler('admin', 'table_slacid_ranges')
                ) ;
                this._table_obj.display() ;
            }
            return this._table_obj ;
        } ;
        
        /**
         * Returns and (optionally) set the editing mode. Changes control
         * buttons accordingly.
         *
         * @param {Boolean} editing
         * @returns {Boolean}
         */
        this._range_edit_mode = function (editing) {
            if (this._range_editing === undefined) this._range_editing = false ;
            if (editing !== undefined) {
                if (this._can_manage()) {
                    this._button_update().button(editing ? 'disable' : 'enable') ;
                    this._button_edit  ().button(editing ? 'disable' : 'enable') ;
                    this._button_save  ().button(editing ? 'enable'  : 'disable') ;
                    this._button_cancel().button(editing ? 'enable'  : 'disable');
                    this._range_editing = editing ;
                } else {
                    this._button_update().button('enable') ;
                    this._button_edit  ().button('disable') ;
                    this._button_save  ().button('disable') ;
                    this._button_cancel().button('disable');
                }
            }
            return this._range_editing ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().children('#tabs').tabs() ;

            this._button_update().click(function () { _that._load() ; }) ;
            this._button_edit  ().click(function () { _that._edit() ; }) ;
            this._button_save  ().click(function () { _that._edit_save() ; }) ;
            this._button_cancel().click(function () { _that._edit_cancel() ; }) ;

            this._range_edit_mode(false) ;
            this._load() ;
        } ;

        this._load = function () {
            if (!this._is_initialized) return ;
            if (!this._range_edit_mode())
                this._action (
                    'Loading...' ,
                    '../irep/ws/slacid_get.php' ,
                    {}) ;
        } ;
        this._edit = function () {
            this._range_edit_mode(true) ;
            this._display() ;
        } ;
        this._edit_save = function () {

            // Collect the updates

            var ranges = [] ;
            var firsts       = this._table().get_container().find('.first') ;
            var lasts        = this._table().get_container().find('.last') ;
            var descriptions = this._table().get_container().find('.description') ;

            if (firsts.length !== lasts.length) {
                report_error('Admin_SLACid._edit_save() implementation error, please contact developers') ;
                return ;
            }
            for (var i=0; i < firsts.length; i++) {
                var first       = firsts      [i] ;
                var last        = lasts       [i] ;
                var description = descriptions[i] ;
                if ((first.name !== last.name) || (first.name !== description.name)) {
                    report_error('Admin_SLACid._edit_save(): internal implementation error') ;
                    return ;
                }
                ranges.push({
                    id          : first.name ,
                    first       : first.value ,
                    last        : last.value ,
                    description : description.value
                });
            }
            
            // Save modifications to teh database

            this._set_updated('Saving...') ;
            Fwk.web_service_POST ('../irep/ws/slacid_range_save.php', {'ranges': JSON.stringify(ranges)}, function (data) {
                _that._range = data.range ;
                _that._range_edit_mode(false) ;
                _that._display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._edit_cancel = function () {
            this._range_edit_mode(false) ;
            this._display() ;
        } ;

        this._action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET (url, params, function (data) {
                _that._range = data.range ;
                _that._display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        function count_elements_in_array (obj) {
            var size = 0 ;
            for (var key in obj) size++ ;
            return size ;
        }
        this._display = function () {
            var rows = [] ;
            for (var i in this._range) {
                var range = this._range[i] ;
                if (this._range_edit_mode() && this._can_manage())
                    rows.push ([
                        '' ,
                        SimpleTable.html.TextInput ({classes: 'first', name: ''+range.id, value: range.first, size: 6}) ,
                        SimpleTable.html.TextInput ({classes: 'last',  name: ''+range.id, value: range.last,  size: 6}) ,
                        '' ,
                        '' ,
                        SimpleTable.html.TextArea ({classes: 'description', name: ''+range.id }, 4, 36) ,
                        ''
                    ]) ;
                else
                    rows.push([
                        this._can_manage() ?
                            SimpleTable.html.Button ('X', {
                                name:    range.id ,
                                classes: 'control-button control-button-small control-button-important range-delete' ,
                                title:   'delete this range' }) : ' ' ,
                        range.first ,
                        range.last ,
                        range.last - range.first + 1 ,
                        count_elements_in_array(range.available) ,
                        '<div class="description" ><pre>'+range.description+'</pre></div>' ,
                        SimpleTable.html.Button ('SEARCH', {
                                name:    range.id ,
                                classes: 'control-button control-button-small range-search' ,
                                title:   'search all equipment associated with this range' })
                    ]) ;
            }
            if (this._range_edit_mode() && this._can_manage())
                rows.push ([
                    '' ,
                    SimpleTable.html.TextInput ({classes: 'first', name: '0', value: '0', size: 6}) ,
                    SimpleTable.html.TextInput ({classes: 'last',  name: '0', value: '0', size: 6}) ,
                    '' ,
                    '' ,
                    SimpleTable.html.TextArea ({classes: 'description', name: '0'}, 4, 36) ,
                    ''
                ]) ;
            this._table().load(rows) ;
        } ;
        
        this._range_delete = function (id) {
            Fwk.ask_yes_no (
                'Removing a range' ,
                'Are you sure you want to remove the range? '+
                'Note that all equipment associated with the range will be removed from the database.' ,
                function () {
                    _that._action (
                        'Removing the range...' ,
                        '../irep/ws/slacid_range_delete.php' ,
                        {range_id: id}
                    ) ;
                }
            ) ;
        } ;
    }
    Class.define_class (Admin_SLACid, FwkApplication, {}, {}) ;
    
    return Admin_SLACid ;
}) ;

