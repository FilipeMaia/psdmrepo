define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../irep/css/Dictionary_Location.css') ;

    /**
     * The application for browsing and managing a dictionary of locations
     *
     * @returns {Dictionary_Location}
     */
    function Dictionary_Location (app_config) {

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

        this._location = null ;

        this.locations = function () {
            return this._location ;
        } ;

        this._can_manage = function () { return this._app_config.current_user.has_dict_priv ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dictionary-location" >' +

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
      '<div style="float:left; "><input type="text" size="12" name="location" title="fill in new location name, then press RETURN to save" /></div>' +
      '<div style="float:left; padding-top:4px; color:maroon;" >  &larr; add new location here</div>' +
      '<div style="clear:both; "></div>' +
      '<div id="table-location" class="table" ></div>' +
    '</div>' +

    '<div class="table-cont" style="float:left;" >' +
      '<div style="float:left; "><input type="text" size="12" name="room" title="fill in new room name, then press RETURN to save" /></div>' +
      '<div style="float:left; padding-top:4px; color:maroon;" >  &larr; add new room here</div>' +
      '<div style="clear:both; "></div>' +
      '<div id="table-room" class="table" ></div>' +
    '</div>' +

    '<div style="clear:both;" ></div>' +

  '</div>' +

'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dictionary-location') ;
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
        this._table_location = function () {
            if (!this._table_location_obj) {
                this._table_location_elem = this._wa().find('#table-location') ;
                var hdr =  [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_location_elem.find('.location-delete').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        _that._location_delete(id) ;
                                    }) ;
                            }
                        }
                    }) ;
                hdr.push (
                    {   name: 'location', selectable: true ,
                        type: {
                            select_action : function (location_name) {
                                _that._room_display() ;
                            }
                        }
                    } ,
                    {   name: 'created', hideable: true } ,
                    {   name: 'by user', hideable: true } ,
                    {   name: 'in use',  hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_location_elem.find('.location-search').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        global_search_equipment_by_location(id) ;
                                    }) ;
                            }
                        }
                    }
                ) ;
                var rows = [] ;
                this._table_location_obj = new SimpleTable.constructor (
                    this._table_location_elem ,
                    hdr ,
                    rows ,
                    {selected_col: this._can_manage() ? 1 : 0} ,
                    Fwk.config_handler('dict', 'table_location')
                ) ;
                this._table_location_obj.display() ;
            }
            return this._table_location_obj ;
        } ;
        this._table_room = function () {
            if (!this._table_room_obj) {
                this._table_room_elem = this._wa().find('#table-room') ;
                var hdr =  [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_room_elem.find('.room-delete').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        _that._room_delete(id) ; }) ;
                                _that._table_room_elem.find('.room-search').
                                    button().
                                    click(function () {
                                        var id = this.name ;
                                        global_search_equipment_by_room(id) ;
                                    }) ;
                            }
                        }
                    }) ;
                hdr.push (
                    {   name: 'room',    style: 'font-weight:bold;' } ,
                    {   name: 'created', hideable: true } ,
                    {   name: 'by user', hideable: true } ,
                    {   name: 'in use',  hideable: true, sorted: false }
                ) ;
                var rows = [] ;
                this._table_room_obj = new SimpleTable.constructor (
                    this._table_room_elem ,
                    hdr ,
                    rows ,
                    {} ,
                    Fwk.config_handler('dict', 'table_room')
                ) ;
                this._table_room_obj.display() ;
            }
            return this._table_room_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().find('button[name="update"]').button().click(function () {
                _that._load() ;
            }) ;
            this._wa().find('input[name="location"]').keyup(function (e) {
                var name = $(this).val() ;
                if (name === '') { return ; }
                if (e.keyCode === 13) { _that._location_create(name) ; $(this).val('') ; return ; }
            }) ;
            this._wa().find('input[name="room"]').keyup(function (e) {
                var name = $(this).val() ;
                if (name === '') { return ; }
                if (e.keyCode === 13) { _that._room_create(name) ; $(this).val('') ; return ; }
            }) ;

            this._load() ;
        } ;
        this._load = function () {
            if (!this._is_initialized) return ;
            this._location_action (
                'Loading...' ,
                '../irep/ws/location_get.php' ,
                {}
            ) ;
        } ;
        this._preload = function () {
            var dont_display = true ;
            this._location_action (
                'Loading...' ,
                '../irep/ws/location_get.php' ,
                {} ,
                dont_display
            ) ;
        } ;
        this._location_display = function () {
            var rows = [] ;
            for (var i in this._location) {
                var location = this._location[i] ;
                var row = [] ;
                if (this._can_manage()) row.push (
                    SimpleTable.html.Button ('X', {
                        name:    location.id,
                        classes: 'control-button control-button-small control-button-important location-delete',
                        title:   'delete this location from the list' })) ;
                row.push (
                    location.name ,
                    location.created_time ,
                    location.created_uid ,
                    SimpleTable.html.Button ('search', {
                        name:    location.id,
                        classes: 'location-search',
                        title:   'search for all equipment of this location' })
                ) ;
                rows.push(row) ;
            }
            this._table_location().load(rows) ;

            this._room_display() ;

            if (this._can_manage()) {
                var input = this._wa().find('input[name="room"]') ;
                if (this._table_location().selected_object() === null) input.attr('disabled', 'disabled') ;
                else                                                   input.removeAttr('disabled') ;
            }
        } ;
        this._room_display = function () {
            var rows = [] ;
            var location_name = this._table_location().selected_object() ;
            if (location_name !== null) {
                for (var i in this._location) {
                    var location = this._location[i] ;
                    if (location.name === location_name) {
                        for (var j in location.room) {
                            var room = location.room[j] ;
                            var row = [] ;
                            if (this._can_manage()) row.push (
                                room.is_locked ? '&nbsp;' :
                                SimpleTable.html.Button ('X', {
                                    name:    room.id,
                                    classes: 'control-button control-button-small control-button-important room-delete',
                                    title:   'delete this room from the list' })) ;
                            row.push (
                                room.name ,
                                room.created_time ,
                                room.created_uid ,
                                SimpleTable.html.Button ('search', {
                                    name:    room.id,
                                    classes: 'room-search',
                                    title:   'search for all equipment of this location and room' })
                            ) ;
                            rows.push(row) ;
                        }
                        break ;
                    }
                }
            }
            this._table_room().load(rows) ;
        } ;
        this._location_delete = function (id) {
            Fwk.ask_yes_no (
                'Confirm Location Deletion' ,
                'Are you sure you want to delete this location and all its rooms from the Dictionary?' ,
                function () {
                    _that._location_action (
                        'Deleting...' ,
                        '../irep/ws/location_delete.php' ,
                        {id: id}
                    ) ;
                }
            ) ;
        } ;
        this._room_delete = function (id) {
            Fwk.ask_yes_no (
                'Confirm Room Deletion' ,
                'Are you sure you want to delete this room from the Dictionary?' ,
                function () {
                    _that._room_action (
                        'Deleting...' ,
                        '../irep/ws/room_delete.php' ,
                        {id: id}
                    ) ;
                }
            ) ;
        } ;
        this._location_create = function (name) {
            this._location_action (
                'Creating...' ,
                '../irep/ws/location_new.php' ,
                {name: name}
            ) ;
        } ;
        this._room_create = function (name) {
            this._room_action (
                'Creating...' ,
                '../irep/ws/room_new.php' ,
                {location_name: this._table_location().selected_object(), room_name: name}
            ) ;
        } ;
        this._location_action = function (name, url, params, dont_display) {
            if (dont_display) {
                Fwk.web_service_GET (url, params, function (data) {
                    _that._location = data.location ;
                }) ;
            } else {
                this._set_updated(name) ;
                Fwk.web_service_GET (url, params, function (data) {
                    _that._location = data.location ;
                    _that._location_display() ;
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                }) ;
            }
        } ;
        this._room_action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET(url, params, function (data) {
                _that._location = data.location ;
                _that._room_display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
    }
    Class.define_class (Dictionary_Location, FwkApplication, {}, {}) ;
    
    return Dictionary_Location ;
}) ;

