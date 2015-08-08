define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../irep/css/Equipment_Add.css') ;

    /**
     * The application for adding new equipment to the Equipment Inventory
     *
     * @returns {Equipment_Add}
     */
    function Equipment_Add (app_config) {

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

        this.on_update = function () {
            if (this.active) {
                this._init() ;
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

        this._can_edit_inventory = function () { return this._app_config.current_user.can_edit_inventory ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var required_field = '<span style="color:red ; font-size:110% ; font-weight:bold ;"> * </span>' ;
                var html = html ||
'<div id="equipment-add" >' +
  '<div style="margin-bottom:20px ; border-bottom:1px dashed #c0c0c0 ;"> ' +
    '<div style="float:left ;">' +
      '<div style="margin-bottom:10px ; width:480px ;"> ' +
        'When making a clone of an existing equipment record make sure the  <b>serial number</b>, <b>Property Control</b> number, ' +
        'and a <b>SLAC ID</b> of the new equipment differ from the original one. All other attributes of the original equipment ' +
        'will be copied into the new one. The copied equipment will all be put into the <b>Unknown</b> state. ' +
      '</div> ' +
      '<form id="form"> ' +
        '<table><tbody> ' +
          '<tr><td><b>Manufacturer:' + required_field + '</b></td> ' +
              '<td colspan="3"><select name="manufacturer" class="form-element" ></select> ' +
                  '<span class="form-element-info"></span></td> ' +
          '</tr> ' +
          '<tr><td><b>Model:' + required_field + '</b></td> ' +
              '<td colspan="3"><select name="model" class="form-element" ></select> ' +
                  '<span class="form-element-info"></span></td> ' +
          '</tr> ' +
          '<tr><td><b>Serial number:</b></td> ' +
              '<td><input type="text" name="serial" class="form-element" size="20" style="padding:2px ;" value="" /></td> ' +
          '</tr> ' +
          '<tr><td><b>Property Control #:</b></td> ' +
              '<td><input type="text" name="pc" size="20" style="padding:2px ;" value="" /></td></tr> ' +
          '<tr><td><b>SLAC ID:' + required_field + '</b></td> ' +
              '<td colspan="3"><input type="text" name="slacid" class="form-element" size="20" style="padding:2px ;" value="" /> ' +
                  '<span class="form-element-info"></span></td> ' +
          '</tr> ' +
          '<tr><td><b>Location:</b></td> ' +
              '<td><select name="location" class="form-element" ></select></td> ' +
              '<td><b>Room:</b></td> ' +
              '<td><select name="room" class="form-element" ></select></td> ' +
          '</tr> ' +
          '<tr><td colspan="2">&nbsp;</td> ' +
              '<td><b>Rack:</b></td> ' +
              '<td><input type="text" name="rack" class="form-element" size="20" value="" /></td> ' +
          '</tr> ' +
          '<tr><td colspan="2">&nbsp;</td> ' +
              '<td><b>Elevation:</b></td> ' +
              '<td><input type="text" name="elevation" class="form-element" size="20" value="" /></td> ' +
          '</tr> ' +
          '<tr><td><b>Custodian:</b></td> ' +
              '<td colspan="3"><input type="text" name="custodian" size="20" style="padding:2px ;" value="" /> ' +
              '( known custodians: <select name="custodian"></select> )</td> ' +
          '</tr> ' +
          '<tr><td><b>Notes: </b></td> ' +
              '<td colspan="3"><textarea cols=54 rows=4 name="description" style="padding:4px ;" title="Here be arbitrary notes for this equipment"></textarea></td> ' +
          '</tr> ' +
        '</tbody></table> ' +
      '</form> ' +
    '</div> ' +
    '<div style="float:left ; padding:5px ;"> ' +
      '<div> ' +
        '<button id="save">Create</button> ' +
        '<button id="reset">Reset Form</button> ' +
      '</div> ' +
      '<div style="margin-top:5px ;" id="info" >&nbsp;</div> ' +
    '</div> ' +
    '<div style="clear:both ;"></div> ' +
  '</div> ' +
  required_field + ' required field' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#equipment-add') ;
            }
            return this._wa_elem ;
        } ;
        this._button_save = function () {
            if (!this._button_save_obj) {
                this._button_save_obj = this._wa().find('button#save').button() ;
            }
            return this._button_save_obj ;
        } ;
        this._button_reset = function () {
            if (!this._button_reset_obj) {
                this._button_reset_obj = this._wa().find('button#reset').button() ;
            }
            return this._button_reset_obj ;
        } ;
        this._form = function () {
            if (!this._form_elem) {
                this._form_elem = this._wa().find('form#form') ;
            }
            return this._form_elem ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this._can_edit_inventory()) {
                this._wa(this._app_config.no_page_access_html) ;
                return ;
            }
            this._button_save().button('disable').click(function () {
                _that._save() ;
            }) ;
            this._button_reset().click(function () {
                _that._button_save().button('disable') ;
                _that._form_init() ;
            }) ;
            this._wa().find('.form-element').change(function () {
                _that._form_validate() ;
            }) ;
            
            this._form_init() ;
        } ;
        this._form_init = function () {

            var manufacturer_elem    = this._form().find('select[name="manufacturer"]') ;
            var model_elem           = this._form().find('select[name="model"]') ;
            var serial_elem          = this._form().find('input[name="serial"]') ;
            var pc_elem              = this._form().find('input[name="pc"]') ;
            var slacid_elem          = this._form().find('input[name="slacid"]') ;
            var location_elem        = this._form().find('select[name="location"]') ;
            var room_elem            = this._form().find('select[name="room"]') ;
            var rack_elem            = this._form().find('input[name="rack"]') ;
            var elevation_elem       = this._form().find('input[name="elevation"]') ;
            var custodian_elem       = this._form().find('input[name="custodian"]') ;
            var custodian_known_elem = this._form().find('select[name="custodian"]') ;
            var description_elem     = this._form().find('textarea[name="description"]') ;

            model_elem.attr ('disabled', 'disabled') ;
            serial_elem.attr('disabled', 'disabled') ;

            Fwk.web_service_GET ('../irep/ws/equipment_search_options.php', {}, function (data) {

                var html = '<option value="0"></option>' ;
                for (var i in data.option.manufacturer) {
                    var manufacturer = data.option.manufacturer[i] ;
                    html += '<option value="'+manufacturer.id+'">'+manufacturer.name+'</option>' ;
                }
                manufacturer_elem.html(html) ;
                manufacturer_elem.change(function () {
                    var manufacturer_id = this.value ;
                    if (!manufacturer_id) {
                        model_elem.val(0) ;
                        model_elem.attr ('disabled', 'disabled') ;
                        serial_elem.val('') ;
                        serial_elem.attr('disabled', 'disabled') ;
                        return ;
                    }
                    for (var i in data.option.manufacturer) {
                        var manufacturer = data.option.manufacturer[i] ;
                        if (manufacturer_id == manufacturer.id) {
                            var html = '<option value="0"></option>' ;
                            for (var j in manufacturer.model) {
                                var model = manufacturer.model[j] ;
                                html += '<option value="'+model.id+'">'+model.name+'</option>' ;
                            }
                            model_elem.html(html) ;
                            model_elem.removeAttr('disabled') ;
                            break ;
                        }
                    }
                }) ;

                var html = '<option value="0"></option>' ;
                model_elem.html(html) ;
                model_elem.change(function () {
                    var model_id = this.value ;
                    if (!model_id) {
                        serial_elem.val('') ;
                        serial_elem.attr('disabled', 'disabled') ;
                    } else {
                        serial_elem.removeAttr('disabled') ;
                    }
                }) ;

                serial_elem.val('') ;
                pc_elem.val('') ;
                slacid_elem.val('') ;

                var html  = '<option value="0"></option>' ;
                for (var i in data.option.location) {
                    var location = data.option.location[i] ;
                    html += '<option value="'+location.id+'">'+location.name+'</option>' ;
                }
                location_elem.html(html) ;

                var html2 = '<option value="0"></option>' ;
                room_elem.html(html2) ;

                location_elem.change (function () {
                    var selected_location_id = $(this).val() ;
                    var html2 = '<option value="0"></option>' ;
                    for (var i in data.option.location) {
                        var location = data.option.location[i] ;
                        if (selected_location_id == location.id) {
                            for (var j in location.room) {
                                var room = location.room[j] ;
                                html2 += '<option value="'+room.id+'">'+room.name+'</option>' ;
                            }
                        }
                    }
                    room_elem.html(html2) ;
                }) ;
                rack_elem.val('') ;
                elevation_elem.val('') ;

                custodian_elem.val(_that._app_config.current_user.uid) ;
                var html = '<option value="'+_that._app_config.current_user.uid+'"></option>' ;
                for (var i in data.option.custodian) {
                    var custodian = data.option.custodian[i] ;
                    html += '<option value="'+custodian+'">'+custodian+'</option>' ;
                }
                custodian_known_elem.html(html) ;
                custodian_known_elem.change(function () {
                    var custodian = this.value ;
                    custodian_elem.val(custodian) ;
                }) ;

                description_elem.val('') ;

                _that._form_validate() ;
            }) ;
        } ;
        this._form_validate = function () {

            var manufacturer_elem = this._form().find('select[name="manufacturer"]') ;
            var model_elem        = this._form().find('select[name="model"]') ;
            var slacid_elem       = this._form().find('input[name="slacid"]') ;
            var slacid_info_elem  = slacid_elem.next ('span.form-element-info') ;

            var manufacturer_id = parseInt(manufacturer_elem.val()) ;
            var model_id        = parseInt(model_elem.val()) ;
            var slacid          = parseInt(slacid_elem.val()) ;

            manufacturer_elem.next('span.form-element-info').html(manufacturer_id ? '' : '&larr; please select') ; 
            model_elem.next       ('span.form-element-info').html(model_id        ? '' : '&larr; please select') ; 
            if (!slacid) {
                slacid_info_elem.html('&larr; please fill and press ENTER to validate') ;
                this._button_save().button (
                    manufacturer_id && model_id && slacid ?
                    'enable' :
                    'disable'
                ) ;
            } else {
                Fwk.web_service_GET (
                    '../irep/ws/slacid_validate.php' ,
                    {slacid: slacid} ,
                    function (data) {
                        if (data.slacid.in_use) {
                            slacid_info_elem.html('&larr; the identifier is already in use') ;
                            slacid = 0 ;
                        } else {
                            if (data.slacid.is_valid ) {
                                slacid_info_elem.html('') ;
                            } else {
                                slacid_info_elem.html('&larr; the identifier is not allowed, please correct') ;
                                slacid = 0 ;
                            }
                        }
                        _that._button_save().button (
                            manufacturer_id && model_id && slacid ?
                            'enable' :
                            'disable'
                        ) ;
                    } ,
                    function () {
                        slacid_info_elem.html('&larr; the validation failed, but you may still proceed with saving results') ;
                        _that._button_save().button (
                            manufacturer_id && model_id && slacid ?
                            'enable' :
                            'disable'
                        ) ;
                    }
                ) ;
            }
        } ;
        this._save = function () {
            this._button_save().button('disable') ;
            Fwk.web_service_POST (
                '../irep/ws/equipment_new.php' ,
                {
                    model_id:    this._form().find('select[name="model"]').val() ,
                    serial:      this._form().find('input[name="serial"]').val() ,
                    pc:          this._form().find('input[name="pc"]').val() ,
                    slacid:      this._form().find('input[name="slacid"]').val() ,
                    location_id: this._form().find('select[name="location"]').val() ,
                    room_id:     this._form().find('select[name="room"]').val() ,
                    rack:        this._form().find('input[name="rack"]').val() ,
                    elevation:   this._form().find('input[name="elevation"]').val() ,
                    custodian:   this._form().find('input[name="custodian"]').val() ,
                    description: this._form().find('textarea[name="description"]').val()
                } ,
                function (data) {
                    _that._form_init() ;
                } ,
                function (msg) {
                    _that._button_save().button('enable') ;
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;
    } ;
    Class.define_class (Equipment_Add, FwkApplication, {}, {}) ;
    
    return Equipment_Add ;
}) ;

