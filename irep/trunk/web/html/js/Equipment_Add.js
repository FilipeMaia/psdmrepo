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
   + required_field + ' required field' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#equipment-add') ;
            }
            return this._wa_elem ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this._can_edit_inventory()) {
                this._wa(this._app_config.no_page_access_html) ;
                return ;
            }
            this._wa().find('button#save').
                button().
                button('disable').
                click(function () {
                    that.equipment_save() ; }) ;

            $('#equipment-add-reset').
                button().
                click(function () {
                    that.init_add() ;
                    $('#equipment-add-save').button('disable') ; }) ;

            $('.equipment-add-form-element').change(function () {
                that.add_form_validate() ;
            }) ;
        } ;
    }
    Class.define_class (Equipment_Add, FwkApplication, {}, {}) ;
    
    return Equipment_Add ;
}) ;

