define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

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

        this._update_ival_sec = 1000 ;
        this._prev_update_sec = Fwk.now().sec ;

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

        this._can_manage = function () { return this._app_config.current_user.is_administrator ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dictionary-status" >' +

  '<div style="float:left;" class="notes" >' +
    '<p>This application manages a dictionary of status coded assigned to the equipment.' +
    '   The codes are organized into a hierarchy of the mandatory <b>status</b> codes' +
    '   and  optional <b>sub-status</b> codes. To see sub-status coded available for a particular  Note, that some codes are predefined, and they can\t be' +
    '   removed by the application.</p>' +
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
                this._wa_elem = this.container.children('#dictionary-status') ;
            }
            return this._wa_elem ;
        } ;

        this._init = function () {
            if (this._is_initialized) return ;
            this._is_initialized = true ;
        } ;
    }
    Class.define_class (Dictionary_Status, FwkApplication, {}, {}) ;
    
    return Dictionary_Status ;
}) ;

