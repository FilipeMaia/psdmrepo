define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class' ,
    'webfwk/FwkApplication' ,
    'webfwk/Fwk' ,
    'regdb/DRPMgr_Defs'] ,

function (
    cssloader ,
    Class ,
    FwkApplication ,
    Fwk, 
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

        var _DOCUMENT = {
            input:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Press ENTER to save the new value of the parameter in the database.') ,
            add:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Add another experiment to the table for managing its exceptions.') ,
            update:
                DRPMgr_Defs.DOCUMENT_METHOD (
                    'Click this button to update experiment-specific exceptions \n' +
                    'from the database.')
        } ;
        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ;
                if (!this_html) {
                    this_html =
'<div id="drpmgr-exceptions" >' +

    '<div class="info" id="updated" style="float:right;" >Loading...</div> ' +
    '<div style="clear:both;" ></div> ' +

    '<div class="notes" style="float:left;" > ' +
      '<p>This application allows to view and manage experiment-specific exceptions ' +
         'to the General Data Retention Policy. Use the <b>ADD EXPERIMENT</b> button to add ' +
         'a new experiment which is not listed in the table below. Use the <b>REMOVE</b> ' +
         'button to remove an experiment in the corresponidng row of the table.</p> ' +
      '<p>Press *RETURN* after entering a new value of the overriden parameter to save ' +
         'it in the database. Put an empty string as a value of a parameter to remove ' +
         'an exception for that particular parameter. If all exceptions for an experiment ' +
         'are reoved the application will suggest to remove the corresponding experiment ' +
         'from the table of exceptions.<p> ' +

    '</div> ' +

    '<div class="buttons" style="float:left;" > ' +
      '<button name="add"    class="control-button-important" '+_DOCUMENT.add   +' >ADD EXPERIMENT</button> ' +
      '<button name="update" class="control-button" '          +_DOCUMENT.update+' ><img src="../webfwk/img/Update.png" /></button> ' +
    '</div> ' +
    '<div style="clear:both;" ></div> ' +

    '<div id="table > ' +
    '</div> ' +

'</div>' ;
                }
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('#drpmgr-exceptions') ;
            }
            return this._wa_elem ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // Proceed to the first loading
            this._load() ;
        } ;
        this._load = function () {

            if (!this._is_initialized) return ;

        } ;
        this._display = function () {
        } ;
    }
    Class.define_class (DRPMgr_Exceptions, FwkApplication, {}, {}) ;
    
    return DRPMgr_Exceptions ;
}) ;