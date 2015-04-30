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
    Fwk ,
    DRPMgr_Defs) {

    cssloader.load('../regdb/css/DRPMgr_Totals.css') ;

    /**
     * The application for displaying and managin experiment-specific
     * policy exceptions.
     *
     * @returns {DRPMgr_Totals}
     */
    function DRPMgr_Totals (app_config) {

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
    Class.define_class (DRPMgr_Totals, FwkApplication, {}, {}) ;
    
    return DRPMgr_Totals ;
}) ;