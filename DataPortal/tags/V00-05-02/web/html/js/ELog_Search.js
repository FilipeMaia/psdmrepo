/**
 * The application for searching messages in the experimental e-Log
 *
 * @returns {ELog_Search}
 */
function ELog_Search (experiment, access_list) {

    var that = this ;

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
        this.init() ;
    } ;

    this.on_update = function () {
        if (this.active) {
            this.init() ;
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

    this.is_initialized = false ;

    this.wa = null ;

    this.init = function () {

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        this.container.html('<div id="elog-search"></div>') ;
        this.wa = this.container.find('div#elog-search') ;

        if (!this.access_list.elog.read_messages) {
            this.wa.html(this.access_list.no_page_access_html) ;
            return ;
        }

        var html =
'' ;
        this.wa.html(html) ;
    } ;
}
define_class (ELog_Search, FwkApplication, {}, {});
