/**
 * The application for viewing message attachments in the experimental e-Log
 *
 * @returns {ELog_Attachments}
 */
function ELog_Attachments (experiment, access_list) {

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

    this.on_update = function (sec) {
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

    this.init = function () {
        if (this.is_initialized) return ;
        this.is_initialized = true ;
    } ;
}
define_class (ELog_Attachments, FwkApplication, {}, {});
