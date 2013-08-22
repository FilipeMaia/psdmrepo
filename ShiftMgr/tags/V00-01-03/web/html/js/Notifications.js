/**
 * The application for managing e-mail notifications on changes in the database
 *
 * @returns {Notifications}
 */
function Notifications () {

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
        this.init() ;
        if (this.active) {
            ;
        }
    } ;

    // --------------------
    // Own data and methods
    // --------------------

    this.is_initialized = false ;

    this.init = function () {
        if (this.is_initialized) return ;
        this.is_initialized = true ;
    } ;
}
define_class (Notifications, FwkApplication, {}, {});
