/**
 * The application for making shift reports accross all instrument
 *
 * @returns {Reports4all}
 */
function Reports4all () {

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

    // --------------------
    // Own data and methods
    // --------------------

    this.is_initialized = false ;

    this.init = function () {
        if (this.is_initialized) return ;
        this.is_initialized = true ;
    } ;
}
define_class (Reports4all, FwkApplication, {}, {});
