/**
 * The analytics application for an instrument
 *
 * @param string instr_name
 * @returns {Analytics}
 */
function Analytics (instr_name) {

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    FwkApplication.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.instr_name = instr_name ;

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

    // --------------------
    // Own data and methods
    // --------------------

    this.is_initialized = false ;

    this.init = function () {
        if (this.is_initialized) return ;
        this.is_initialized = true ;
    } ;
}
define_class (Analytics, FwkApplication, {}, {}) ;
