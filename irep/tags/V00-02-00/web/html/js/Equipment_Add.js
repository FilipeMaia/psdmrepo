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
    function Equipment_Add () {

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

        // --------------------
        // Own data and methods
        // --------------------

        this.is_initialized = false ;

        this.init = function () {
            if (this.is_initialized) return ;
            this.is_initialized = true ;
        } ;
    }
    Class.define_class (Equipment_Add, FwkApplication, {}, {}) ;
    
    return Equipment_Add ;
}) ;

