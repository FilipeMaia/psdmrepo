define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../irep/css/Equipment_Inventory.css') ;

    /**
     * The application for browsing and searching the Equipment Inventory
     *
     * @returns {Equipment_Inventory}
     */
    function Equipment_Inventory () {

        var _taht = this ;

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

        this.search_equipment_by_slacid_range = function (id) {
            console.log('Equipment_Inventory.search_equipment_by_slacid_range('+id+')  NOT IMPLEMENTED') ;
            return null ;
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
    Class.define_class (Equipment_Inventory, FwkApplication, {}, {}) ;
    
    return Equipment_Inventory ;
}) ;

