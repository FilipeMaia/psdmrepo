define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../shiftmgr/css/shiftmgr.css') ;

   /**
     * The application for managing rules for creating shifts
     *
     * @returns {Rules}
     */
    function Rules () {

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
            
            this.container.html (
'Various configuration parameters and rules, such: when shifts usuall begin, create shift placeholders automatically or not'
            ) ;
        } ;
    }
    Class.define_class (Rules, FwkApplication, {}, {}) ;

    return Rules ;
}) ;
