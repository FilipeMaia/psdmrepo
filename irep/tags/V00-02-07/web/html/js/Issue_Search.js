define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../irep/css/Issue_Search.css') ;

    /**
     * The application for browsing and searching of issues
     *
     * @returns {Issue_Search}
     */
    function Issue_Search () {

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
    Class.define_class (Issue_Search, FwkApplication, {}, {}) ;
    
    return Issue_Search ;
}) ;

