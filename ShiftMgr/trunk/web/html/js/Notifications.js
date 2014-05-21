define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../shiftmgr/css/shiftmgr.css') ;

    /**
     * The application for managing e-mail notifications on changes in the database
     *
     * @returns {Notifications}
     */
    function Notifications (instr_name) {

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------
        // Parameters of the object
        // ------------------------

        this.instr_name = instr_name || '' ;

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
            
            this.container.html (
'View and manage push notifications: who will get an event and what kind of events (new shift created, data updated, etc.)'
            ) ;
        } ;
    }
    Class.define_class (Notifications, FwkApplication, {}, {}) ;

    return Notifications ;
}) ;
