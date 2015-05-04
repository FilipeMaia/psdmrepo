define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Filemanager_Files_USR.css') ;

    /**
     * The application for displaying the detailed info about the data files of the experiment
     *
     * @returns {Filemanager_Files_USR}
     */
    function Filemanager_Files_USR (experiment, access_list) {

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
    Class.define_class (Filemanager_Files_USR, FwkApplication, {}, {}) ;
    
    return Filemanager_Files_USR ;
}) ;

