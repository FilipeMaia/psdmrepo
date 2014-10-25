define ([] ,

function () {

    /**
     * @brief The base class for user-defined applications.
     *
     * @returns {FwkApplication}
     */
    function FwkApplication () {

        this.container = null ;
        this.set_container = function (container) {
            if (this.container === null) this.container = container ;
        } ;

        this.application_name = "" ;
        this.context1_name    = "" ;
        this.context2_name    = "" ;
        this.set_path = function (application_name, context1_name, context2_name) {
            this.application_name = application_name ;
            this.context1_name    = context1_name ;
            this.context2_name    = context2_name ;
        } ;

        this.active = false ;
        this.activate = function (container) {
            this.set_container(container) ;
            this.active = true ;
            this.on_activate() ;
        } ;
        this.deactivate = function (container) {
            this.set_container(container) ;
            this.active = false ;
            this.on_deactivate() ;
        } ;
        this.update = function (container) {
            this.set_container(container) ;
            this.on_update() ;
        } ;

        // These methods are supposed to be implemented by derived classes

        this.on_activate   = function () { console.log('FwkApplication::on_activate() NOT IMPLEMENTED') ; } ;
        this.on_deactivate = function () { console.log('FwkApplication::on_deactivate() NOT IMPLEMENTED') ; } ;
        this.on_update     = function () { console.log('FwkApplication::on_update() NOT IMPLEMENTED') ; } ;
    }

    return FwkApplication ;
}) ;
