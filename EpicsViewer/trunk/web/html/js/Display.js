define ([
    'Class' ,
    'Widget'] ,

function (
    Class ,
    Widget) {

    /**
     * Base class for the application displays
     * 
     * @returns {Display._Display}
     */
    function _Display () {

        Widget.Widget.call(this) ;

        this.active = false ;
        this.activate = function () {
            this.active = true ;
            this.on_activate() ;
        } ;
        this.deactivate = function () {
            this.active = false ;
            this.on_deactivate() ;
        } ;
        this.resize = function () {
            this.on_resize() ;
        } ;

        // These methods are supposed to be implemented by derived classes

        this.on_activate   = function () { console.log('Display::on_activate() NOT IMPLEMENTED') ; } ;
        this.on_deactivate = function () { console.log('Display::on_deactivate() NOT IMPLEMENTED') ; } ;
        this.on_resize     = function () { console.log('Display::on_resize() NOT IMPLEMENTED') ; } ;
    }
    Class.define_class(_Display, Widget.Widget, {}, {}) ;

    return _Display ;
}) ;


