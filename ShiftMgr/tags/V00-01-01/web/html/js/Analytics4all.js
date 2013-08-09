function Analytics4all () {
    FwkDispatcherBase.call(this) ;
}
define_class (Analytics4all, FwkDispatcherBase, {}, {

    on_activate : function() {
        this.on_update() ;
    } ,

    on_deactivate : function() {
        this.init() ;
    } ,

    on_update : function (sec) {
        this.init() ;
        if (this.active) ;
    } ,

    is_initialized : false ,

    init : function () {
        if (this.is_initialized) return ;
        var that = this ;
    }
});
