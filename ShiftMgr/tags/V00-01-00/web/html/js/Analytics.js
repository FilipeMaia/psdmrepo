function Analytics (instr_name) {

    FwkDispatcherBase.call(this) ;

    this.instr_name = instr_name ;
}
define_class (Analytics, FwkDispatcherBase, {}, {

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
