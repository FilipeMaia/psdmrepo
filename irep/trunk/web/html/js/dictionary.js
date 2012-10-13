function p_appl_dictionary () {

    var that = this ;

    this.when_done = null ;

    /* -------------------------------------------------------------------------
     *   Data structures and methods to be used/called by external users
     *
     *   select(context, when_done)
     *      select a specific context
     *
     *   select_default()
     *      select default context as implemented in the object
     *
     *   if_ready2giveup(handler2call)
     *      check if the object's state allows to be released, and if so then
     *      call the specified function. Otherwise just ignore it. Normally
     *      this operation is used as a safeguard preventing releasing
     *      an interface focus if there is on-going unfinished editing
     *      within one of the interfaces associated with the object.
     *
     * -------------------------------------------------------------------------
     */
    this.name      = 'dictionary' ;
    this.full_name = 'Dictionary' ;
    this.context   = '' ;
    this.default_context = 'manufacturers' ;

    this.select = function (context) {
        that.context = context ;
        this.init() ;
    } ;
    this.select_default = function () {
        this.init() ;
        if (this.context == '') this.context = this.default_context ;
    } ;
    this.if_ready2giveup = function (handler2call) {
        this.init() ;
        handler2call() ;
    } ;

    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
    this.initialized = false ;
    this.init = function () {
        if (this.initialized) return ;
        this.initialized = true ;
        this.init_manufacturers() ;
        this.init_locations() ;
    } ;
    this.can_manage = function () {
        return global_current_user.has_dict_priv ;
    } ;
    this.web_service_GET = function (url, params, data_handler) {
        this.init() ;
        var jqXHR = $.get(url, params, function (data) {
            var result = eval(data) ;
            if (result.status != 'success') { report_error(result.message, null) ; return ; }
            data_handler(result) ;
        },
        'JSON').error(function () {
            report_error('update failed because of: '+jqXHR.statusText) ;
        }) ;
    } ;

    // ------------------------
    // MANUFACTURERS and MODELS
    // ------------------------

    this.init_manufacturers = function () {
        
    } ;

    // ---------
    // LOCATIONS
    // ---------

    this.init_locations = function () {
        
    } ;
    return this;
}
var dict = new p_appl_dictionary();
