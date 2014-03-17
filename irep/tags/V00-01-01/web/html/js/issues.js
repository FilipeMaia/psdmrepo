function p_appl_issues () {

    var that = this ;

    this.when_done = null ;

    /* -------------------------------------------------------------------------
     * Data structures and methods to be used/called by external users:
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
     */
    this.name      = 'issues' ;
    this.full_name = 'Issues' ;
    this.context   = '' ;
    this.default_context = 'search' ;

    this.select = function (context,when_done) {
        that.context   = context ;
        this.when_done = when_done ;
        this.init() ;
        switch (this.context) {
            case 'search': this.init_search() ; return ;
        }
    } ;
    this.select_default = function () {
        if (this.context == '') this.context = this.default_context ;
        this.init() ;
        switch (this.context) {
            case 'search': this.init_search() ; return ;
        }
    } ;
    this.if_ready2giveup = function (handler2call) {
        this.init() ;
        handler2call() ;
    } ;

    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */

    this.tabs = null ;

    this.initialized = false ;
    this.init = function () {
        if (this.initialized) return ;
        this.initialized = true ;

    } ;
    
    this.init_search = function () {
        
    } ;
    return this ;
}
var issues = new p_appl_issues() ;
