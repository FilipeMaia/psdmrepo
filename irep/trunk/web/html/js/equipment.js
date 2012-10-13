function p_appl_equipment () {

    var that = this ;

    this.when_done           = null ;
    this.create_form_changed = false ;

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
    this.name      = 'equipment' ;
    this.full_name = 'Equipment' ;
    this.context   = '' ;
    this.default_context = 'inventory' ;

    this.select = function (context,when_done) {
        that.context   = context ;
        this.when_done = when_done ;
        this.init() ;
    } ;
    this.select_default = function () {
        if (this.context == '') this.context = this.default_context ;
        this.init() ;
    } ;
    this.if_ready2giveup = function (handler2call) {
        if((this.context == 'add') && this.create_form_changed) {
            ask_yes_no (
                'Unsaved Data Warning',
                'You are about to leave the page while there are unsaved data in the form. Are you sure?',
                handler2call,
                null) ;
            return ;
        }
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

        var inventory_controls = $('#equipment-inventory-controls') ;
        inventory_controls.find('button[name="search"]').button().click(function () { that.search() ;      }) ;
        inventory_controls.find('button[name="reset"]') .button().click(function () { that.search_reset() ; }) ;

        $('#equipment-add-form').find('input[name="due_time"]').
            each(function () {
                $(this).datepicker().datepicker('option','dateFormat','yy-mm-dd') ; }) ;
        $('#equipment-add-save').
            button().
            button('disable').
            click(function () {
                that.create_project() ; }) ;
        $('#equipment-add-reset').
            button().
            click(function () {
                $('#equipment-add-save').button('disable') ; }) ;
        $('.equipment-add-form-element').
            change(function () {
                that.create_form_changed = true ;
                $('#equipment-add-save').button('enable') ; }) ;            
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

    return this ;
}
var equipment = new p_appl_equipment() ;
