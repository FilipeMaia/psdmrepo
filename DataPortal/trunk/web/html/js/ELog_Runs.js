/**
 * The application for searching messages within runs in the experimental e-Log
 *
 * @returns {ELog_Runs}
 */
function ELog_Runs (experiment, access_list) {

    var _that = this ;

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
        this._init() ;
    } ;

    this.on_update = function () {
        if (this.active) {
            this._init() ;
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

    /**
     * Initialize the work area and return an element. Use an html document from 
     * an optional parameter if the one passed into the function. Otherwise use
     * the standard initialization.
     *
     *   ()     - the standard initialization
     *   (html) - initialization with the specified content
     *
     * NOTE: if the parameter is present the method would always go
     * for the forced (re-)initialization regardless of any prior
     * initialization attempts.
     * 
     * @param   {string} the new content
     * @returns {string} the JQuery element
     */
    this._wa = function (html) {
        if (this._wa_elem) {
            if (html !== undefined) {
                this._wa_elem.html(html) ;
            }
        } else {
            this.container.html('<div id="elog-runs"></div>') ;
            this._wa_elem = this.container.find('div#elog-runs') ;
            if (html === undefined) {
                html =
'<div id="ctrl">' +
'  <div class="search-dialog">' +
'    <div  class="group" >' +
'      <span class="label">Range of Runs:</span>' +
'      <input class="update-trigger" type="text" name="runs2search" value="" size=24 title=' +
'"Enter a run number or a range of runs where to look for messages. \n' +
'For a single run put its number. \n' +
'For a range the correct syntax is: 12-35"' +
'      />' +
'    </div>' +
'    <div class="group">' +
'      <span class="label">Include:</span>' +
'      <div><input class="update-trigger" type="checkbox" name="search_in_deleted" checked="checked" /> deleted messages</div>' +
'    </div>' +
'    <div class="buttons" style="float:left;" >' +
'      <button class="control-button" name="search" title="search and display results">Search</button>' +
'      <button class="control-button" name="reset"  title="reset the form">Reset</button>' +
'    </div>' +
'    <div style="clear:both;"></div>' +
'  </div>' +
'</div>' +
'<div id="body">' +
'  <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div id="viewer" class="elog-msg-viewer"></div>' +
'</div>' ;
            }
            this._wa_elem.html(html) ;
        }
        return this._wa_elem ;
    } ;

    this._ctrl = function () {
        if (!this._ctrl_elem) {
            this._ctrl_elem = this._wa().children('#ctrl') ;
        }
        return this._ctrl_elem ;
    } ;

    /**
     * Get or set a value of the control, depending on a presense of
     * the optional parameter to the method:
     *
     *   ()     - get the current value
     *   (val)  - set the new value and return it back
     * 
     * @param   {string} the new value
     * @returns {string} the updated value
     */
    this._runs2search = function (range) {
        if (!this._runs2search_elem) {
            this._runs2search_elem = this._ctrl().find('input[name="runs2search"]') ;
        }
        if (range !== undefined) this._runs2search_elem.val(range) ;
        return this._runs2search_elem.val() ;
    } ;

    /**
     * Get or set a value of the control, depending on a presense of
     * the optional parameter to the method:
     *
     *   ()     - get the current value
     *   (val)  - set the new value and return it back
     * 
     * @param   {string} the new value
     * @returns {string} the updated value
     */
    this._search_in_deleted = function (flag) {
        if (!this._search_in_deleted_elem) {
            this._search_in_deleted_elem = this._ctrl().find('input[name="search_in_deleted"]') ;
        }
        if (flag !== undefined) {
            if (flag) this._search_in_deleted_elem.attr('checked', 'checked') ;
            else      this._search_in_deleted_elem.removeAttr('checked') ;
        }
        return this._search_in_deleted_elem.attr('checked') ? true : false ;
    } ;

    this._body = function () {
        if (!this._body_elem) {
            this._body_elem = this._wa().children('#body') ;
        }
        return this._body_elem ;
    } ;
    this._set_info = function (html) {
        if (!this._info_elem) {
            this._info_elem = this._body().children('#info') ;
        }
        this._info_elem.html(html) ;
    } ;
    this._set_updated = function (html) {
        if (!this._updated_elem) {
            this._updated_elem = this._body().children('#updated') ;
        }
        this._updated_elem.html(html) ;
    } ;

    this._run_table  = function () {
        if (!this._viewer_obj) {
            this._viewer_elem = this._body().children('#viewer') ;

            var hdr = [
                {id: 'begin',        title: 'Begin time',  width:  150} ,
                {id: 'run',          title: 'Run',         width:   30, align: 'right'} ,
                {id: 'duration',     title: 'Length',      width:   55, align: 'right', style: 'color:maroon;'} ,
                {id: 'duration_bar', title: '&nbsp',       width:  400}
            ] ;
            this._run_table_obj = new StackOfRows (
                hdr ,
                [] ,
                {
                    theme: 'stack-theme-mustard'
                }
            ) ;
            this._run_table_obj.display(this._viewer_elem) ;
        }
        return this._run_table_obj ;
    }

    // ---------------------------------------
    //   BEGIN INITIALIZING THE UI FROM HERE
    // ---------------------------------------

    this._is_initialized = false ;

    this._init = function () {

        if (this._is_initialized) return ;
        this._is_initialized = true ;

        // -- no further initialization beyond this point if not authorized

        if (!this.access_list.elog.read_messages) {
            this._wa(this.access_list.no_page_access_html) ;
            return ;
        }

        // -- set up event handlers

        this._ctrl().find('input.update-trigger').change(function () {
            _that._search() ; }) ;

        this._ctrl().find('button.control-button').button().click(function () {
            switch (this.name) {
                case 'search':
                    _that._search() ;
                    break ;
                case 'reset' :
                    _that._runs2search('') ;
                    _that._search_in_deleted(true) ;
                    _that._search() ;
                    break ; }}) ;
  
        // -- initiate the loading

        this._search() ;

    } ;

    this._last_request = [] ;
    this._max_seconds = 0 ;

    this._search = function () {
        this._set_updated('Loading...') ;
        var params = {
            exper_id:      this.experiment.id ,
            range_of_runs: this._runs2search()
        } ;
        Fwk.web_service_GET (
            '../logbook/ws/RequestAllRuns.php' ,
            params ,
            function (data) {
                _that._last_request = data.Runs ;
                _that._max_seconds  = data.MaxSeconds ;
                _that._set_info('<b>'+_that._last_request.length+'</b> runs') ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                _that._display() ;
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
            }
        ) ;
    } ;
    
    this._display = function () {
        var table = this._run_table() ;
        for (var i in this._last_request) {
            var r = this._last_request[i] ;
            
            var row = {
                title: {
                    begin: '<b>'+r.ymd+'</b>&nbsp;&nbsp;'+r.hms ,
                    run:   '<div class="m-run">'+r.num+'</div>' ,
                    duration:  r.durat1 ,
                    duration_bar: ''
                } ,
                body: 'Here be a list of messages, ...soon' ,       // new ELog_Runs_RunBody(this, r) ;
                block_common_expand: true
            } ;
            table.append(row) ;
        }
    } ;
}
define_class (ELog_Runs, FwkApplication, {}, {});


function ELog_RunViewer (parent, cont, options) {

    // -- parameters

    this.parent = parent ;
    this.experiment = parent.experiment ;
    this.access_list = parent.access_list ;
    this.cont = cont ;

    // Must be the last call. Otherwise the widget won't be able to see
    // functon 'render()' defined above in this code.

    this.display(this.cont) ;
}
define_class (ELog_MessageViewer, Widget, {}, {}) ;
