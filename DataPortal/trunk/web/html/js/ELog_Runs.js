define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/Widget' , 'webfwk/StackOfRows', 'webfwk/FwkApplication', 'webfwk/Fwk' ,
    'portal/ELog_MessageViewer'] ,

function (
    cssloader ,
    Class, Widget, StackOfRows, FwkApplication, Fwk ,
    ELog_MessageViewer) {

    cssloader.load('../portal/css/ELog_Runs.css') ;

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
            this._init() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        this._prev_refresh_sec     = 0 ;
        this._refresh_interval_sec = 10 ;

        this.on_update = function () {
            this._init() ;
            if (this.active && this._refresh_interval_sec) {
                var now_sec = Fwk.now().sec ;
                if (Math.abs(now_sec - this._prev_refresh_sec) > this._refresh_interval_sec) {
                    this._prev_refresh_sec = now_sec ;
                    this._search() ;
                }
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

        this._last_request = null ;
        this._max_seconds = 0 ;

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
            if (!this._run_table_obj) {
                this._viewer_elem = this._body().children('#viewer') ;

                var hdr = [
                    {id: 'begin',        title: 'Begin time',  width:  150} ,
                    {id: 'run',          title: 'Run',         width:   30, align: 'right'} ,
                    {id: 'duration',     title: 'Length',      width:   55, align: 'right', style: 'color:maroon;'} ,
                    {id: 'duration_bar', title: '&nbsp',       width:  200}
                ] ;
                this._run_table_obj = new StackOfRows.StackOfRows (
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


        /**
         * Search for runs in the specified range
         * @returns {undefined}
         */
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
                    _that._set_info('<b>'+data.Runs.length+'</b> runs') ;
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data.Runs, data.MaxSeconds) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;

        /**
         * Display a table of runs
         *
         * @returns {undefined}
         */    
        this._display = function (runs, max_seconds) {
            var table = this._run_table() ;

            // Redisplay the table from scratch if:
            // 
            //   - this is the very first request
            //   - the maxumu run duration has changed (and we need to rescale the run duration bars)
            //
            // Only apply updates to the table if:
            //
            //   - changed status of the last run from the previous request
            //   - new runs appeared since the previous request

            if (!this._last_request) {

                this._last_request = runs ;
                this._max_seconds  = max_seconds ;

                for (var i in this._last_request) {
                    var r = this._last_request[i] ;
                    r.row_id = table.append(this._run2row(r, max_seconds)) ;
                }

            } else if (this._max_seconds !== max_seconds) {

                table.reset() ;

                // TODO: consider rescaling the run duration bars w/o redisplaying
                //       the table from scratch

                this._last_request = runs ;
                this._max_seconds  = max_seconds ;

                for (var i in this._last_request) {
                    var r = this._last_request[i] ;
                    r.row_id = table.append(this._run2row(r, max_seconds)) ;
                }

            } else {

                // -- update the first row if there were any changes in the run status

                var r_last_old = this._last_request[0] ;
                var r_last_new = runs[runs.length - this._last_request.length] ;    // in case if there are more new runs
                if (r_last_old.sec !== r_last_new.sec) {
                    table.update_row(r_last_old.row_id, this._run2row(r_last_new, max_seconds)) ;
                    this._last_request[0] = r_last_new ;
                    this._last_request[0].row_id = r_last_old.row_id ;
                }

                // -- check if more runs should be added to the front

                if (runs.length > this._last_request.length) {

                    // -- insert new runs at the begining of the list

                    for (var i = runs.length - this._last_request.length - 1; i >= 0; i--) {
                        var r = runs[i] ;
                        r.row_id = table.insert_front(this._run2row(r, max_seconds)) ;
                    }

                    // -- and do NOT bother to carry over identifiers of rows from
                    //    the previous request

                    ;

                    this._last_request = runs ;
                    this._max_seconds  = max_seconds ;
                }
            }
        } ;

        this._run2row = function (r, max_seconds) {
            var duration_bar_width = 0 ;
            if (max_seconds) duration_bar_width = Math.floor(185.0 * (r.sec / max_seconds)) ;
            var row = {
                title: {
                    begin: '<b>'+r.ymd+'</b>&nbsp;&nbsp;'+r.hms ,
                    run:   '<div class="m-run">'+r.num+'</div>' ,
                    duration:  r.durat1 ,
                    duration_bar: '<div style="margin-left:15px; width:'+duration_bar_width+'px; background:#5C5C33;">&nbsp;</div>'
                } ,
                body: new ELog_Runs_RunBody(this, r) ,
                block_common_expand: true
            } ;
            return row ;
        } ;
    }
    Class.define_class (ELog_Runs, FwkApplication, {}, {}) ;


    function ELog_Runs_RunBody (parent, run) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        Widget.Widget.call(this) ;

        // -- parameters

        this.parent = parent ;
        this.experiment = parent.experiment ;
        this.access_list = parent.access_list ;

        this.run = run ;

        this._run_url = function () {
            var idx = window.location.href.indexOf('?') ;
            var url = (idx < 0 ? window.location.href : window.location.href.substr(0, idx))+'?exper_id='+this.experiment.id+'&app=elog:search&params=run:'+this.run.num;
            var html = '<a href="'+url+'" target="_blank" title="Click to open in a separate tab, or cut and paste to incorporate into another document as a link."><img src="../portal/img/link.png"></img></a>' ;
            return html ;
        }

        this._cont = function () {
            if (!this._cont_elem) {
                var html =
'<div class="run-cont">' +
'  <div id="ctrl">' +
'    <div style="float:right;">'+this._run_url()+'</div>' +
'    <div style="clear:both;"></div>' +
'  </div>' +
'  <div id="messages"></div>' +
'</div>' ;
                this.container.html(html) ;        
                this._cont_elem = this.container.children('.run-cont') ;
            }
            return this._cont_elem ;
        } ;
        this._ctrl = function () {
            if (!this._ctrl_elem) {
                this._ctrl_elem = this._cont().children('#ctrl') ;
            }
            return this._ctrl_elem ;
        } ;
        this._messages = function () {
            if (!this._messages_elem) {
                this._messages_elem = this._cont().children('#messages') ;
            }
            return this._messages_elem ;
        } ;

        this._viewer = function () {
            if (!this._viewer_obj) {
                this._viewer_obj = new ELog_MessageViewer (this, this._messages(), {}) ;
            }
            return this._viewer_obj ;
        } ;

        // ------------------------------------------------
        // Override event handler defined in thw base class
        // ------------------------------------------------

        this._is_rendered = false ;

        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this._messages().html('Loading...') ;

            var params = {
                exper_id: this.experiment.id ,
                run:      this.run.num ,
                inject_deleted_messages: ''
            } ;
            Fwk.web_service_GET (
                '../logbook/ws/message_search_run.php' ,
                params ,
                function (data) {
                    _that._viewer().load(data.ResultSet.Result) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;
    }
    Class.define_class (ELog_Runs_RunBody, Widget.Widget, {}, {}) ;

    return ELog_Runs ;
}) ;
