define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk' ,
    'portal/ELog_MessageViewer'] ,

function (
    cssloader, Class, FwkApplication, Fwk ,
    ELog_MessageViewer) {

    cssloader.load('../portal/css/ELog_Live.css') ;

    /**
     * The application for displaying & managing the live stream of e-log messages & runs
     *
     * @returns {ELog_Live}
     */
    function ELog_Live (experiment, access_list) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            if (this._is_initialized) {
                this._update_viewer(function () {
                    _that._enable_autoexpander(true) ;
                }) ;
            } else {
                this._init(function () {
                    _that._load_viewer() ;
                    _that._enable_autoexpander(true) ;
                }) ;
            }
        } ;

        this.on_deactivate = function() {
            this._enable_autoexpander(false) ;
        } ;

        this._prev_refresh_sec     = 0 ;
        this._refresh_interval_sec = 0 ; // no automatic updates by default

        this.on_update = function () {
            if (this.active && this._refresh_interval_sec) {
                var now_sec = Fwk.now().sec ;
                if (Math.abs(now_sec - this._prev_refresh_sec) > this._refresh_interval_sec) {
                    this._prev_refresh_sec = now_sec ;
                    this._update_viewer() ;
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

        this._is_initialized = false ;

        this._wa      = null ;

        this._loading = null ;
        this._info    = null ;
        this._updated = null ;

        this._viewer = null ;

        this._init = function (when_done) {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this.container.html('<div id="elog-live"></div>') ;
            this._wa = this.container.find('div#elog-live') ;

            if (!this.access_list.elog.read_messages) {
                this._wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div id="ctrl">' +
'  <div style="float:left;" >' +
'    <span>Last messages</span>' +
'    <select class="refresh-action" name="num_messages" title="specify how many messages to load" >' +
'      <option value="100" >        100 </option>' +
'      <option value="12h" >      shift </option>' +
'      <option value="24h" >     24 hrs </option>' +
'      <option value="7d"  >     7 days </option>' +
'      <option value=""    > everything </option>' +
'    </select>' +
'  </div>' + (this.experiment.is_facility ? '' :
'  <div style="float:left;" >' +
'    <span>Include runs</span>' +
'    <input class="refresh-action" name="include_runs" type="checkbox" checked="checked" title="search for runs as well" />' +
'  </div>') +
'  <div style="float:left;" >' +
'    <span>Show deleted</span>' +
'    <input class="refresh-action" name="show_deleted" type="checkbox" checked="checked" title="display deleted messages" />' +
'  </div>' +
'  <div style="float: right;" >' +
'    <button class="control-button" name="refresh" title="click to refresh the display">Refresh</button>' +
'  </div>' +
'  <div style="float: right;" >' +
'    <span>Auto-refresh</span>' +
'    <select name="refresh_interval" title="show frequently to check for updates" >' +
'      <option value="0"                     >   Off</option>' +
'      <option value="2" selected="selected" > 2 sec</option>' +
'      <option value="5"                     > 5 sec</option>' +
'      <option value="10"                    >10 sec</option>' +
'    </select>' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div id="body">' +
'  <div class="info" id="info" style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div id="viewer" class="elog-msg-viewer"></div>' +
'</div>' +
'<div id="loading"><img src="../webfwk/img/loading.gif"></div>' ;

            this._wa.html(html) ;

            this._wa.find('.refresh-action').change(function () { _that._load_viewer() ; }) ;

            var refresh_interval = this._wa.find('select[name="refresh_interval"]') ;
            refresh_interval.change(function () {
                _that._refresh_interval_sec = parseInt($(this).val()) ;
                _that._update_viewer() ;
            }) ;
            this._refresh_interval_sec = parseInt(refresh_interval.val()) ;

            this._wa.find('button[name="refresh"]').button().click(function () { _that._load_viewer() ; }) ;

            this._loading = this._wa.find('div#loading') ;

            var body      = this._wa.find('div#body') ;
            this._info    = body.find('div#info') ;
            this._updated = body.find('div#updated') ;

            this._viewer = new ELog_MessageViewer (
                this ,
                this._wa.find('#viewer') ,
                {
                    allow_groups: true ,
                    allow_runs:   !this.experiment.is_facility ,
                    allow_shifts: !this.experiment.is_facility
                }
            ) ;
            if (when_done) when_done () ;
        } ;

        this._latest_timestamp = 0 ;
        this._oldest_timestamp = 0 ;

        function event_timestamp (m) {
            if (typeof m === 'string') {
                m = eval('('+m+')') ;
            }
            return m.event_timestamp ;
        }
        this._load_viewer = function () {

            this._updated.html('Loading messages...') ;

            var params = {
                id:                      this.experiment.id ,
                scope:                   'experiment' ,
                search_in_messages:      1 ,
                search_in_tags:          1 ,
                search_in_values:        0 ,
                posted_at_experiment:    1 ,
                posted_at_shifts:        1 ,
                posted_at_runs:          1 ,
                format:                  'detailed' ,
                limit:                   this._wa.find('select[name="num_messages"]').val()
            } ;
            if (!this.experiment.is_facility) {
                if (this._wa.find('input[name="include_runs"]').attr('checked')) params.inject_runs = 1 ;
            }
            if (this._wa.find('input[name="show_deleted"]').attr('checked')) params.inject_deleted_messages = 1 ;

            Fwk.web_service_GET (
                '../logbook/ws/message_search.php' ,
                params ,
                function (data) {

                    var num_threads = data.ResultSet.Result.length ;
                    if (num_threads) {
                        _that._latest_timestamp = event_timestamp(data.ResultSet.Result[num_threads-1]) ;
                        _that._oldest_timestamp = event_timestamp(data.ResultSet.Result[0]) ;
                        _that._viewer.load(data.ResultSet.Result) ;
                    }
                    var num_runs = _that._viewer.num_runs() ;
                    var num_msgs = _that._viewer.num_rows() - num_runs ;
                    _that._info.html('<b>'+num_msgs+'</b> messages'+(num_runs ? ', runs: <b>'+_that._viewer.min_run()+' .. '+_that._viewer.max_run() : '')) ;
                    _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;

        this._update_viewer = function (when_done) {

            if (!this._latest_timestamp) return ;

            var params = {
                id:                      this.experiment.id ,
                scope:                   'experiment' ,
                search_in_messages:      1 ,
                search_in_tags:          0 ,
                search_in_values:        0 ,
                posted_at_experiment:    1 ,
                posted_at_shifts:        1 ,
                posted_at_runs:          1 ,
                format:                  'detailed' ,
                since:                   this._latest_timestamp
            } ;
            if (!this.experiment.is_facility) {
                if (this._wa.find('input[name="include_runs"]').attr('checked')) params.inject_runs = 1 ;
            }
            if (this._wa.find('input[name="show_deleted"]').attr('checked')) params.inject_deleted_messages = 1 ;

            Fwk.web_service_GET (
                '../logbook/ws/message_search.php' ,
                params ,
                function (data) {

                    var num_threads = data.ResultSet.Result.length ;
                    if (num_threads) {
                        _that._latest_timestamp = event_timestamp(data.ResultSet.Result[num_threads-1]) ;
                        _that._oldest_timestamp = event_timestamp(data.ResultSet.Result[0]) ;
                        _that._viewer.update(data.ResultSet.Result) ;
                    }
                    var num_runs = _that._viewer.num_runs() ;
                    var num_msgs = _that._viewer.num_rows() - num_runs ;
                    _that._info.html('<b>'+num_msgs+'</b> messages'+(num_runs ? ', runs: <b>'+_that._viewer.min_run()+' .. '+_that._viewer.max_run() : '')) ;
                    _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;

                    if (when_done) when_done() ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;

        this._backtrack = function () {

            if (!this._oldest_timestamp) return ;

            // - display the loader at the botton in the middle of the page

            this._loading.css({
                'display' : 'block' ,
                'right'   : this._wa.width() / 2 - this._loading.width() / 2
            }) ;

            var params = {
                id:                      this.experiment.id ,
                scope:                   'experiment' ,
                search_in_messages:      1 ,
                search_in_tags:          0 ,
                search_in_values:        0 ,
                posted_at_experiment:    1 ,
                posted_at_shifts:        1 ,
                posted_at_runs:          1 ,
                format:                  'detailed' ,
                before:                  this._oldest_timestamp ,
                limit:                   100
            } ;
            if (this._wa.find('input[name="include_runs"]').attr('checked')) params.inject_runs = 1 ;
            if (this._wa.find('input[name="show_deleted"]').attr('checked')) params.inject_deleted_messages = 1 ;

            Fwk.web_service_GET (
                '../logbook/ws/message_search.php' ,
                params ,
                function (data) {

                    _that._loading.css('display', 'none') ;

                    var num_threads = data.ResultSet.Result.length ;
                    if (num_threads) {
                        _that._oldest_timestamp = event_timestamp(data.ResultSet.Result[0]) ;
                        _that._viewer.append(data.ResultSet.Result) ;
                    }
                    var num_runs = _that._viewer.num_runs() ;
                    var num_msgs = _that._viewer.num_rows() - num_runs ;
                    _that._info.html('<b>'+num_msgs+'</b> messages'+(num_runs ? ', runs: <b>'+_that._viewer.min_run()+' .. '+_that._viewer.max_run() : '')) ;
                    _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;

                } ,
                function (msg) {
                    _that._loading.css('display', 'none') ;
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;

        this._enable_autoexpander = function (enable) {

            function evaluate () {
                var top    = _that._wa.offset().top ,
                    height = _that._wa.height() ;

                if (top < 0 && Math.abs(top) + $(document).height() >= height) {
                    _that._backtrack() ;
                }
            }
            if (enable) $('#fwk-center').bind  ('scroll', evaluate) ;
            else        $('#fwk-center').unbind('scroll', evaluate) ;
        } ;
    }
    Class.define_class (ELog_Live, FwkApplication, {}, {}) ;

    return ELog_Live ;
}) ;
