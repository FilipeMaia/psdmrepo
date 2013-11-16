/**
 * The application for displaying & managing the live stream of e-log messages & runs
 *
 * @returns {ELog_Live}
 */
function ELog_Live (experiment, access_list) {

    var that = this ;

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    FwkApplication.call(this) ;

    // ------------------------------------------------
    // Override event handler defined in the base class
    // ------------------------------------------------

    this.on_activate = function() {
        this.init() ;
        this.update_viewer() ;
    } ;

    this.on_deactivate = function() {
        this.init() ;
    } ;

    this.prev_refresh_sec     = 0 ;
    this.refresh_interval_sec = 0 ; // no automatic updates by default

    this.on_update = function () {
        this.init() ;
        if (this.active && this.refresh_interval_sec) {
            var now_sec = Fwk.now().sec ;
            if (Math.abs(now_sec - this.prev_refresh_sec) > this.refresh_interval_sec) {
                this.prev_refresh_sec = now_sec ;
                this.update_viewer() ;
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

    this.is_initialized = false ;

    this.wa      = null ;
    this.info    = null ;
    this.updated = null ;

    this.viewer = null ;

    this.init = function () {

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        this.container.html('<div id="elog-live"></div>') ;
        this.wa = this.container.find('div#elog-live') ;

        if (!this.access_list.elog.post_messages) {
            this.wa.html(this.access_list.no_page_access_html) ;
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
'  </div>' +
'  <div style="float:left;" >' +
'    <span>Include runs</span>' +
'    <input class="refresh-action" name="include_runs" type="checkbox" checked="checked" title="search for runs as well" />' +
'  </div>' +
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
'  <div id="viewer" class="elog-live-viewer"></div>' +
'</div>' ;
        this.wa.html(html) ;

        this.wa.find('.refresh-action').change(function () { that.load_viewer() ; }) ;

        var refresh_interval = this.wa.find('select[name="refresh_interval"]') ;
        refresh_interval.change(function () {
            that.refresh_interval_sec = parseInt($(this).val()) ;
            that.update_viewer() ;
        }) ;
        this.refresh_interval_sec = parseInt(refresh_interval.val()) ;

        this.wa.find('button[name="refresh"]').button().click(function () { that.load_viewer() ; }) ;

        var body     = this.wa.find('div#body') ;
        this.info    = body.find('div#info') ;
        this.updated = body.find('div#updated') ;

        this.viewer = new ELog_MessageViewer(this, this.wa.find('#viewer')) ;

        this.load_viewer() ;
    } ;

    this.latest_timestamp = 0 ;

    this.load_viewer = function () {

        this.updated.html('Loading messages...') ;

        var params = {
            id:                      this.experiment.id ,
            scope:                   'experiment' ,
            search_in_messages:      1 ,
            search_in_tags:          1 ,
            search_in_values:        1 ,
            posted_at_experiment:    1 ,
            posted_at_shifts:        1 ,
            posted_at_runs:          1 ,
            format:                  'detailed' ,
            limit:                   this.wa.find('select[name="num_messages"]').val()
        } ;
        if (this.wa.find('input[name="include_runs"]').attr('checked')) params.inject_runs = 1 ;
        if (this.wa.find('input[name="show_deleted"]').attr('checked')) params.inject_deleted_messages = 1 ;

        Fwk.web_service_GET (
            '../logbook/ws/Search.php' ,
            params ,
            function (data) {

                var num_threads = data.ResultSet.Result.length ;
                if (num_threads) {
                    that.latest_timestamp = data.ResultSet.Result[num_threads-1].event_timestamp ;
                    that.viewer.load(data.ResultSet.Result) ;
                }
                that.info.html('<b>'+that.viewer.num_rows()+'</b> threads') ;
                that.updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            } ,
            function (msg) {
                Fwk.report_error(msg) ; 
            }
        ) ;
    } ;

    this.update_viewer = function () {

        if (!this.latest_timestamp) return ;

        var params = {
            id:                      this.experiment.id ,
            scope:                   'experiment' ,
            search_in_messages:      1 ,
            search_in_tags:          1 ,
            search_in_values:        1 ,
            posted_at_experiment:    1 ,
            posted_at_shifts:        1 ,
            posted_at_runs:          1 ,
            format:                  'detailed' ,
            since:                   this.latest_timestamp
        } ;
        if (this.wa.find('input[name="include_runs"]').attr('checked')) params.inject_runs = 1 ;
        if (this.wa.find('input[name="show_deleted"]').attr('checked')) params.inject_deleted_messages = 1 ;

        Fwk.web_service_GET (
            '../logbook/ws/Search.php' ,
            params ,
            function (data) {

                var num_threads = data.ResultSet.Result.length ;
                if (num_threads) {
                    that.latest_timestamp = data.ResultSet.Result[num_threads-1].event_timestamp ;
                    that.viewer.update(data.ResultSet.Result) ;
                }
                that.info.html('<b>'+that.viewer.num_rows()+'</b> threads') ;
                that.updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            } ,
            function (msg) {
                Fwk.report_error(msg) ; 
            }
        ) ;
    } ;
}
define_class (ELog_Live, FwkApplication, {}, {});
