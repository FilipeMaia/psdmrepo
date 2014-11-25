define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../sysmon/css/DMMon_History.css') ;

    /**
     * The application for displaying the historical data for data movers stats
     *
     * @returns {DMMon_History}
     */
    function DMMon_History (instr_name, app_config) {

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

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 10 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._instr_name = instr_name ;
        this._app_config = app_config ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._xfer = {} ;
        this._last_request_time = 0 ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dmmon-history" >' +

  '<div style="float:right;" >' +
    '<button name="update" class="control-button" title="update from the database" >UPDATE</button>' +
  '</div>' +
  '<div style="clear:both;" ></div>' +

  '<div class="info" id="info"    style="float:left;"  >&nbsp;</div>' +
  '<div class="info" id="updated" style="float:right;" >&nbsp;</div>' +
  '<div style="clear:both;"></div>' +

  '<div id="rates" >' +
  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dmmon-history') ;
            }
            return this._wa_elem ;
        } ;
        this._table = function (hosts, reset) {

            // Always recreate the table if the "reset" mode is requested

            if (!this._table_obj || reset) {
                var rows = [] ;
                var source_host_coldef = [] ;
                for (var i in hosts) {
                    source_host_coldef.push (
                        {   name: hosts[i], sorted: false}
                    ) ;
                }
                var hdr = [
                    {   name: 'time', sorted: false} ,
                    {   name: 'input host rate [MB/s]', coldef: source_host_coldef, align: "center"}
                ] ;
                this._table_obj = new SimpleTable.constructor (this._wa().find('#rates'), hdr, rows) ;
                this._table_obj.display() ;
            }
            return this._table_obj ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().children('#info') ;
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'update' : _that._load() ; break ;
                }
            }) ;

            this._load() ;
        } ;
        this._load = function () {
            this._action (
                'Loading...' ,
                '../sysmon/ws/dmmon_xfer_get.php' ,
                {   instr_name: this._instr_name ,
                    direction:  'DSS2FFB' ,
                    begin_time: this._last_request_time}
            ) ;
        } ;
        this._action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET (url, params, function (data) {

                // Reset the display if the list of hosts won't match the previous one
                
                var reset = _that._xfer.in_hosts ? false : true ;
                if (!reset) {
                    reset |= _that._xfer.in_hosts.length === data.xfer.in_hosts.length ;
                    if (!reset) {
                        for (var i in _that._xfer.in_hosts) {
                            if (_that._xfer.in_hosts[i] !== data.xfer.in_hosts[i]) {
                                reset = true ;
                                break ;
                            }
                        }
                    }
                }
                _that._xfer = data.xfer ;
                _that._last_request_time = data.request_time ;
                _that._display(reset) ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._display = function (reset) {
            var rows = [] ;
            for (var i in this._xfer.stats) {
                var stats = this._xfer.stats[i] ;
                rows.push([stats.time].concat(stats.rates)) ;
            }
            this._table(this._xfer.in_hosts, reset).load(rows) ;
        } ;
    }
    Class.define_class (DMMon_History, FwkApplication, {}, {}) ;
    
    return DMMon_History ;
}) ;

