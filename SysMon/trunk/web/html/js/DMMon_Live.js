define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../sysmon/css/DMMon_Live.css') ;

    var _DIRECTIONS = ['DSS2FFB', 'FFB2ANA', 'OTHER'] ;

    var _DIRECTION_TITLE = {
        'DSS2FFB' : 'FFB' ,
        'FFB2ANA' : 'ANA' ,
        'OTHER'   : 'Other transfers'
    } ;

    /**
     * The application for displaying the data movers stats in live mode
     *
     * @returns {DMMon_Live}
     */
    function DMMon_Live (instr_name, app_config) {

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

        this._update_ival_sec = 1 ;
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

        this._xfer = null ;
        this._begin_time_sec = 0 ;
        this._end_time_sec  = 0 ;

        this._wa = function () {
            if (!this._wa_elem) {
                var html =
'<div id="dmmon-live-'+this._instr_name+'" class="dmmon-live"> ' +

  '<div id="controls" > ' +
    '<div style="float:right;" > ' +
      '<button name="update" class="control-button" title="update from the database" >UPDATE</button> ' +
    '</div> ' +
    '<div style="clear:both;" ></div> ' +
  '</div> ' +

  '<div id="update_info" > ' +
    '<div class="info" id="info"    style="float:left;"  >&nbsp;</div> ' +
    '<div class="info" id="updated" style="float:right;" >&nbsp;</div> ' +
    '<div style="clear:both;"></div> ' +
  '</div> ' +

  '<div id="tabs" > ' +
    '<ul> ' +
                _.reduce(_DIRECTIONS, function (html, dir) { return html +=
      '<li><a href="#'+dir+'" >'+_DIRECTION_TITLE[dir]+'</a></li> ' ; }, '') +
    '</ul> ' +
                _.reduce(_DIRECTIONS, function (html, dir) { return html +=
    '<div id="'+dir+'" > ' +
      '<div class="tab-cont" > ' +
        '<div id="view-'+_that._instr_name+'-'+dir+'" > ' +
          '<input type="radio" id="view-'+_that._instr_name+'-'+dir+'-table" name="view-'+_that._instr_name+'-'+dir+'" checked="checked" ><label for="view-'+_that._instr_name+'-'+dir+'-table" title="view as a table" ><img src="../sysmon/img/enumeration1.png" /></label> ' +
          '<input type="radio" id="view-'+_that._instr_name+'-'+dir+'-plot"  name="view-'+_that._instr_name+'-'+dir+'"                   ><label for="view-'+_that._instr_name+'-'+dir+'-plot"  title="view as a plot"  ><img src="../sysmon/img/scatter1.png" /></label> ' +
        '</div> ' +
        '<div id="table" class="presentation" ></div> ' +
        '<div id="plot"  class="presentation" ></div> ' +
      '</div> ' +
    '</div> ' ; }, '') +

  '</div> ' +

'</div> ' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dmmon-live-'+this._instr_name) ;
            }
            return this._wa_elem ;
        } ;
        this._tabs = function () {
            if (!this._tabs_elem) {
                this._tabs_elem = this._wa().children('#tabs').tabs() ;
            }
            return this._tabs_elem ;
        } ;
        this._view_selector = function (dir, name) {
            if (!this._view_selector_elem) {
                this._view_selector_elem  = {} ;
                this._view_buttonset_elem = {} ;
            }
            if (!this._view_selector_elem[dir]) {
                this._view_selector_elem [dir] = {} ;
                this._view_buttonset_elem[dir] = this._tabs().children('#'+dir).find('div#view-'+this._instr_name+'-'+dir).buttonset() ;
            }
            if (!this._view_selector_elem[dir][name]) {
                this._view_selector_elem[dir][name] = this._view_buttonset_elem[dir].children('#view-'+this._instr_name+'-'+dir+'-'+name) ;
            }
            return this._view_selector_elem[dir][name] ;
        } ;
        this._view = function (dir, name) {
            if (!this._view_elem)            this._view_elem            = {} ;
            if (!this._view_elem[dir]) this._view_elem[dir] = {} ;
            if (!this._view_elem[dir][name]) {
                this._view_elem[dir][name] = this._tabs().children('div#'+dir).find('div#'+name) ;
            }
            return this._view_elem[dir][name] ;
        } ;
        this._table = function (dir, hosts, reset) {

            // Always recreate the table if the "reset" mode is requested

            if (!this._table_obj) this._table_obj = {} ;
            if (!this._table_obj[dir] || reset) {
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
                this._table_obj[dir] = new SimpleTable.constructor (this._view(dir, 'table'), hdr, rows) ;
                this._table_obj[dir].display() ;
            }
            return this._table_obj[dir] ;
        } ;
        this._plot = function (dir, hosts, reset) {

            // Always recreate the table if the "reset" mode is requested

            if (!this._plot_obj) this._plot_obj = {} ;
            if (!this._plot_obj[dir] || reset) {
                Highcharts.setOptions({
                    global: {
                        useUTC: false
                    }
                }) ;
                var chartdef = {
                    chart: {
                        renderTo:  this._view(dir, 'plot').get(0) ,
                        animation: Highcharts.svg , // don't animate in old IE
                        type:      'spline' ,
                        margin:    [40, 20, 70, 40] ,
                    } ,
                    title: {
                        text: '<b>MB/s</b>'
                    } ,
                    subtitle: {
                        text: ''
                    } ,
                    xAxis: {
                        type: 'datetime' ,
                        gridLineWidth:  1 /*,
                        minPadding:     0.2 ,
                        maxPadding:     0.2 ,
                        maxZoom:       60*/
                    } ,
                    yAxis: {
                        title: {
                            text: ''
                        } ,
                        minPadding:  0.1 ,
                        maxPadding:  0.1 ,
                        //maxZoom:    60 ,
                        plotLines: [{
                            value: 0 ,
                            width: 1 ,
                            color: '#808080'
                        }]
                    } ,
                    legend: {
                        enabled: true
                    } ,
                    exporting: {
                        enabled: false
                    } ,
                    plotOptions: {
                        series: {
                            lineWidth: 2 ,
                            point: {
                            }
                        }
                    } ,
                    series: []
                } ;
                for (var i in hosts) chartdef.series.push({
                    name: hosts[i] ,
                    data: []
                }) ;
                this._plot_obj[dir] = new Highcharts.Chart(chartdef) ;
            }
            return this._plot_obj[dir] ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().children('#update_info').children('#info') ;
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#update_info').children('#updated') ;
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

            // Initialize view controls

            for (var i in _DIRECTIONS) {
                var dir = _DIRECTIONS[i] ;
                this._view_selector(dir, 'table').click(function () { _that._display() ; }) ;
                this._view_selector(dir, 'plot') .click(function () { _that._display() ; }) ;
            }
            this._load() ;
        } ;
        this._load = function () {
            this._action (
                '../sysmon/ws/dmmon_xfer_get.php' ,
                {   instr_name:  this._instr_name ,
                    begin_time:  this._end_time_sec ,
                    max_entries: 10
                }
            ) ;
        } ;
        this._action = function (url, params) {
            Fwk.web_service_GET (url, params, function (data) {

                // Reset the display if the list of hosts won't match the previous one
                // in any scope.
                
                var reset = false || !_that._xfer ;
                if (!reset) {
                    for (var dir in data.xfer.directions) {

                        reset = reset || (_that._xfer.directions[dir] === undefined) ;
                        if (reset) break ;

                        reset = reset || (_that._xfer.directions[dir].in_hosts.length !== data.xfer.directions[dir].in_hosts.length) ;
                        if (reset) break ;

                        for (var i in _that._xfer.directions[dir].in_hosts) {
                            reset = reset || (_that._xfer.directions[dir].in_hosts[i] !== data.xfer.directions[dir].in_hosts[i]) ;
                            if (reset) break ;
                        }
                        if (reset) break ;
                    }
                }
                _that._xfer = data.xfer ;
                if (!_that._begin_time_sec) _that._begin_time_sec = _that._xfer.begin_time_sec ;
                _that._end_time_sec = _that._xfer.end_time_sec ;
                _that._display(reset) ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._display = function (reset) {
            for (var dir in this._xfer.directions) {
                if (this._view_selector(dir, 'table').attr('checked')) {
                    this._view(dir, 'table').removeClass('view-hidden') .addClass('view-visible') ;
                    this._view(dir, 'plot') .removeClass('view-visible').addClass('view-hidden') ;
                    this._display_table(dir, reset) ;
                } else if (this._view_selector(dir, 'plot').attr('checked')) {
                    this._view(dir, 'table').removeClass('view-visible').addClass('view-hidden') ;
                    this._view(dir, 'plot') .removeClass('view-hidden') .addClass('view-visible') ;
                    this._display_plot(dir, reset) ;
                }
            }
        } ;
        this._display_table = function (dir, reset) {
            var rows = [] ;
            for (var i in this._xfer.directions[dir].stats) {
                var stats = this._xfer.directions[dir].stats[i] ;
                var row = ['<span style="font-weight:normal;" >'+stats.time.day+'</span>&nbsp;&nbsp;'+stats.time.hms] ;
                for (var j in stats.rates) {
                    var rate = stats.rates[j] ;
                    row.push(rate ? rate : '&nbsp;') ;
                }
                rows.push(row) ;
            }
            this._table(dir, this._xfer.directions[dir].in_hosts, reset).load(rows) ;
        } ;
        this._display_plot = function (dir, reset) {

            var plot = this._plot(dir, this._xfer.directions[dir].in_hosts, reset) ;

            // Bulk update to the series before redrawing teh whole
            // chart.

            for (var i in this._xfer.directions[dir].stats) {

                var stats = this._xfer.directions[dir].stats[i] ;
                var t_sec = stats.timestamp ;

                for (var j in this._xfer.directions[dir].in_hosts) {

                    // shift when reaching maximum history depth

                    var redraw_now = false ;
                    var max_depth_sec = 60 ;
                    var shift = plot.series[j].data.length && (t_sec - plot.series[j].data[0].x > max_depth_sec) ;

                    plot.series[j].addPoint([t_sec * 1000, stats.rates[j]], redraw_now, shift) ;
                }
            }
            plot.redraw() ;
        } ;
    }
    Class.define_class (DMMon_Live, FwkApplication, {}, {}) ;
    
    return DMMon_Live ;
}) ;
