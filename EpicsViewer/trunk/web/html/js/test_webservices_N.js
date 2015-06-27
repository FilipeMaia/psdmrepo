require.config ({
    baseUrl: '..' ,

    waitSeconds : 15,
    urlArgs     : "bust="+new Date().getTime() ,

    paths: {
        'jquery'            : '/jquery/js/jquery-1.8.2' ,
        'jquery-ui'         : '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'jquery.resize'     : '/jquery/js/jquery.resize' ,
        'jquery.mousewheel' : '/jquery/js/jquery.mousewheel' ,
        'underscore'        : '/underscore/underscore-min' ,
        'webfwk'            : 'webfwk/js' ,
        'EpicsViewer'       : 'EpicsViewer/js'
    } ,
    shim : {
        'jquery' : {
            exports : '$'
        } ,
        'jquery-ui' : {
            exports : '$' ,
            deps : ['jquery']
        } ,
        'jquery.resize' :  {
            deps : ['jquery']
        } ,
        'jquery.mousewheel' :  {
            deps : ['jquery']
        } ,
        'underscore' : {
            exports  : '_'
        }
    }
}) ;

require ([
    'webfwk/CSSLoader' ,
    'EpicsViewer/TimeSeriesPlotN' , 'webfwk/RadioBox' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'jquery.resize', 'jquery.mousewheel', 'underscore'] ,

function (
    cssloader ,
    TimeSeriesPlotN, RadioBox) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    // Polifills
    var Integer = {
        MAX_VALUE: Math.pow(2, 31) - 1
    } ;
    if (!Date.prototype.toISOString) {
        (function () {
            function pad (number) {
                if (number < 10) {
                    return '0' + number ;
                }
                return number ;
            }
            Date.prototype.toISOString = function () {
                return this.getUTCFullYear() +
                    '-' + pad(this.getUTCMonth() + 1) +
                    '-' + pad(this.getUTCDate()) +
                    'T' + pad(this.getUTCHours()) +
                    ':' + pad(this.getUTCMinutes()) +
                    ':' + pad(this.getUTCSeconds()) +
                    '.' + (this.getUTCMilliseconds() / 1000).toFixed(3).slice(2, 5) +
                    'Z' ;
            } ;
        } ()) ;
    }
    function Test () {

        var _that = this ;

        var _INTERVAL_DEFS = [
            {name: ''+                 30, text:   "30 s",        title: "30 seconds" } ,
            {name: ''+                 60, text:    "1 m",        title:  "1 minutes"} ,
            {name: ''+             5 * 60, text:    "5 m",        title:  "5 minutes"} ,
            {name: ''+            10 * 60, text:   "10 m",        title: "10 minutes"} ,
            {name: ''+            30 * 60, text:   "30 m",        title: "30 minutes"} ,
            {name: ''+            60 * 60, text:    "1 h",        title:  "1 hour"} ,
            {name: ''+        2 * 60 * 60, text:    "2 h",        title:  "2 hours"} ,
            {name: ''+        4 * 60 * 60, text:    "4 h",        title:  "4 hours"} ,
            {name: ''+        8 * 60 * 60, text:    "8 h",        title:  "8 hours"} ,
            {name: ''+       12 * 60 * 60, text:   "12 h",        title: "12 hours"} ,
            {name: ''+       18 * 60 * 60, text:   "18 h",        title: "18 hours"} ,
            {name: ''+       24 * 60 * 60, text:    "1 d",        title:  "1 day"} ,
            {name: ''+       60 * 60 * 60, text:  "2.5 d",        title:  "2 and a half days"} ,
            {name: ''+   7 * 24 * 60 * 60, text:    "1 w",        title:  "1 week"} ,
            {name: ''+  14 * 24 * 60 * 60, text:    "2 w",        title:  "2 weeks"} ,
            {name: ''+  30 * 24 * 60 * 60, text:    "1 month",    title:  "1 month"} ,
            {name: ''+ 365 * 24 * 60 * 60, text:    "1 year",     title:  "1 year"}
        ] ;

        // make sure we always have some valid configuration, even in case when
        // no specific time range is provided to the appliucation.
        this._options = {
            pvs:  window.global_options.pvs  ?          window.global_options.pvs   : [] ,
            from: window.global_options.from ? new Date(window.global_options.from) : new Date(+(new Date()) - 2 * 24 * 3600. * 1000) ,
            to:   window.global_options.to   ? new Date(window.global_options.to)   : new Date()
        } ;

        this._timeSeriesPlot = null ;
        this._interval = null ;
        this._end = null ;

        this._num2load = 0 ;    // total number of PVs to load
        
        function padTimeWithZeroes (n) {
            return (n < 10 ? '0' : '') + n ;
        } ;
        
        function date2YmdLocal (d) {
            return d.getFullYear() +
                    '-' + padTimeWithZeroes(d.getMonth() + 1) +
                    '-' + padTimeWithZeroes(d.getDate()) ;
        }

        this._timeline_change = function (get_new_range_func) {

            // no timeline changes while loading a set of PVs
            if (this._num2load) return ;

            var t_min = this._options.from / 1000. ,
                t_max = this._options.to   / 1000. ;

            var t_range = get_new_range_func (
                t_min ,
                t_max ,
                Math.floor(Math.floor((t_max - t_min)/10))
            ) ;

            this._options.from = new Date(1000*t_range.min) ;
            this._options.to   = new Date(1000*t_range.max) ;

            this.load_all_timelines(t_range.xbins) ;
        } ;
        this._time_zoom_in = function (xbins) {
            this._timeline_change (function (t_min, t_max, t_delta_10percent) {
                return {
                    min: Math.max(0,                 t_min + t_delta_10percent) ,
                    max: Math.min(Integer.MAX_VALUE, t_max - t_delta_10percent) ,
                    xbins: xbins
                } ;
            }) ;
        } ;

        this._time_zoom_out = function (xbins) {
            this._timeline_change (function (t_min, t_max, t_delta_10percent) {
                return {
                    min: Math.max(0,                 t_min - t_delta_10percent) ,
                    max: Math.min(Integer.MAX_VALUE, t_max + t_delta_10percent) ,
                    xbins: xbins
                } ;
            }) ;
        } ;
        this._time_move_left = function (xbins, dx) {
            this._timeline_change (function (t_min, t_max, t_delta_10percent) {
                var t_delta = dx ? Math.round(dx * (t_max - t_min) / xbins) : t_delta_10percent ;
                return {
                    min: Math.max(0, t_min - t_delta) ,
                    max: Math.max(1, t_max - t_delta) ,
                    xbins: xbins
                } ;
            }) ;
        } ;
        this._time_move_right = function (xbins, dx) {
            this._timeline_change (function (t_min, t_max, t_delta_10percent) {
                var t_delta = dx ? Math.round(dx * (t_max - t_min) / xbins) : t_delta_10percent ;
                return {
                    min: Math.min(Integer.MAX_VALUE - 1, t_min + t_delta) ,
                    max: Math.min(Integer.MAX_VALUE,     t_max + t_delta) ,
                    xbins: xbins
                } ;
            }) ;
        } ;
        this._end_time_changed = function (date) {
            var end = date ;
            if (_.isUndefined(end)) {
                var ymd = this._end_ymd.val().split('-') ,
                    year  = ymd[0] ,
                    month = ymd[1] - 1 ,
                    day   = ymd[2] ,
                    hh = this._end_hh.val() ,
                    mm = this._end_mm.val() ,
                    ss = this._end_ss.val() ;
                end = new Date (year, month, day, hh, mm, ss) ;
            } else {
                this._end_ymd.val(date2YmdLocal(end)) ;
                this._end_hh. val(padTimeWithZeroes(end.getHours())) ;
                this._end_mm. val(padTimeWithZeroes(end.getMinutes())) ;
                this._end_ss. val(padTimeWithZeroes(end.getSeconds())) ;
            }
            var deltaMS = end - this._options.to ;
            this._options.from = new Date(+this._options.from + deltaMS) ;
            this._options.to   = new Date(+this._options.to   + deltaMS) ;
            console.log('_end_time_changed:', end.toISOString()+' deltaMS:', deltaMS ,
                        'from:', this._options.from.toISOString() ,
                        'to:', this._options.to.toISOString()) ;
            this.load_all_timelines() ;
        } ;

        this._y_range_lock = {
            
        } ;

        this._is_rendered = false ;
        this.run = function () {
            
            // make sure all windows and UI resources are properly
            // initialzied

            if (!this._is_rendered) {
                this._is_rendered = true ;
                
                this._interval = new RadioBox (
                    _INTERVAL_DEFS ,
                    function (name) {
                        console.log('selected interval: '+name)
                    } ,
                    {   activate: "60"}
                ) ;
            
                var now = new Date() ;

                this._end_ymd = $('#end_ymd > input')
                    .datepicker({
                        changeMonth: true,
                        changeYear: true})
                    .datepicker('option', 'dateFormat', 'yy-mm-dd')
                    .datepicker('setDate', date2YmdLocal(now))
                    .change(function () {
                        _that._end_time_changed() ;
                    }) ;
                    
                this._end_hh = $('#end_hh > input')
                    .val(padTimeWithZeroes(now.getHours()))
                    .change(function () {
                        var v = parseInt($(this).val()) || 0 ;
                        // Validate before triggering further actions.
                        if (v > 23 || v < 0) {
                            $(this).val(padTimeWithZeroes(_that._options.to.getHours())) ;
                            return ;
                        }
                        _that._end_time_changed() ;
                    }) ;
                    
                this._end_mm = $('#end_mm > input')
                    .val(padTimeWithZeroes(now.getMinutes()))
                    .change(function () {
                        var v = parseInt($(this).val()) || 0 ;
                        // Validate before triggering further actions.
                        if (v > 59 || v < 0) {
                            $(this).val(padTimeWithZeroes(_that._options.to.getMinutes())) ;
                            return ;
                        }
                        _that._end_time_changed() ;
                    }) ;
                this._end_ss = $('#end_ss > input')
                    .val(padTimeWithZeroes(now.getSeconds()))
                    .change(function () {
                        var v = parseInt($(this).val()) || 0 ;
                        // Validate before triggering further actions.
                        if (v > 59 || v < 0) {
                            $(this).val(padTimeWithZeroes(_that._options.to.getSeconds())) ;
                            return ;
                        }
                        _that._end_time_changed() ;
                    }) ;
                this._end_ss = $('#end_now > button')
                    .button()
                    .click(function () {
                        _that._end_time_changed(new Date()) ;
                    }) ;

                this._interval.display($('#interval')) ;
                this._timeSeriesPlot = new TimeSeriesPlotN ({
                    x_zoom_in:      function (e) { _that._time_zoom_in   (e.xbins) ; } ,
                    x_zoom_out:     function (e) { _that._time_zoom_out  (e.xbins) ; } ,
                    x_move_left:    function (e) { _that._time_move_left (e.xbins, e.dx) ; } ,
                    x_move_right:   function (e) { _that._time_move_right(e.xbins, e.dx) ; } ,
                    y_range_change: function (name, yRange) {
                        if (yRange) {
                            _that._y_range_lock[name] = {
                                min: yRange.min ,
                                max: yRange.max
                            } ;
                            console.log("locked Y range of plot '"+name+"' to:", yRange) ;
                        } else {
                            if (_that._y_range_lock[name]) {
                                delete _that._y_range_lock[name] ;
                                console.log("unlocked Y range of plot '"+name) ;
                            }
                        }
                    }
                }) ;
                this._timeSeriesPlot.display($('#getdata_timeseries')) ;
            }
            
            // start with requesting a list of all kown PVs
            this.load_pvs() ;
        } ;
        this.loadlog = function (msg) {
            if (!this._loadlog_elem) { this._loadlog_elem = $('#getdata > #loadlog') ; }
            this._loadlog_elem.prepend (
                _.isUndefined(msg) ?
                    '<div style="width:100%; height:4px; margin-top:5px; border-top:1px solid #e0e0e0;">&nbsp</div>' :
                    (Date.now() / 1000.)+': '+msg+'<br>'
            ) ;
        } ;
        this.info = function (msg) {
            if (!this._info_elem) { this._info_elem = $('#getdata > #info') ; }
            this._info_elem.prepend (
                _.isUndefined(msg) ?
                    '<div style="width:100%; height:4px; margin-top:5px; border-top:1px solid #e0e0e0;">&nbsp</div>' :
                    msg+'<br>'
            ) ;
        } ;
        this.report_error = function (msg) {
            this._loadlog('<span class="error">'+msg+'</span>') ;
        } ;
        this.web_service_GET = function (url, params, on_success, on_failure) {
            var jqXHR = $.get(url, params, function (data) {
                if (on_success) on_success(data) ;
            },
            'JSON').error(function () {
                var msg = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
                if (on_failure) on_failure(msg) ;
                else            _that.report_error(msg) ;
            }) ;
        } ;

        this.pvs = null ;
        this.load_pvs = function () {
            this.web_service_GET (
                "/epics/mgmt/bpl/getAllPVs" ,
                {} ,
                function (data) {
                    _that.pvs = data ;
                    _that.display_pvs() ;
                }
            ) ;
        } ;
        this.display_pvs = function () {
            $('#loaded').text('getAllPVs ('+this.pvs.length+')') ;
            var elem = $('#getallpvs') ;
            elem.html (
                _.reduce (
                    this.pvs ,
                    function (html, pv) {
                        return html += '<div class="pvname">'+pv+'</div>';
                    } ,
                    ''
                )+'<div class="pvname_endoflist"></div>'
            ) ;
            var pvname_elements = elem.children('.pvname') ;
            pvname_elements.click(function () {
                var e = $(this) ;
                var pvname = e.text() ;
                console.log('display_pvs - old list of PVs:', _that._options.pvs) ;
                if (_.indexOf(_that._options.pvs, pvname) === -1) {
                    e.addClass('selected') ;
                    _that._options.pvs.push(pvname) ;
                    _that._num2load++ ;
                    _that.load_pvtypeinfo(pvname) ;
                } else {
                    e.removeClass('selected') ;
                    _that._options.pvs = _.filter(_that._options.pvs, function (v) { return v !== pvname ; }) ;
                    delete _that.pvtypeinfo[pvname] ;
                    _that.display_timeline() ;
                }
                console.log('display_pvs - new list of PVs:', _that._options.pvs) ;
            }) ;

            // Load the specified PV if the one was passed as the parameter
            // to the application.
            if (this._options.pvs.length) {
                for (var i = 0; i < this._options.pvs.length; ++i) {
                    this._num2load++ ;
                    this.load_pvtypeinfo(this._options.pvs[i]) ;
                }
            }
        } ;
        this.pvtypeinfo = {} ;
        this.load_pvtypeinfo = function (pvname) {
            console.log('load_pvtypeinfo - num2load:', this._num2load) ;
            $('#selected').text('getData ('+pvname+')') ;
            this.loadlog() ;
            this.loadlog('loading PV type info for '+pvname+'...') ;
            this.web_service_GET (
                "/epics/mgmt/bpl/getPVTypeInfo" ,
                {pv: pvname} ,
                function (data) {
                    if (data.extraFields.RTYP === 'waveform') {
                        _that.loadlog('loaded, but ignoring because this is the Waveform') ;
                        if (_.indexOf(_that._options.pvs, pvname) !== -1)
                            delete _that._options.pvs[pvname] ;
                        return ;
                    }
                    _that.pvtypeinfo[pvname] = data ;
                    _that.load_timeline(pvname) ;
                }
            ) ;
        } ;
        this.load_all_timelines = function (xbins) {
            this._num2load = _that._options.pvs.length ;
            for (var pvname in this.pvtypeinfo) {
                this.load_timeline(pvname, xbins) ;
            }
        } ;
        this.pvdata = {} ;
        this.load_timeline = function (pvname, xbins) {

            this.loadlog('loading timeline: '+pvname) ;

            // Apply a bin aggregation function if the current time range includes
            // too many measurements. The bining algorithm is based on the sampling
            // period of a PV.
            //
            // TODO: the present algorithm is way to primitive to assume that
            //       PVs get updated at the specified sampling rate rate.
            //       The actual sampling rate definition varies depending on
            //       the sampling method:
            //         'MONITOR' - monitoring status changes (frequency is limited by the sampling period)
            //         'SCAN'    - fixed frequency (exactly sampling period)
            //       Besides, PVs may not be archived for some reason.
            //       One way to deal with that would be to make a separate request
            //       to the backend using some smaller range (for speed) to "calibrate"
            //       the actual update frequency of a PV.

            var delta_sec = Math.abs(this._options.to - this._options.from) / 1000 ;
            var samplingPeriod_sec = +this.pvtypeinfo[pvname].samplingPeriod ;
            var samplesInDelta = samplingPeriod_sec ? Math.round(delta_sec / samplingPeriod_sec) : 1 ;

            var pv_fetch_method = pvname ;  // raw data, no server-side processing

            var xbins_best_guess = xbins ? xbins : 1024 ;
            if (xbins_best_guess) {
                if (samplesInDelta > 2 * xbins_best_guess) {
                    var binFactor = Math.round(delta_sec / xbins_best_guess) ;
                    pv_fetch_method = 'mean_'+binFactor+'('+pvname+')' ;
                }
            }
            var params = {
                pv:   pv_fetch_method ,
                from: this._options.from.toISOString() ,
                to:   this._options.to.toISOString() ,
            } ;
            console.log('load_timeline - params:', params) ;
            this.web_service_GET (
                "/epics/retrieval/data/getData.json" ,
                params  ,
                function (data) {
                    _that.pvdata[pvname] = data ;
                    _that._num2load-- ;
                    if (!_that._num2load) {
                        _that.display_timeline() ;
                    }
                } ,
                function (msg) {
                    console.log('load_timeline: error:', msg) ;
                    _that._num2load-- ;
                }
            ) ;
            this.info() ;
            this.info('type info: scalar='+this.pvtypeinfo[pvname].scalar+' RTYP='+this.pvtypeinfo[pvname].extraFields.RTYP) ;
        } ;
        this.display_timeline = function () {

            var num_pvs = 0 ;
            for (var pvname in this.pvtypeinfo) {
                ++num_pvs;
                console.log('display_timeline - pvtypeinfo['+pvname+']', this.pvtypeinfo[pvname]) ;
                console.log('display_timeline - pvdata['+pvname+'].length', this.pvdata[pvname].length) ;
                console.log('display_timeline - pvdata['+pvname+'][0].data.length', this.pvdata[pvname].length ? this.pvdata[pvname][0].data.length : ' n/a') ;
            }
            console.log('display_timeline - num pvs', num_pvs) ;
            if (!num_pvs) {
                console.log('display_timeline - noting to display', num_pvs) ;
                return ;
            }

            this.loadlog("preparing data...") ;

            var x_range = {
                min: this._options.from / 1000. ,   // msec -> secs.*
                max: this._options.to   / 1000.     // msec -> secs.*
            } ;
            console.log('display_timeline - x_range', x_range) ;

            // Extract data to plot from the loaded data object

            var many_series = [] ;
            for (var pvname in this.pvtypeinfo) {

                var y_min = Number.POSITIVE_INFINITY ;
                var y_max = Number.NEGATIVE_INFINITY ;

                var points = [] ;
                var pvdata = this.pvdata[pvname] ;
                for (var i in pvdata) {
                    var d = pvdata[i].data ;
                    for (var j in d) {
                        var sample = d[j] ;

                        // bypass measurements which are not in the timeframe
                        var t = sample.secs + 1.e-9 * sample.nanos ;
                        if (t < x_range.min || t > x_range.max) continue;

                        var v = sample.val ;
                        if (_.isFinite(v)) {
                            y_min = Math.min(y_min, v) ;
                            y_max = Math.max(y_max, v) ;
                            points.push([t, v]) ;
                        }
                    }
                }
                console.log('display_timeline - points.length', points.length) ;

                many_series.push({
                    name: pvname ,
                    yRange: {
                        min: y_min ,
                        max: y_max
                    } ,
                    yLockedRange: this._y_range_lock[pvname] ? this._y_range_lock[pvname] : undefined ,
                    points: points
                }) ;
            }

            // Update controls accordingly
    
            this._end_ymd.datepicker('setDate', date2YmdLocal(this._options.to)) ;
            this._end_hh.val(padTimeWithZeroes(this._options.to.getHours())) ;
            this._end_mm.val(padTimeWithZeroes(this._options.to.getMinutes())) ;
            this._end_ss.val(padTimeWithZeroes(this._options.to.getSeconds())) ;


            // Plot the points using an appropriate method

            this.loadlog("rendering...") ;

            this._timeSeriesPlot.load(x_range, many_series) ;

            console.log('display_timeline - updated plot') ;
            this.loadlog('done') ;

            this.info('interval: '+t2str(x_range.min)+' - '+t2str(x_range.max)) ;
            this.info('interval length [s]: '+t2str(x_range.max - x_range.min)) ;
            this.info('min value: '+y_min) ;
            this.info('max value: '+y_max) ;
        } ;
        function t2str(t) {
            var s = Math.floor(t) ;
            var nsec = Math.floor(1e9*(t - s));
            return '<b>'+s+'</b>.'+nsec ;
        }
    }

    // Starting point for the application
    $(function () {
        var test = new Test() ;
        test.run() ;
    }) ;

}) ;