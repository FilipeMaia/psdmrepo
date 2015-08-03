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
    'EpicsViewer/TimeSeriesPlot', 'EpicsViewer/WaveformPlot' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'jquery.resize', 'jquery.mousewheel', 'underscore'] ,

function (
    cssloader ,
    TimeSeriesPlot, WaveformPlot) {

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
            Date.prototype.toISOString1 = function () {
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

        this._timeSeriesPlot = null ;
        this._waveformPlot = null ;

        this.run = function () {
            // make sure all windows and UI resources are properly
            // initialzied
            if (!this._timeSeriesPlot) {
                this._timeSeriesPlot = new TimeSeriesPlot ({
                    time_zoom_in: function (e) {
                        if (_that.pvdata) {
                            var pvdata_length      = _that.pvdata.length ,
                                pvdata_data_length = _that.pvdata[pvdata_length-1].data.length ,
                                t_min_sec          = _that.pvdata[0]              .data[0]                   .secs ,
                                t_max_sec          = _that.pvdata[pvdata_length-1].data[pvdata_data_length-1].secs ,
                                t_delta            = t_max_sec - t_min_sec ,
                                t_delta_10percent  = Math.floor(t_delta/10) ,
                                t_new_min_sec      = Math.max(0,                 t_min_sec + t_delta_10percent) ,
                                t_new_max_sec      = Math.min(Integer.MAX_VALUE, t_max_sec - t_delta_10percent) ,
                                t_new_min_date     = new Date(1000*t_new_min_sec) ,
                                t_new_max_date     = new Date(1000*t_new_max_sec) ;

                            window.global_options.from = t_new_min_date.toISOString() ;
                            window.global_options.to   = t_new_max_date.toISOString() ;

                            _that.load_timeline(e.xbins) ;
                        }
                    } ,
                    time_zoom_out: function (e) {
                        if (_that.pvdata) {
                            var pvdata_length      = _that.pvdata.length ,
                                pvdata_data_length = _that.pvdata[pvdata_length-1].data.length ,
                                t_min_sec          = _that.pvdata[0]              .data[0]                   .secs ,
                                t_max_sec          = _that.pvdata[pvdata_length-1].data[pvdata_data_length-1].secs ,
                                t_delta            = t_max_sec - t_min_sec ,
                                t_delta_10percent  = Math.floor(t_delta/10) ,
                                t_new_min_sec      = Math.max(0,                 t_min_sec - t_delta_10percent) ,
                                t_new_max_sec      = Math.min(Integer.MAX_VALUE, t_max_sec + t_delta_10percent) ,
                                t_new_min_date     = new Date(1000*t_new_min_sec) ,
                                t_new_max_date     = new Date(1000*t_new_max_sec) ;

                            window.global_options.from = t_new_min_date.toISOString() ;
                            window.global_options.to   = t_new_max_date.toISOString() ;

                            _that.load_timeline(e.xbins) ;
                        }
                    } ,
                    time_move_left: function (e) {
                        if (_that.pvdata) {
                            var pvdata_length      = _that.pvdata.length ,
                                pvdata_data_length = _that.pvdata[pvdata_length-1].data.length ,
                                t_min_sec          = _that.pvdata[0]              .data[0]                   .secs ,
                                t_max_sec          = _that.pvdata[pvdata_length-1].data[pvdata_data_length-1].secs ,
                                t_delta            = t_max_sec - t_min_sec ,
                                t_delta_10percent  = Math.floor(t_delta/10) ,
                                // slide the time window to the left  
                                t_new_min_sec      = Math.max(0,                 t_min_sec - t_delta_10percent) ,
                                t_new_max_sec      = Math.min(Integer.MAX_VALUE, t_max_sec - t_delta_10percent) ,
//                                // move the plot to the left
//                                t_new_min_sec      = Math.min(Integer.MAX_VALUE, t_min_sec + t_delta_10percent) ,
//                                t_new_max_sec      = Math.max(0,                 t_max_sec + t_delta_10percent) ,
                                t_new_min_date     = new Date(1000*t_new_min_sec) ,
                                t_new_max_date     = new Date(1000*t_new_max_sec) ;

                            window.global_options.from = t_new_min_date.toISOString() ;
                            window.global_options.to   = t_new_max_date.toISOString() ;

                            _that.load_timeline(e.xbins) ;
                        }
                    } ,
                    time_move_right: function (e) {
                        console.log('time_move_right -') ;
                        if (_that.pvdata) {
                            var pvdata_length      = _that.pvdata.length ,
                                pvdata_data_length = _that.pvdata[pvdata_length-1].data.length ,
                                t_min_sec          = _that.pvdata[0]              .data[0]                   .secs ,
                                t_max_sec          = _that.pvdata[pvdata_length-1].data[pvdata_data_length-1].secs ,
                                t_delta            = t_max_sec - t_min_sec ,
                                t_delta_10percent  = Math.floor(t_delta/10) ,
                                // move the time window to the right
                                t_new_min_sec      = Math.min(Integer.MAX_VALUE, t_min_sec + t_delta_10percent) ,
                                t_new_max_sec      = Math.max(0,                 t_max_sec + t_delta_10percent) ,
//                                // move the plot to the right
//                                t_new_min_sec      = Math.max(0,                 t_min_sec - t_delta_10percent) ,
//                                t_new_max_sec      = Math.min(Integer.MAX_VALUE, t_max_sec - t_delta_10percent) ,
                                t_new_min_date     = new Date(1000*t_new_min_sec) ,
                                t_new_max_date     = new Date(1000*t_new_max_sec) ;

                            window.global_options.from = t_new_min_date.toISOString() ;
                            window.global_options.to   = t_new_max_date.toISOString() ;

                            _that.load_timeline(e.xbins) ;
                        }
                    }
                }) ;
                this._timeSeriesPlot.display($('#getdata_timeseries')) ;
            }
            if (!this._waveformPlot) {
                this._waveformPlot = new WaveformPlot() ;
//                this._waveformPlot.display($('#getdata_waveform')) ;
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
                pvname_elements.removeClass('selected') ;
                e.addClass('selected') ;
                var pvname = e.text() ;
                _that.load_pvtypeinfo(pvname) ;
            }) ;
            // Load the specified PV if the one was passed as teh parameter
            // to the application.
            if (window.global_options.pv)
                this.load_pvtypeinfo(window.global_options.pv) ;
        } ;
        this.pvtypeinfo = null ;
        this.load_pvtypeinfo = function (pvname) {
            $('#selected').text('getData ('+pvname+')') ;
            this.loadlog() ;
            this.loadlog("loading PV type info...") ;
            this.web_service_GET (
                "/epics/mgmt/bpl/getPVTypeInfo" ,
                {pv: pvname} ,
                function (data) {
                    _that.pvtypeinfo = data ;
                    _that.load_timeline() ;
                }
            ) ;
        } ;
        this.pvdata = null ;
        this.load_timeline = function (xbins) {
            if (this._is_loading) return ;
            this._is_loading = true ;
            this.loadlog('loading timeline...') ;
            var pv = this.pvtypeinfo.pvName ;
            if (xbins && window.global_options.from && window.global_options.to) {
                var from_msec = +(new Date(window.global_options.from)) ,
                    to_msec   = +(new Date(window.global_options.to)) ,
                    delta_sec = Math.abs(to_msec - from_msec) / 1000 ;
                if (delta_sec > 2 * xbins) {
                    var binFactor = Math.round(delta_sec / xbins) ;
                    pv = 'mean_'+binFactor+'('+pv+')' ;
                }
            }
            var params = {
                pv: pv
            } ;
            if (window.global_options.from) params.from = window.global_options.from ;
            if (window.global_options.to)   params.to   = window.global_options.to ;
            this.web_service_GET (
                "/epics/retrieval/data/getData.json" ,
                params  ,
                function (data) {
                    _that.pvdata = data ;
                    _that.display_timeline() ;
                    _that._is_loading = false ;
                } ,
                function (msg) {
                    console.log('load_timeline: error:', msg) ;
                    _that._is_loading = false ;
                }
            ) ;
            this.info() ;
            this.info('type info: scalar='+this.pvtypeinfo.scalar+' RTYP='+this.pvtypeinfo.extraFields.RTYP) ;
        } ;
        this.display_timeline = function () {

            console.log('pvtypeinfo', this.pvtypeinfo) ;

            this.loadlog("preparing data...") ;

            // Extract data to plot from the loaded data object

            var is_waveform = this.pvtypeinfo.extraFields.RTYP == 'waveform' ;

            var x_min = 0 ;
            var x_max   = 0 ;
            var y_min   = Number.POSITIVE_INFINITY ;
            var y_max   = Number.NEGATIVE_INFINITY ;
            var points  = [] ;
            for (var i in this.pvdata) {
                var d = this.pvdata[i].data ;
                for (var j in d) {
                    var sample = d[j] ;
                    var t = sample.secs + 1.e-9 * sample.nanos ;
                    if (!x_min) x_min = t ;
                    x_max = t ;

                    var v = sample.val ;
                    if (is_waveform) {
                        for (var k in v) {
                            var vk = v[k] ;
                            y_min = Math.min(y_min, vk) ;
                            y_max = Math.max(y_max, vk) ;
                        }
                        points.push([t, v]) ;
                    } else {
                        if (_.isFinite(v)) {
                            y_min = Math.min(y_min, v) ;
                            y_max = Math.max(y_max, v) ;
                            points.push([t, v]) ;
                        }
                    }
                }
            }
            var x_range = {min: x_min, max: x_max} ,
                y_range = {min: y_min, max: y_max} ;

            // Plot the points using an appropriate method

            this.loadlog("rendering...") ;

            if (this.pvtypeinfo.extraFields.RTYP == 'waveform') { 
                this._waveformPlot.load(x_range, y_range, this.pvtypeinfo.elementCount, points) ;
            } else {
                this._timeSeriesPlot.load (
                    x_range ,
                    {   yRange: y_range ,
                        points: points
                    }
                ) ;
            }
            this.loadlog('done') ;

            this.info('# measurements: '+points.length) ;
            this.info('interval: '+t2str(x_min)+' - '+t2str(x_max)) ;
            this.info('interval length [s]: '+t2str(x_max - x_min)) ;
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