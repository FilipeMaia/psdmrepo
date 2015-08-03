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
    
    /**
     * Utility class representing a window in the timeline
     *
     * @returns {test_webservices_N.Interval}
     */
    function Interval () {}
    
    /**
     * Window definitions
     */
    Interval.WINDOW_DEFS = [
        {name: ''+ 365 * 24 * 60 * 60, text:   "1 year",  title:  "1 year"} ,
        {name: ''+  30 * 24 * 60 * 60, text:   "1 month", title:  "1 month"} ,
        {name: ''+  14 * 24 * 60 * 60, text:   "2 w",     title:  "2 weeks"} ,
        {name: ''+   7 * 24 * 60 * 60, text:   "1 w",     title:  "1 week"} ,
        {name: ''+       60 * 60 * 60, text: "2.5 d",     title:  "2 and a half days"} ,
        {name: ''+       24 * 60 * 60, text:   "1 d",     title:  "1 day"} ,
        {name: ''+       18 * 60 * 60, text:  "18 h",     title: "18 hours"} ,
        {name: ''+       12 * 60 * 60, text:  "12 h",     title: "12 hours"} ,
        {name: ''+        8 * 60 * 60, text:   "8 h",     title:  "8 hours"} ,
        {name: ''+        4 * 60 * 60, text:   "4 h",     title:  "4 hours"} ,
        {name: ''+        2 * 60 * 60, text:   "2 h",     title:  "2 hours"} ,
        {name: ''+            60 * 60, text:   "1 h",     title:  "1 hour"} ,
        {name: ''+            30 * 60, text:  "30 m",     title: "30 minutes"} ,
        {name: ''+            10 * 60, text:  "10 m",     title: "10 minutes"} ,
        {name: ''+             5 * 60, text:   "5 m",     title:  "5 minutes"} ,
        {name: ''+                 60, text:   "1 m",     title:  "1 minutes"} ,
        {name: ''+                 30, text:  "30 s",     title: "30 seconds"}
    ] ;
    
    Interval.maxZoomOut = Interval.WINDOW_DEFS[0].name ;
    Interval.minZoomIn  = Interval.WINDOW_DEFS[Interval.WINDOW_DEFS.length - 1].name ;

    /**
     * Find the name of the previous more narrow window if the one is available.
     * Return the input name if it was the first one. The function returns
     * undefined if the wrong window name is passed as the parameter.
     * 
     * @param {String} name
     * @returns {String|undefined}
     */
    Interval.zoomIn = function  (name) {
        for (var i = 0, num = Interval.WINDOW_DEFS.length; i < num; ++i) {
            var w = Interval.WINDOW_DEFS[i] ;
            if (w.name === name) {
                var wPrev = Interval.WINDOW_DEFS[i+1] ;
                return _.isUndefined(wPrev) ? name : wPrev.name ;
            }
        }
        console.log('Interval.zoomIn: unknown window: '+name) ;
        return undefined ;
    } ;

    /**
     * Find the name of the next wider window if the one is available.
     * Return the input name if it was the last one. The function returns
     * undefined if the wrong window name is passed as the parameter.
     * 
     * @param {String} name
     * @returns {String|undefined}
     */
    Interval.zoomOut = function  (name) {
        for (var i = 0, num = Interval.WINDOW_DEFS.length; i < num; ++i) {
            var w = Interval.WINDOW_DEFS[i] ;
            if (w.name === name) {
                var wNext = Interval.WINDOW_DEFS[i-1] ;
                return _.isUndefined(wNext) ? name : wNext.name ;
            }
        }
        console.log('Interval.zoomOut: unknown window: '+name) ;
        return undefined ;
    } ;

    function Test () {

        var _that = this ;


        // Always use the smallest time window to initialize the application

        var to   = new Date() ,
            from = new Date(+to - Interval.minZoomIn * 1000) ;

        this._options = {
            pvs:  window.global_options.pvs ,
            from: from ,
            to:   to
        } ;
        console.log('this._options:', this._options, 'Interval.minZoomIn:', Interval.minZoomIn) ;

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

        this._timeline_change = function (range) {

            // no timeline changes while loading a set of PVs
            if (this._num2load) return ;

            console.log('range.deltaZoom:', range.deltaZoom) ;
            var zoom = this._interval.active() ,
                to = Math.min(+(new Date()), +this._options.to + Math.round(zoom * 1000 * range.deltaZoom)) ;

            this._options.from = new Date(to - zoom * 1000) ;
            this._options.to   = new Date(to) ;

            this._loadAllTimeLines(range.xbins) ;
        } ;
        this._timeZoomIn = function (xbins) {

            // no timeline changes while loading a set of PVs
            if (this._num2load) return ;

            // Stop when reaching the minimal zoom to prevent plot jittering
            var prevZoom = this._interval.active() ,
                zoom     = Interval.zoomIn(prevZoom) ;

            if (prevZoom !== zoom) {

                 this._options.from = new Date(+this._options.to - zoom * 1000) ;

                 this._interval.activate(zoom) ;
                 this._loadAllTimeLines(xbins) ;
             }
        } ;

        this._timeZoomOut = function (xbins) {

            // no timeline changes while loading a set of PVs
            if (this._num2load) return ;

            // Stop when reaching the minimal zoom to prevent plot jittering
            var prevZoom = this._interval.active() ,
                zoom     = Interval.zoomOut(prevZoom) ;

            if (prevZoom !== zoom) {

                this._options.from = new Date(+this._options.to - zoom * 1000) ;

                this._interval.activate(zoom) ;
                this._loadAllTimeLines(xbins) ;
             }
        } ;
        this._timeZoom = function (zoom) {

            // no timeline changes while loading a set of PVs
            if (this._num2load) return ;

            this._options.from = new Date(+this._options.to - zoom * 1000) ;

            this._interval.activate(zoom) ;
            this._loadAllTimeLines() ;
        } ;

        this._timeMoveLeft = function (xbins, dx) {
            this._timeline_change ({
                deltaZoom: -(dx ? dx / xbins : 1.) ,
                xbins: xbins
            }) ;
        } ;
        this._timeMoveRight = function (xbins, dx) {
            this._timeline_change ({
                deltaZoom: dx ? dx / xbins : 1. ,
                xbins: xbins
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
            this._loadAllTimeLines() ;
        } ;

        this._y_range_lock = {
            
        } ;

        this._is_rendered = false ;
        this.run = function () {
            
            // make sure all windows and UI resources are properly
            // initialzied

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this._selected = $('#getdata_control > #selected > table') ;
            
            this._interval = new RadioBox (
                Interval.WINDOW_DEFS ,
                function (zoom) { _that._timeZoom(zoom) ; } ,
                {activate: Interval.minZoomIn }
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
            this._end_now = $('#end_now > button')
                .button()
                .click(function () {
                    _that._end_time_changed(new Date()) ;
                }) ;

            this._interval.display($('#interval')) ;
            this._timeSeriesPlot = new TimeSeriesPlotN ({
                x_zoom_in:      function (e) { _that._timeZoomIn   (e.xbins) ; } ,
                x_zoom_out:     function (e) { _that._timeZoomOut  (e.xbins) ; } ,
                x_move_left:    function (e) { _that._timeMoveLeft (e.xbins, e.dx) ; } ,
                x_move_right:   function (e) { _that._timeMoveRight(e.xbins, e.dx) ; } ,
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
            
            this._finder = {
                input:   $('#finder > #input > input') ,
                results: $('#finder > #results')
            } ;

            var _KEY_ENTER = 13 ,
                _KEY_ESC = 27 ;

            this._finder.input.keyup(function (e) {
                switch (e.keyCode) {
                    case _KEY_ENTER:
                        var pattern = $(this).val() ;
                        if (pattern === '') {
                            _that._finder.results.removeClass('visible') ;
                            return ;
                        }
                        _that.load_pvs(pattern) ;
                        break ;

                    case _KEY_ESC:
                        _that._finder.results.removeClass('visible') ;
                        $(this).val('') ;
                        break ;
                }
            }) ;
            this._finder.results.css('max-height', (window.innerHeight-this._finder.results.offset().top - 40)+'px') ;
            $(window).resize (function () {
                _that._finder.results.css('max-height', (window.innerHeight-_that._finder.results.offset().top - 40)+'px') ;
            }) ;

            // Load the specified PV if the one was passed as the parameter
            // to the application.
            if (this._options.pvs) {
                for (var i = 0; i < this._options.pvs.length; ++i) {
                    this._num2load++ ;
                    this.load_pvtypeinfo(this._options.pvs[i]) ;
                }
            }
        } ;
        this._disableControls = function (yes) {
            this._interval.disableAll(yes) ;
            this._end_ymd.datepicker(yes ? 'disable' : 'enable') ;
            this._end_hh .prop('disabled', yes) ;
            this._end_mm .prop('disabled', yes) ;
            this._end_ss .prop('disabled', yes) ;
            this._end_now.button(yes ? 'disable' : 'enable') ;
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
        this.load_pvs = function (pattern) {
            this._finder.results.removeClass('visible') ;
            var params = {
                pv: pattern
            } ;
            this.web_service_GET (
                "/epics/mgmt/bpl/getAllPVs" ,
                params ,
                function (data) {
                    _that.pvs = data ;
                    _that.display_pvs() ;
                }
            ) ;
        } ;
        this._pvs_dict = null ;
        function _add_pvpath_to_dict (dict, path, i) {
            if (i < path.length) {
                var comp = path[i] ;
                if (!(comp in dict)) dict[comp] = {} ;
                dict[comp] = _add_pvpath_to_dict(dict[comp], path, ++i) ;
            }
            return dict ;
        } ;
        this._make_pvs_dict = function () {
            this._pvs_dict = {} ;
            for (var i in this.pvs) {
                var pv = this.pvs[i] ;
                this._pvs_dict = _add_pvpath_to_dict(this._pvs_dict, pv.split(':'), 0) ;
            }
            for (var comp in this._pvs_dict) {
                console.log(comp) ;
            }
        } ;
        this.display_pvs = function () {
            $('#loaded').text('getAllPVs ('+this.pvs.length+')') ;
            if (!this.pvs.length) return ;

            this._finder.results.addClass('visible') ;
            this._finder.results.html (
                _.reduce (
                    this.pvs ,
                    function (html, pv) {
                        return html += '<div class="pvname">'+pv+'</div>';
                    } ,
                    ''
                )+'<div class="pvname_endoflist"></div>'
            ) ;
            var pvname_elements = this._finder.results.children('.pvname') ;
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
                _that._finder.results.removeClass('visible') ;
                console.log('display_pvs - new list of PVs:', _that._options.pvs) ;
            }) ;
            
            // TODO: This is for debugging purposes.
            this._make_pvs_dict() ;
        } ;
        this.pvtypeinfo = {} ;
        this.load_pvtypeinfo = function (pvname) {
            console.log('load_pvtypeinfo - num2load:', this._num2load) ;
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
                    _that._addEntryToSelected(pvname) ;
                }
            ) ;
        } ;
        var _DEFAULT_PLOT_COLOR = ['#0071bc', '#983352', '#277650', '#333676', '#AA5939'] ;
        this._colorCounter = 0 ;
        this._getNextColor = function () {
            return _DEFAULT_PLOT_COLOR[this._colorCounter++ % _DEFAULT_PLOT_COLOR.length] ;
        } ;
        this._colors = {} ;
        this._selectedPVs = {} ;
        this._addEntryToSelected = function (pvname) {
            this._colors[pvname] = this._getNextColor() ;
            this._selected.children('tbody').append (
'<tr id="'+pvname+'" > ' +
  '<td><button name="delete" class="control-button-important" >x</button></td> ' +
  '<td><input  name="plot"   type="checkbox" checked="checked" /></td> ' +
  '<td><div    name="color"  style="width: 12px; height:12px; background-color: '+this._colors[pvname]+';" >&nbsp;</div></td> ' +
  '<td>'+pvname+'</td> ' +
  '<td>'+this.pvtypeinfo[pvname].extraFields.RTYP+'</td> ' +
  '<td> ' +
    '<select   name="method" > ' +
      '<option val=""        >raw</option> ' +
      '<option val="average" >average</option> ' +
      '<option val="count"   >count</option> ' +
    '</select> ' +
  '</td> ' +
  '<td name="bins" ></td> ' +
  '<td> ' +
    '<select   name="x-scale" > ' +
      '<option val="linear" >linear</option> ' +
      '<option val="log10"  >log10</option> ' +
      '<option val="log2"   >log2</option> ' +
    '</select> ' +
  '</td> ' +
'</tr> '
            ) ;
            this._selectedPVs[pvname] = this._selected.children('tbody').find('tr[id="'+pvname+'"]') ;
            this._selectedPVs[pvname].find('button[name="delete"]').button().click(function () {
                var tr = $(this).closest('tr') ;
                var pvname = tr.prop('id') ;
                _that._removeEntryFromSelected(pvname) ;
            }) ;
        } ;
        this._removeEntryFromSelected = function (pvname) {
            console.log('_removeEntryFromSelected: '+pvname) ;
            this._selectedPVs[pvname].remove() ;
            delete this._selectedPVs[pvname] ;
            delete this.pvtypeinfo[pvname] ;
            console.log('_removeEntryFromSelected:', this.pvtypeinfo) ;
            delete this.pvdata[pvname] ;
            this._options.pvs = _.filter(this._options.pvs, function (pv) { return pv !== pvname ; }) ;
            console.log('_removeEntryFromSelected:', this._options.pvs) ;
            this._loadAllTimeLines() ;
        } ;
        this._loadAllTimeLines = function (xbins) {
            this._num2load = this._options.pvs.length ;
            if (this._num2load)
                this._disableControls(true) ;
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

            // Make sure the previous value is always set to someting.
            // Keep refreshing it after each call with a valid parameter
            // of 'xbin'.
            if (!this._last_xbins) { this._last_xbins = 1024 ;}
            var xbins_best_guess = xbins ? xbins : this._last_xbins ;
            if (xbins_best_guess) {
                if (samplesInDelta > 2 * xbins_best_guess) {
                    var binFactor = Math.round(delta_sec / xbins_best_guess) ;
                    pv_fetch_method = 'mean_'+binFactor+'('+pvname+')' ;
                }
            }
            this._last_xbins = xbins_best_guess ;   // refresh

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
                        _that._disableControls(false)
                    }
                } ,
                function (msg) {
                    console.log('load_timeline: error:', msg) ;
                    _that._num2load-- ;
                    if (!_that._num2load) {
                        _that._disableControls(false)
                    }
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

                console.log('display_timeline - pvname', pvname) ;

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
                    points: points ,
                    color: this._colors[pvname]
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