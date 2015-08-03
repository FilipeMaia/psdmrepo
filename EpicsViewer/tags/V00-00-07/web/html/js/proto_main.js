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
    'webfwk/Class' ,
    'EpicsViewer/TimeSeriesPlotN' ,
    'EpicsViewer/Definitions' ,
    'EpicsViewer/Interval' ,
    'EpicsViewer/WebService' ,
    'EpicsViewer/Finder' ,
    'EpicsViewer/Display' ,
    'EpicsViewer/DisplaySelector' ,
    'EpicsViewer/DataTableDisplay' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'jquery.resize', 'jquery.mousewheel', 'underscore'] ,

function (
    cssloader ,
    Class ,
    TimeSeriesPlotN ,
    Definitions ,
    Interval ,
    WebService ,
    Finder ,
    Display ,
    DisplaySelector, 
    DataTableDisplay) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    function DummyPlot (parent, message) {

        Display.call(this) ;

        this._parent = parent ;
        this._message = message ;

        this.on_activate   = function () { this._display() ; } ;
        this.on_deactivate = function () { } ;
        this.on_resize     = function () { } ;

        this._isRendered = false ;
        this.render = function () {
            if (this._isRendered) return ;
            this._isRendered = true ;
            this._display() ;
        } ;
        this._display = function () {
            if (!this._isRendered) return ;
            if (!this.active) return ;
            this.container.html(this._message) ;
        } ;
    }
    Class.define_class(DummyPlot, Display, {}, {}) ;

    function EpicsViewer (pvs) {

        var _that = this ;

        this._options = {
            pvs: pvs ? pvs : []
        } ;

        this._pvfinder = null ;         // UI for searching PVs to be included into the work set
        this._selected = null ;         // the current workset table
        this._interval = null ;         // the current timeine interval management

        // displays (plots)
        this._displaySelector = null ;

        // total number of PVs to be loaded
        this._num2load = 0 ;

        // range locking for PVs
        this._y_range_lock = {} ;

        // rendering is done only once
        this._is_rendered = false ;

        this.run = function () {
            
            // make sure all windows and UI resources are properly
            // initialzied

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            $('#subtitle').click(function () {
                alert('Configuration option is yeat to be implemented') ;
            }) ;

            this._pvfinder = new Finder ($('#finder'), {
                on_select: function (pvname) {
                    if (_.indexOf(_that._options.pvs, pvname) === -1) {
                        _that._options.pvs.push(pvname) ;
                        _that._num2load++ ;
                        _that.load_pvtypeinfo(pvname) ;
                    }
                }
            }) ;
            this._selected = $('#getdata_control > #selected > table') ;
            this._selected.children('thead').children('tr:first-child').click(function () {
                var tbody = _that._selected.children('tbody') ;
                if (tbody.hasClass   ('selected-tbody-visible')) {
                    tbody.removeClass('selected-tbody-visible').addClass('selected-tbody-hidden') ;
                } else {
                    tbody.removeClass('selected-tbody-hidden') .addClass('selected-tbody-visible') ;
                }
            }) ;

            this._interval = new Interval.Interval ({
                changes_allowed: function () {
                    // no timeline changes while loading a set of PVs
                    return !this._num2load ;
                } ,
                on_change: function (xbins) {
                    _that._loadAllTimeLines(xbins) ;
                }
            }) ;
            this._displaySelector = new DisplaySelector($('#display') , [
                {   id:     "timeseries" ,
                    name:   "T<sub>series</sub>" ,
                    descr:  "Time series plots for PVs and functions" ,
                    widget: new TimeSeriesPlotN ({
                        x_zoom_in:      function (e) { _that._interval.zoomIn   (e.xbins) ; } ,
                        x_zoom_out:     function (e) { _that._interval.zoomOut  (e.xbins) ; } ,
                        x_move_left:    function (e) { _that._interval.moveLeft (e.xbins, e.dx) ; } ,
                        x_move_right:   function (e) { _that._interval.moveRight(e.xbins, e.dx) ; } ,
                        y_range_change: function (name, yRange) {
                            if (yRange) {
                                _that._y_range_lock[name] = {
                                    min: yRange.min ,
                                    max: yRange.max
                                } ;
                            } else {
                                if (_that._y_range_lock[name]) {
                                    delete _that._y_range_lock[name] ;
                                }
                            }
                        } ,
                        y_toggle_scale: function (name) {
                            switch (_that._scales[name]) {
                                case 'linear': _that._scales[name] = 'log10' ;  break ;
                                case 'log10' : _that._scales[name] = 'linear' ; break ;
                            }
                            _that._selectedPVs[name].find('select[name="scale"]').val(_that._scales[name]) ;
                            _that._loadAllTimeLines() ;
                        } ,
                        ruler_change: function (values) {
                            for (var pvname in values) {
                                var v = values[pvname] ;
                                if (_.isUndefined(v)) continue ;
                                var msec = Math.floor(1000. * v[0]) ,
                                    t = new Date(msec) ;
                                _that._selectedPVs[pvname].children('td.time') .html(Interval.time2htmlLocal(t)) ;
                                _that._selectedPVs[pvname].children('td.value').text(v[1]) ;
                            }
                        } ,
                        download_requested: function (dataURL) {
                            // Change MIME type to trick the browser to download
                            // the file instead of displaying it.
//                            dataURL = dataURL.replace(/^data:image\/[^;]*/, 'data:application/octet-stream');

                            // In addition to <a>'s "download" attribute, you can
                            // define HTTP-style headers.
//                            dataURL = dataURL.replace(/^data:application\/octet-stream/, 'data:application/octet-stream;headers=Content-Disposition%3A%20attachment%3B%20filename=Canvas.png');

                            window.open(dataURL, '_blank') ;
                        } ,
                        download_allowed: function () {
                            return !_that._interval.inAutoTrackMode()
                        }
                    })} ,

                {   id:     "waveform" ,
                    name:   "W<sub>form</sub>" ,
                    descr:  "Waveform plots for PVs and functions" ,
                    widget: new DummyPlot(this, 'Waveform')} ,

                {   id:     "correlation" ,
                    name:   "C<sub>plot</sub>" ,
                    descr:  "Correlation plots for select PVs and functions" ,
                    widget: new DummyPlot(this, 'Correlation plot')} ,

                {   id:     "histogram" ,
                    name:   "H-gram" ,
                    descr:  "Histograms for all relevant PVs and functions" ,
                    widget: new DummyPlot(this, 'Histograms')} ,

                {   id:     "data" ,
                    name:   "Data" ,
                    descr:  "Detailed information on PVs, functions and plots, \n" +
                            "including tabular representation of data points" ,
                    widget: new DataTableDisplay()}
            ]) ;

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
            this._interval.disable(yes) ;
        } ;

        this.pvtypeinfo = {} ;
        this.load_pvtypeinfo = function (pvname) {
            WebService.GET (
                "/epics/mgmt/bpl/getPVTypeInfo" ,
                {pv: pvname} ,
                function (data) {
                    if (data.extraFields.RTYP === 'waveform') {
                        // ignoring waveforms for now
                        if (_.indexOf(_that._options.pvs, pvname) !== -1) {
                            delete _that._options.pvs[pvname] ;
                        }
                        return ;
                    }
                    _that.pvtypeinfo[pvname] = data ;
                    _that._addEntryToSelected(pvname) ;
                    _that.load_timeline(pvname) ;
                }
            ) ;
        } ;
        var _DEFAULT_PLOT_COLOR = ['#0071bc', '#983352', '#277650', '#333676', '#AA5939'] ;
        this._colorCounter = 0 ;
        this._getNextColor = function () {
            return _DEFAULT_PLOT_COLOR[this._colorCounter++ % _DEFAULT_PLOT_COLOR.length] ;
        } ;
        this._plot = {} ;
        this._colors = {} ;
        this._processing = {} ;
        this._scales = {} ;
        this._selectedPVs = {} ;
        this._addEntryToSelected = function (pvname) {
            this._plot[pvname] = true ;
            this._colors[pvname] = this._getNextColor() ;
            this._processing[pvname] = '' ;
            this._scales[pvname] = 'linear' ;
            var html =
'<tr id="'+pvname+'" > ' +
  '<td><button name="delete" class="control-button-important" >x</button></td> ' +
  '<td><input  name="plot"   type="checkbox" checked="checked" /></td> ' +
  '<td class="pvname" >' +
    '<div style="float:left; width: 12px; height:12px; background-color: '+this._colors[pvname]+';" >&nbsp;</div> ' +
    '<div style="float:left; margin-left:4px;" >' + pvname + '</div> ' +
    '<div style="clear:both;" ></div> ' +
  '</td> ' +
  '<td>'+this.pvtypeinfo[pvname].extraFields.RTYP+'</td> ' +
  '<td>'+this.pvtypeinfo[pvname].units+'</td> ' +
  '<td> ' +
    '<select   name="processing" > ' +
      '<option val=""       ></option> ' + _.reduce(Definitions.PROCESSING_OPTIONS, function (html, p) { return html +=
      '<option val="'+p.operator+'" >'+p.operator+'</option> ' ; }, '') +
    '</select> ' +
    '</select> ' +
  '</td> ' +
  '<td> ' +
    '<select name="scale" > ' +
      '<option val="linear" >linear</option> ' +
      '<option val="log10"  >log10</option> ' +
    '</select> ' +
  '</td> ' +
  '<td class="time" ></td> ' +
  '<td class="value" ></td> ' +
'</tr> ' ;
            this._selected.children('tbody').append(html) ;
            this._selectedPVs[pvname] = this._selected.children('tbody').find('tr[id="'+pvname+'"]') ;
            this._selectedPVs[pvname].children('td.pvname')
                .mouseover(function () {
                    var tr = $(this).closest('tr') ;
                    var pvname = tr.prop('id') ;
                    _that._displaySelector.get('timeseries').highlight(pvname, true) ;
                })
                .mouseout(function () {
                    var tr = $(this).closest('tr') ;
                    var pvname = tr.prop('id') ;
                    _that._displaySelector.get('timeseries').highlight(pvname, false) ;
                })
            ;
            this._selectedPVs[pvname].find('button[name="delete"]').button().click(function () {
                var tr = $(this).closest('tr') ;
                var pvname = tr.prop('id') ;
                _that._removeEntryFromSelected(pvname) ;
            }) ;
            this._selectedPVs[pvname].find('input[name="plot"]').change(function () {
                var tr = $(this).closest('tr') ;
                var pvname = tr.prop('id') ;
                _that._plot[pvname] = $(this).prop('checked') ? true : false ;
                _that.display_timeline() ;
            }) ;
            this._selectedPVs[pvname].find('select[name="processing"]').change(function () {
                var tr = $(this).closest('tr') ;
                var pvname = tr.prop('id') ;
                _that._processing[pvname] = $(this).val() ;
                _that._loadAllTimeLines() ;
            }) ;
            this._selectedPVs[pvname].find('select[name="scale"]').change(function () {
                var tr = $(this).closest('tr') ;
                var pvname = tr.prop('id') ;
                _that._scales[pvname] = $(this).val() ;
                //delete _that._y_range_lock[pvname] ;
                _that._loadAllTimeLines() ;
            }).prop('disabled', true) ;
        } ;
        this._removeEntryFromSelected = function (pvname) {
            delete this._plot[pvname] ;
            delete this._colors[pvname] ;
            delete this._scales[pvname] ;
            delete this._processing[pvname] ;
            this._selectedPVs[pvname].remove() ;
            delete this._selectedPVs[pvname] ;
            delete this.pvtypeinfo[pvname] ;
            delete this.pvdata[pvname] ;
            this._options.pvs = _.filter(this._options.pvs, function (pv) { return pv !== pvname ; }) ;
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

            // Disable controls while loading
            this._selectedPVs[pvname].find('select[name="scale"]').prop('disabled', true) ;

            // data subchannel for the PV
            var pvname_archiveFields = pvname  ;

            // Make sure the previous value is always set to something.
            // Keep refreshing it after each call with a valid parameter
            // of 'xbin'.
            if (!this._last_xbins) { this._last_xbins = 1024 ;}
            var xbins_best_guess = xbins ? xbins : this._last_xbins ;
            this._last_xbins = xbins_best_guess ;


            // Compute the bin aggregation factor based on the sampling
            // period of a PV.
            //
            // TODO: this algorithm is based on assumtion which may not always
            // hold to the reality that PVs get always updated at the specified
            // sampling rate.
            // The actual sampling rate definition may vary depending on
            // the sampling method:
            //
            //   'MONITOR' - monitoring status changes (frequency is limited by
            //               the sampling period)
            //   'SCAN'    - fixed frequency (exactly sampling period)
            //
            // Besides, PVs may not be archived for certain perionds of time.
            // One way to deal with that would be to make a separate request
            // to the backend using some smaller range (for speed) to "calibrate"
            // the actual update frequency of a PV.

            var guessedBinFactor = (function () {
                var delta_sec = Math.abs(_that._interval.to - _that._interval.from) / 1000 ;
                var samplingPeriod_sec = +_that.pvtypeinfo[pvname].samplingPeriod ;
                var samplesInDelta = samplingPeriod_sec ? Math.round(delta_sec / samplingPeriod_sec) : 1 ;

                return samplesInDelta > 2 * _that._last_xbins ?
                    Math.round(delta_sec / _that._last_xbins) :
                    1 ;
            })() ;

            // Deduce the processing method based on user input (if any),
            // or use the default method.

            var pv_fetch_method ;

            var operator = this._processing[pvname] ;
            switch (operator) {
                case '':
                    // Apply the avarage bin aggregation function if the current time
                    // range includes too many measurements.
                    pv_fetch_method = guessedBinFactor > 1 ?
                        'mean_' + guessedBinFactor + '(' + pvname_archiveFields + ')' :
                        pvname_archiveFields ;
                    break ;

                case 'raw':
                    pv_fetch_method = pvname_archiveFields ;
                    break ;

                default:
                    // Apply a bin aggregation function requested by a user.
                    pv_fetch_method = operator + '_' + guessedBinFactor + '(' + pvname_archiveFields + ')' ;
                    break ;
            }
            WebService.GET (
                "/epics/retrieval/data/getData.json" ,
                {   pv:   pv_fetch_method ,
                    from: this._interval.from.toISOString() ,
                    to:   this._interval.to.toISOString() ,
                    fetchLatestMetadata: true
                }  ,
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
        } ;
        this.display_timeline = function () {

            var num_pvs = 0 ;
            for (var pvname in this.pvtypeinfo) {

                // skip plots which shouldn't be displayed
                if (!this._plot[pvname]) continue ;

                ++num_pvs;
            }
            if (!num_pvs) {
                this._displaySelector.get('timeseries').reset() ;
                return ;
            }

            var x_range = {
                min: this._interval.from / 1000. ,   // msec -> secs.*
                max: this._interval.to   / 1000.     // msec -> secs.*
            } ;

            // Extract data to plot from the loaded data object

            var many_series = [] ;
            for (var pvname in this.pvtypeinfo) {

                // skip plots which shouldn't be displayed
                if (!this._plot[pvname]) continue ;

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

                if (y_min < 0) {
                    // falback to the linear mode for negative values
                    this._selectedPVs[pvname].find('select[name="scale"]').prop('disabled', true).val('linear') ;
                    this._scales[pvname] = 'linear' ;
                    delete this._y_range_lock[pvname] ;
                } else {
                    // otherwise respect any choice of a user
                    this._selectedPVs[pvname].find('select[name="scale"]').prop('disabled', false) ;
                }
                many_series.push({
                    name: pvname ,
                    yRange: {
                        min: y_min ,
                        max: y_max
                    } ,
                    yLockedRange: this._y_range_lock[pvname] ? this._y_range_lock[pvname] : undefined ,
                    points: points ,
                    color: this._colors[pvname] ,
                    scale: this._scales[pvname]
                }) ;
            }
    
            // Plot the points using an appropriate method
            this._displaySelector.get('timeseries').load(x_range, many_series) ;
            this._displaySelector.get('data').load(x_range, many_series) ;
        } ;
    }

    // Starting point for the application
    $(function () {
        var viewer = new EpicsViewer(window.global_options.pvs) ;
        viewer.run() ;
    }) ;

}) ;