define ([
    'webfwk/Class' ,
    'EpicsViewer/Display' ,
    'EpicsViewer/LabelGenerator' ,
    'EpicsViewer/Log10LabelGenerator'] ,

function (
    Class ,
    Display ,
    LabelGenerator ,
    Log10LabelGenerator) {


    // Plot parameteres
    //
    // TODO: consider external customization via
    //       widget's parameters.
    
    var _LABEL_FONT_SIZE =  9 ,
        _LABEL_FONT = _LABEL_FONT_SIZE + 'pt Calibri' ;

    var _LABEL_EXPONENT_FONT_SIZE = 8 ,
        _LABEL_EXPONENT_FONT = _LABEL_EXPONENT_FONT_SIZE + 'pt Calibri' ;

    var _SEREIS_NAME_FONT_SIZE = 18 ,
        _SEREIS_NAME_FONT = _SEREIS_NAME_FONT_SIZE + 'pt Calibri' ;

    var _TICK_SIZE =  4 ;

    var _PRIMARY_AXIS_COLOR = '#a0a0a0' ;
    var _AXIS_COLOR         = '#a8a8a8' ;
    var _ACTIVE_AXES_COLOR  = 'aliceblue' ;
    var _LABEL_COLOR        = '#a0a0a0' ;
    var _GRID_COLOR         = '#f0f0f0' ;
    var _ZERO_GRID_COLOR    = '#ff0000' ;
    var _DEFAULT_PLOT_COLOR = ['#0071bc', '#983352', '#277650', '#333676', '#AA5939'] ;

    var _KEY_LEFT_ARROW  = 37 ,
        _KEY_RIGHT_ARROW = 39 ,
        _KEY_UP_ARROW    = 38 ,
        _KEY_DOWN_ARROW  = 40 ;

    var _EVENT_HANDLER = [
        'x_move_left', 'x_move_right', 'x_zoom_in', 'x_zoom_out' ,
        'y_range_change' ,
        'ruler_change'
    ] ;
    
    var _LOCK_IMAGE = document.getElementById('lock') ;

    /**
     * The 2-column tabular display representing properties and their values
     *
     * USAGE:
     * 
     *   TO BE COMPLETED...
     *
     * @returns {TimeSeriesPlot}
     */
    function TimeSeriesPlotN (config) {

        var _that = this ;

        // Always call the c-tor of the base class
        Display.call(this) ;

        // Object configuration
        
        this._config = {} ;
        if (!_.isUndefined(config) && _.isObject(config)) {
            
            // register custom event handler
            for (var i in _EVENT_HANDLER) {
                var handler = _EVENT_HANDLER[i] ;
                if (_.has(config, handler)) {
                    var func = config[handler] ;
                    if (_.isFunction(func)) this._config[handler] = func ;
                }
            }
        }

        // Local handlers on operations with Y ranges

        this._report_y_range_change = function (i) {
            if (!this._config.y_range_change) return ;
            this._config.y_range_change (
                this._series[i].name ,
                {   min: this._series[i].yRange.min ,
                    max: this._series[i].yRange.max
                }
            ) ;
        } ;
        this._report_y_range_reset = function (i) {
            if (!this._config.y_range_change) return ;
            this._config.y_range_change (
                this._series[i].name
            ) ;
        } ;

        this._y_move_down = function (i, shift) {
            var series = this._series[i] ,
                yMin = series.yRange.min ,
                yMax = series.yRange.max ;
            switch (series.scale) {
                case 'linear':
                    var deltaY = shift ? shift.dy * (yMax - yMin) / shift.ybins : (yMax - yMin) / 10. ;
                    series.yRange.min -= deltaY ;
                    series.yRange.max -= deltaY ;
                    break ;
                case 'log10':
                    if (!shift || (shift.dy && Math.abs(shift.ybins / shift.dy) > 100 * series.yLabels.get().length)) {
                        series.yRange.min /= 10 ;
                        series.yRange.max /= 10 ;
                    }
                    break ;
            }
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_move_up = function (i, shift) {
            var series = this._series[i] ,
                yMin = series.yRange.min ,
                yMax = series.yRange.max ;
            switch (series.scale) {
                case 'linear':
                    var deltaY = shift ? shift.dy * (yMax - yMin) / shift.ybins : (yMax - yMin) / 10. ;
                    series.yRange.min += deltaY ;
                    series.yRange.max += deltaY ;
                    break ;
                case 'log10':
                    if (!shift || (shift.dy && Math.abs(shift.ybins / shift.dy) > 100 * series.yLabels.get().length)) {
                        series.yRange.min *= 10 ;
                        series.yRange.max *= 10 ;
                        break ;
                    }
            }
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_zoom_in = function (i) {
            var series = this._series[i] ,
                yMin = series.yRange.min ,
                yMax = series.yRange.max ;
            switch (series.scale) {
                case 'linear':
                    var deltaY = (yMax - yMin) / 10. ;
                    series.yRange.min += deltaY ;
                    series.yRange.max -= deltaY ;
                    break ;
                case 'log10':
                    // No zooming in below one order of magnitude
                    if (series.yRange.min && (series.yRange.max / series.yRange.min >= 10)) {
                        series.yRange.min *= 10 ;
                        series.yRange.max /= 10 ;
                    }
                    break ;
            }
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_zoom_out = function (i) {
            var series = this._series[i] ,
                yMin = series.yRange.min ,
                yMax = series.yRange.max ;
            switch (series.scale) {
                case 'linear':
                    var deltaY = (yMax - yMin) / 10. ;
                    series.yRange.min -= deltaY ;
                    series.yRange.max += deltaY ;
                    break ;
                case 'log10':
                    series.yRange.min /= 10 ;
                    series.yRange.max *= 10 ;
                    break ;
            }            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_reset_or_lock = function (i) {
            var series = this._series[i] ;
            if ((series.yRange.min === series.yOriginalRange.min) &&
                (series.yRange.max === series.yOriginalRange.max)) {
                this._y_lock(i) ;
                return ;
            }
            series.yRange.min = series.yOriginalRange.min ;
            series.yRange.max = series.yOriginalRange.max ;
            if (series.yLabelWidth) delete series.yLabelWidth ;
            this._display() ;
            this._report_y_range_reset(i) ;
        } ;
        this._y_lock = function (i) {
            this._y_move_up(i, {
                dy: 1 ,
                ybins: 1e9
            }) ;
        } ;
        this._reportRulerChange = function (x, y) {
            if (this._config.ruler_change) {
                var values = {} ;
                for (var sid = 0; sid < this._series.length; ++sid) {
                    var series = this._series[sid] ;
                    if (series.positions) {
                        values[series.name] = series.points[0] ;    // if there is exactly one point
                        for (var i = 0; i < series.positions.length; ++i) {
                            if (i) {
                                var xPrev = series.positions[i-1][0] ,
                                    xCurr = series.positions[i]  [0] ;
                                if (xPrev <= x && x <= xCurr) {
                                    values[series.name] = series.points[i-1] ;
                                    this._highlightRulerSelection(sid, series.positions[i-1][0], series.positions[i][0], series.positions[i-1][1]) ;
                                    break ;
                                }
                            }
                        }                        
                    }
                }
                this._config.ruler_change(values) ;
            }
        } ;

        // Rendering is done only once
        this._is_rendered = false ;

        this._canvasPlot = null ;
        this._canvasGrid = null ;
        this._cxtPlot = null ;
        this._cxtGrid = null ;
        this._geom = {} ;
        this._CANVAS_WIDTH  = 0 ;
        this._CANVAS_HEIGHT = 0 ;

        // Metods implementing the Display contract
        this.on_activate   = function () { this._display() ; } ;
        this.on_deactivate = function () { } ;
        this.on_resize     = function () { this._resize() ; } ;

        this._resize = function () {
            this._canvasPlot.css('height', (window.innerHeight-this._canvasPlot.offset().top - 30)+'px') ;
            this._canvasGrid.css('height', (window.innerHeight-this._canvasGrid.offset().top - 30)+'px') ;
        } ;

        /**
         * Implement the widget rendering protocol
         *
         * @returns {undefined}
         */
        this.render = function () {
            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this.container.html (
'<canvas id="plot" ></canvas> ' + 
'<canvas id="grid" ></canvas> '
            ) ;

            this._canvasPlot = this.container.children('#plot') ;
            this._canvasGrid = this.container.children('#grid') ;

            // Resizing canvas's HTML element to fit into the visible window
            // and to prevent it from clipping or leaving extra space at the bottom
            // of the wndow.
            //
            // TODO: this algorithms assumes that canvas is located at the very
            //       bottom of the application's window. This may change if
            //       the application gets a different layout

            this._resize() ;
            $(window).resize(function () {
                _that._display() ;  // redisplay is needed to prevent plots
                                    // from being scaled.
            }) ;

            // Track mouse position in order to intersept and process events
            // generated by a keyboard and mouse scroll (real mouse pointer - for
            // mouse-based inputs, or touch screen zoom in/outs for tables
            // and Apple laptops).
            //
            // ATTENTION: tracking is being made on the top layer canvas only
            // whose z-index is greater than teh one of the plotting area.

            this._inCanvas = false ;
            this._canvasGrid.mouseover(function (e) {
                e.preventDefault() ;
                // This is needed for canva's operation's
                _that._inCanvas = true ;
            }) ;
            this._canvasGrid.mouseout(function (e) {
                e.preventDefault() ;
                _that._inCanvas = false ;
                _that._trackRegion() ;
                _that._clearGrid() ;
            }) ;
            this._canvasGrid.mousedown(function (e) {
                e.preventDefault() ;
                // Store the present position of the mouse for range change
                // operations performed when the mouse will be up.
                _that._mouseDownPosition = {
                    x: e.offsetX ,
                    y: e.offsetY
                };
            }) ;
            this._canvasGrid.mousemove(function (e) {
                e.preventDefault() ;

                // Skip this if in the mouse drag mode

                if (_that._mouseDownPosition) {

                    // Find a region where the mouse is in
                    var x = e.offsetX ,
                        y = e.offsetY ,
                        deltaX = x - _that._mouseDownPosition.x ,
                        deltaY = y - _that._mouseDownPosition.y ;

                    _that._mouseMoved = deltaY ;

                    if (_that._activeRegion) {
                        switch (_that._activeRegion.direction) {
                            case 'Y':
                                if (deltaY) {
                                    var sid = _that._activeRegion.position ;
                                    switch (_that._series[sid].scale) {
                                        case 'linear':
                                            var shift = {
                                                ybins: _that._geom.PLOT_HEIGHT ,
                                                dy: Math.abs(deltaY)
                                            } ;
                                            if (deltaY > 0) _that._y_move_up  (sid, shift) ;
                                            else            _that._y_move_down(sid, shift) ;
                                            break ;
                                        case 'log10':
                                            // no shift until the mouse is releases. Otherwise
                                            // there will be too many moves on the scale.
                                            break ;
                                    }
                                }
                                _that._mouseDownPosition.x = x ;
                                _that._mouseDownPosition.y = y ;
                                break ;

                            case 'X':
                                if (deltaX) {
                                    _that._drawAxisX() ;
                                    _that._clearDataPlotArea(deltaX) ;
                                    _that._drawGridY() ;
                                    _that._plotData(deltaX) ;
                                    _that._clearGrid() ;
                                }
                                break ;
                        }
                    }
                    return ;
                }

                // Find a region where the mouse is in
                var x = e.offsetX ,
                    y = e.offsetY ;

                for (var i in _that._region.X) {
                    var reg = _that._region.X[i] ;
                    if (reg.xMin < x && x < reg.xMax &&
                        reg.yMin < y && y < reg.yMax) {
                        _that._trackRegion({direction: 'X', position: +i}) ;
                        _that._drawRulers(x, y) ;
                        _that._reportRulerChange(x, y) ;
                        return ;
                    }
                }
                for (var i in _that._region.Y) {
                    var reg = _that._region.Y[i] ;
                    if (reg.xMin < x && x < reg.xMax &&
                        reg.yMin < y && y < reg.yMax) {
                        _that._trackRegion({direction: 'Y', position: +i}) ;
                        _that._drawRulers(x, y) ;
                        return ;
                    }
                }
            }) ;
            this._canvasGrid.mouseup(function (e) {
                e.preventDefault() ;

                // Find a region where the mouse is in
                var x = e.offsetX ,
                    y = e.offsetY ,
                    deltaX = x - _that._mouseDownPosition.x ,
                    deltaY = y - _that._mouseDownPosition.y ;

                if (_that._activeRegion) {
                    switch (_that._activeRegion.direction) {
                        case 'X':
                            if (deltaX) {
                                var shift = {
                                    xbins: _that._geom.PLOT_WIDTH ,
                                    dx: Math.abs(deltaX)
                                } ;
                                if (deltaX > 0) {
                                    if (_that._config.x_move_left)
                                        _that._config.x_move_left (shift) ;
                                } else {
                                    if (_that._config.x_move_right)
                                        _that._config.x_move_right (shift) ;
                                }
                            }
                            break ;

                        case 'Y':
                            var sid = _that._activeRegion.position ;

                            if (deltaY || _that._mouseMoved) {
                                var shift = undefined ;
                                switch (_that._series[sid].scale) {
                                    case 'linear':
                                        shift = {
                                            ybins: _that._geom.PLOT_HEIGHT ,
                                            dy: Math.abs(deltaY)
                                        } ;
                                        break ;
                                    case 'log10':
                                        // using the default shift in the direction of
                                        // the last mouse move.
                                        // ATTENTION: This logic may produce some confusingresults
                                        // in case if a users moves teh mouse in opposite directons
                                        // before releasing it.
                                        deltaY = _that._mouseMoved ;
                                        break ;
                                }
                                if (deltaY > 0) _that._y_move_up  (sid, shift) ;
                                else            _that._y_move_down(sid, shift) ;
                            }
                            if (!_that._mouseMoved) _that._y_reset_or_lock(sid) ;
                            break ;
                    }
                }

                // Always reset this
                _that._mouseDownPosition = null ;
                _that._mouseMoved = 0 ;
            }) ;
            this._canvasGrid.mousewheel(function (e) {

                // TODO: These are some ideas on how to do the so called "zoom math"
                //
                //   http://stackoverflow.com/questions/2916081/zoom-in-on-a-point-using-scale-and-translate
                //   http://stackoverflow.com/questions/6775168/zooming-with-canvas

                e.preventDefault() ;
                if (_that._activeRegion) {
                    if (e.deltaY > 0) {
                        switch (_that._activeRegion.direction) {
                            case 'X':
                                if (_that._config.x_zoom_in)
                                    _that._config.x_zoom_in({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;
                            case 'Y':
                                if (_that._y_zoom_in)
                                    _that._y_zoom_in(_that._activeRegion.position) ;
                                break ;
                        }
                    } else {
                        switch (_that._activeRegion.direction) {
                            case 'X':
                                if (_that._config.x_zoom_out)
                                    _that._config.x_zoom_out({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;
                            case 'Y':
                                if (_that._y_zoom_out)
                                    _that._y_zoom_out(_that._activeRegion.position) ;
                                break ;
                        }
                    }
                }
            }) ;
            $(document).keydown(function (e) {

                // Intercepting _specific_ keyboard events and _only_ when
                // the mouse pointer is within the canvas. This won't affect
                // other operations.

                if (_that._inCanvas) {
                    switch (e.keyCode) {

                        case _KEY_LEFT_ARROW:
                            e.preventDefault() ;
                            if (_that._activeRegion) {
                                switch (_that._activeRegion.direction) {
                                    case 'X':
                                        if (_that._config.x_move_left)
                                            _that._config.x_move_left({xbins: _that._geom.PLOT_WIDTH}) ;
                                        break ;
                                    case 'Y':
                                        if (_that._y_zoom_out)
                                            _that._y_zoom_out(_that._activeRegion.position) ;
                                        break ;
                                }
                            }
                            break ;

                        case _KEY_RIGHT_ARROW:
                            e.preventDefault() ;
                            if (_that._activeRegion) {
                                switch (_that._activeRegion.direction) {
                                    case 'X':
                                        if (_that._config.x_move_right)
                                            _that._config.x_move_right({xbins: _that._geom.PLOT_WIDTH}) ;
                                        break ;
                                    case 'Y':
                                        if (_that._y_zoom_in)
                                            _that._y_zoom_in(_that._activeRegion.position) ;
                                        break ;
                                }
                            }
                            break ;

                        case _KEY_UP_ARROW:
                            e.preventDefault() ;
                            if (_that._activeRegion) {
                                switch (_that._activeRegion.direction) {
                                    case 'X':
                                        if (_that._config.x_zoom_in)
                                            _that._config.x_zoom_in({xbins: _that._geom.PLOT_WIDTH}) ;
                                        break ;
                                    case 'Y':
                                        if (_that._y_move_up)
                                            _that._y_move_up(_that._activeRegion.position) ;
                                        break ;
                                }
                            }
                            break ;

                        case _KEY_DOWN_ARROW:
                            e.preventDefault() ;
                            if (_that._activeRegion) {
                                switch (_that._activeRegion.direction) {
                                    case 'X':
                                        if (_that._config.x_zoom_out)
                                            _that._config.x_zoom_out({xbins: _that._geom.PLOT_WIDTH}) ;
                                        break ;
                                    case 'Y':
                                        if (_that._y_move_down)
                                            _that._y_move_down(_that._activeRegion.position) ;
                                        break ;
                                }
                            }
                            break ;
                    }
                }
            }) ;

            // Initialize the plot with default settings. Also draw axes
            // just to show a user the geometry of the canvas as we don't have
            // any real data at this point.

            this.reset() ;

            this._initDrawingContext() ;
            this._prepareAxes() ;
        } ;

        this._xRange = null  ;
        this._series = null ;
        this._region = null ;
        this._activeRegion = null ;

        this.reset = function () {
            this._xRange = {
                min: (new Date()) / 1000. - 7 * 24 * 3600. ,    // 7 days ago
                max: (new Date()) / 1000.                       // right now
            } ;
            this._series = [{
                name: '' ,
                yLabels: null ,
                yRange: {
                    min: 0 ,
                    max: 1.
                } ,
                yOriginalRange: {
                    min: 0 ,
                    max: 1.
                } ,
                points: [] ,
                color: _DEFAULT_PLOT_COLOR[0] ,
                scale: 'linear'
            }] ;
            this._region = {
                X: [] ,
                Y: []
            } ;
            this._activeRegion = null ;
        } ;

        this._trackRegion = function (region) {

            // Swap the region status before applying the algorithm
            var prev_activeRegion = this._activeRegion ;
            this._activeRegion = region ;

            if (prev_activeRegion) {

                switch (prev_activeRegion.direction) {
                    case 'X':
                        if (region) {
                            switch (region.direction) {
                                case 'Y':
                                    this._clearDataPlotArea() ;
                                    this._drawAxisX() ;
                                    this._drawAxisYofSeries(region.position, true) ;
                                    this._drawGridY() ;
                                    this._plotData() ;
                                    break ;
                            }
                        } else {
                            this._drawAxisX() ;
                        }
                        break ;

                    case 'Y':
                        if (region) {
                            switch (region.direction) {
                                case 'X':
                                    this._drawAxisYofSeries(prev_activeRegion.position) ;
                                    this._drawAxisX() ;
                                    this._drawGridY() ;
                                    this._plotData() ;
                                    break ;

                                case 'Y':
                                    if (prev_activeRegion.position !== region.position) {
                                        this._clearDataPlotArea() ;
                                        this._drawAxisYofSeries(prev_activeRegion.position) ;
                                        this._drawAxisYofSeries(region.position, true) ;
                                        this._drawGridY() ;
                                        this._plotData() ;
                                    }
                                    break ;
                            }
                        } else {
                            this._drawAxisYofSeries(prev_activeRegion.position) ;
                        }
                        break ;
                }

            } else {
                if (region) {
                    switch (region.direction) {
                        case 'X':
                            this._drawAxisX() ;
                            break ;
                        case 'Y':
                            this._drawAxisYofSeries(region.position, true) ;
                            break ;
                    }
                }
            }
        } ;
        this.load = function (xRange, many_series) {
            this._xRange = xRange ;
            this._series = [] ;
            for (var i in many_series) {
                var s = {
                    name: many_series[i].name ,
                    yLabels: null ,
                    yRange: many_series[i].yLockedRange ? many_series[i].yLockedRange : {
                        min: many_series[i].yRange.min ,
                        max: many_series[i].yRange.max
                    } ,
                    yOriginalRange: {
                        min: many_series[i].yRange.min ,
                        max: many_series[i].yRange.max
                    } ,
                    points: many_series[i].points ,
                    color: _.isUndefined(many_series[i].color) ? _DEFAULT_PLOT_COLOR[i % _DEFAULT_PLOT_COLOR.length] : many_series[i].color ,
                    scale: _.isUndefined(many_series[i].scale) ? 'linear' : many_series[i].scale
                } ;
            
                // add a little to the range to preven the lockup in
                // the widget implementation.
                //
                // TODO: consider a more reliable way of sanitizing
                //       the implementation.
                if (s.yRange.min === s.yRange.max) {
                    console.log('TimeSeries.load() trap yRange.min === yRange.max', s.yRange.max) ;
                    s.yRange.max += !s.yRange.max ? 
                        1. :
                        Math.abs(s.yRange.max / 2) ;
                    s.yOriginalRange.max = s.yRange.max ;
                }
                this._series.push(s) ;
            }
            this._display() ;
        } ;
        this._display = function () {

            // Display is not active - no display
            if (!this.active) return ;

            // No data or empty data - no display
            if (!this._series.length) return ;

            this._resize() ;

            this._initDrawingContext() ;
            this._prepareAxes() ;
            this._plotData() ;
        } ;
        this.highlight = function (pvname, on) {
            if (on) {
                for (var sid = 0; sid < this._series.length; ++sid) {
                    if (pvname === this._series[sid].name) {
                        this._activeRegion = {
                            direction: 'Y' ,
                            position: sid
                        } ;
                        this._display() ;
                        return ;
                        this._trackRegion({
                            direction: 'Y' ,
                            position: sid
                        }) ;
                        return ;
                    }
                }
            } else {
                this._activeRegion = null ;
                this._display() ;
            }
        } ;
        this._initDrawingContext = function () {

            // IMPORTANT: Make sure that canvas' attributes always stay
            // in sync with the HTML element's CSS geometry _BEFORE_ obtaining
            // the drawing context. In this case we'll be guaranteed that we
            // always have the correct 1:1 mapping between window's pixels
            // and canvas' pixels. Otherwise an unpleasant scaling will
            // occure each time the window gets resized.

            this._CANVAS_WIDTH  = this._canvasPlot.css('width') .substr(0, this._canvasPlot.css('width') .length - 2) ,  // remove 'px'
            this._CANVAS_HEIGHT = this._canvasPlot.css('height').substr(0, this._canvasPlot.css('height').length - 2) ;  // remove 'px'

            this._canvasPlot.attr('width',  this._CANVAS_WIDTH) ;
            this._canvasPlot.attr('height', this._CANVAS_HEIGHT) ;

            this._canvasGrid.attr('width',  this._CANVAS_WIDTH) ;
            this._canvasGrid.attr('height', this._CANVAS_HEIGHT) ;

            // Get and tune up the 2D drawing context _AFTER_ resynching
            // canvas' geometries. Also note the half-step translation
            // which is needed to prevent line blurring.
            this._cxtPlot = this._canvasPlot.get(0).getContext('2d') ;
            this._cxtPlot.translate(0.5, 0.5) ;     

            this._cxtGrid = this._canvasGrid.get(0).getContext('2d') ;
            this._cxtGrid.translate(0.5, 0.5) ;     

            this._initGeometry() ;
            this._clearPlot() ;
        } ;

        this._initGeometry = function () {    

            // Calculate total space in X dimension which would be required to
            // draw all Y labels on vertical axes.

            this._geom.X_MIN = 2 * _LABEL_FONT_SIZE ;
            this._geom.PLOT_X_MIN = this._geom.X_MIN ;
            this._geom.PLOT_LABEL_X = [] ;

            for (var i = 0; i < this._series.length; ++i) {
                
                var series = this._series[i] ;

                // Set up the label generators of all series for the Y axis here because
                // we need to calculate the left offset for the plot.

                // Calculate a desired number of ticks based on the size of
                // the canvas and font size of the labels.

                var y_labels_generator ;
                switch (series.scale) {
                    case 'linear':
                        y_labels_generator = new LabelGenerator.base10() ;
                        break ;
                    case 'log10':
                        y_labels_generator = new Log10LabelGenerator() ;
                        break ;
                }
                series.yLabels = y_labels_generator.search (
                    series.yRange.min ,
                    series.yRange.max ,
                    Math.max (
                        2 ,         // at least
                        Math.min (
                            Math.floor(this._CANVAS_HEIGHT / (4 * _LABEL_FONT_SIZE)) ,
                            20      // at most
                        )
                    )
                ) ;
            
                // Calculate the maximum width of the formatted labels
                // and use this to define the left offset of the plot
                // so that the labels would fit in there.

                var yLabelWidth = 0 ;
                var yFormattedLabels = series.yLabels.pretty_formatted() ;
                console.log(yFormattedLabels) ;
                
                for (var j in yFormattedLabels) {
                    var label = yFormattedLabels[j] ;
                    switch (series.scale) {
                        case 'linear':
                            yLabelWidth = Math.max (
                                yLabelWidth ,
                                this._cxtPlot.measureText(label.text).width) ;
                            break ;
                        case 'log10':
                            yLabelWidth = Math.max (
                                yLabelWidth ,
                                this._cxtPlot.measureText(label.base_text + ' ' + label.exponent_text).width) ;
                            break ;
                    }
                }
                // Cache and use the widest width to avoid jittering of
                // the plot width when changing Y ranges.
                if (series.yLabelWidth && series.yLabelWidth > yLabelWidth) {        yLabelWidth = series.yLabelWidth ; }
                else                                                        { series.yLabelWidth =        yLabelWidth ; }

                this._geom.PLOT_X_MIN += yLabelWidth + 4 * _TICK_SIZE ;
                this._geom.PLOT_LABEL_X.push(this._geom.PLOT_X_MIN) ;
            }

            this._geom.PLOT_X_MAX  = this._CANVAS_WIDTH   - 2 * _LABEL_FONT_SIZE ;
            this._geom.PLOT_Y_MIN  =                       25 ;
            this._geom.PLOT_Y_MAX  = this._CANVAS_HEIGHT - 60 ;
            this._geom.PLOT_WIDTH  = this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN ;
            this._geom.PLOT_HEIGHT = this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN ;

            // Actual limits for plotting in the Y dimension for each series
            // 
            //   PLOT_Y_STEP_SIZE:  plotted interval between ticks
            //   PLOT_Y_BEGIN:      the position of the first tick

            this._geom.PLOT_Y_STEP_SIZE = [] ;
            this._geom.PLOT_Y_BEGIN = [] ;
            for (var i = 0;  i < this._series.length; ++i) {
                var series = this._series[i] ;
                var stepSize ,
                    yBegin ;
                switch (series.scale) {
                    case 'linear':
                         // two extra steps - on on top, and the other one at the bottom
                         // for the overflow content
                        stepSize = this._geom.PLOT_HEIGHT / (series.yLabels.get().length + 1) ;
                        yBegin   = this._geom.PLOT_Y_MAX - stepSize ;
                        break ;
                    case 'log10':
                        // no extra steps on ether sides - the plotted data must
                        // fit in between.
                        stepSize = this._geom.PLOT_HEIGHT / (series.yLabels.get().length - 1) ;
                        yBegin   = this._geom.PLOT_Y_MAX ;
                        break ;
                }
                this._geom.PLOT_Y_STEP_SIZE.push(stepSize) ;
                this._geom.PLOT_Y_BEGIN.push(yBegin) ;
            }
            console.log('this._geom', this._geom) ;
        } ;

        this._clearPlot = function () {
            this._cxtPlot.clearRect(0, 0, this._CANVAS_WIDTH, this._CANVAS_HEIGHT) ;
        } ;
        this._clearGrid = function () {
            this._cxtGrid.clearRect(0, 0, this._CANVAS_WIDTH, this._CANVAS_HEIGHT) ;
        } ;
        this._prepareAxes = function () {
            this._region.X = [] ;
            this._region.Y = [] ;
            this._drawAxisX() ;
            this._drawAxisY() ;
        } ;
        this._drawAxisX = function (deltaX) {

            var now_sec = (new Date()) / 1000. ;
            var xRangeMin = this._xRange ? this._xRange.min : now_sec - 7 * 24 * 3600. ,
                xRangeMax = this._xRange ? this._xRange.max : now_sec ,
                xDelta    = xRangeMax - xRangeMin ;

            // Draw/clear the area background depenidng on its present status

            var xMin = this._geom.PLOT_X_MIN ,
                xMax = this._geom.PLOT_X_MAX ,
                yMin = this._geom.PLOT_Y_MAX ,
                yMax = this._geom.PLOT_Y_MAX + 5 * _LABEL_FONT_SIZE + _TICK_SIZE ,
                width  = xMax - xMin ,
                height = yMax - yMin ;

            if (this._activeRegion && this._activeRegion.direction === 'X') {
                this._cxtPlot.fillStyle = _ACTIVE_AXES_COLOR ;
                this._cxtPlot.fillRect(xMin+1,  yMin+1, width-1, height-1) ;
            } else {
                this._cxtPlot.clearRect(xMin+1, yMin,   width-1, height) ;
            }
            var xFormattedLabels ;
            if (xDelta < 61) {
                var format = ':SS' ;
                var xDesiredTicks = 10 ;
                var xLabelsGenerator = new LabelGenerator.forSeconds () ;
                var xLabels = xLabelsGenerator.search(xRangeMin, xRangeMax, xDesiredTicks) ;
                xFormattedLabels = xLabels.empty_duplicates(xLabels.pretty_formatted_timestamps()[format]) ;
            } else if (xDelta < 10 * 60 + 1) {
                var format = 'HH:MM:SS' ;
                var xDesiredTicks = 6 ;
                var xLabelsGenerator = new LabelGenerator.forMinutes () ;
                var xLabels = xLabelsGenerator.search(xRangeMin, xRangeMax, xDesiredTicks) ;
                xFormattedLabels = xLabels.empty_duplicates(xLabels.pretty_formatted_timestamps()[format]) ;
            } else if (xDelta < 2 * 24 * 3600) {
                var format = 'HH:MM' ;
                var xLabelWidth = this._cxtPlot.measureText(format).width ;
                var xDesiredTicks = 10 * Math.max(1, Math.floor(this._geom.PLOT_WIDTH / (1.5 * xLabelWidth) / 10)) ;
                var xLabelsGenerator = new LabelGenerator.forMinutes() ;
                var xLabels = xLabelsGenerator.search(xRangeMin, xRangeMax, xDesiredTicks) ;
                xFormattedLabels = xLabels.empty_duplicates(xLabels.pretty_formatted_timestamps()[format]) ;
            } else {
                var format = 'YYYY-MM-DD' ;
                var xLabelWidth = this._cxtPlot.measureText(format).width ;
                var xDesiredTicks = 2 * Math.max(1, Math.floor(this._geom.PLOT_WIDTH / (1.5 * xLabelWidth) / 2)) ;
                var xLabelsGenerator = new LabelGenerator.forDays() ;
                var xLabels = xLabelsGenerator.search(xRangeMin, xRangeMax, xDesiredTicks) ;
                xFormattedLabels = xLabels.empty_duplicates(xLabels.pretty_formatted_timestamps()[format]) ;
            }

            var xStepSize = this._geom.PLOT_WIDTH / (xFormattedLabels.length + 1) ;

            var y = this._geom.PLOT_Y_MAX ,
                yLabelOffset = y + 2 * _LABEL_FONT_SIZE ;    // start position of labels is 2 simbols below.

            this._cxtPlot.beginPath() ;
            this._cxtPlot.font      = _LABEL_FONT ;
            this._cxtPlot.fillStyle = _LABEL_COLOR ;
            this._cxtPlot.textAlign = 'center';
            this._cxtPlot.moveTo(this._geom.PLOT_X_MIN, y) ;
            this._cxtPlot.lineTo(this._geom.PLOT_X_MAX, y) ;
          
            for (var step = 0, x = this._geom.PLOT_X_MIN + xStepSize ;  // no tick at position 0
                     step < xFormattedLabels.length ;                   // no tick at the end
                   ++step,     x += xStepSize)
            {
                this._cxtPlot.moveTo(x, y) ;
                this._cxtPlot.lineTo(x, y + _TICK_SIZE) ;
                var label = ''+xFormattedLabels[step].text ;
                // Note that the starting point of the text is
                // depends on the alignment. If the text is center aligned then
                // the central position fo the label text will be consider for text placement .
                this._cxtPlot.fillText(label, x, yLabelOffset) ;
            }
            this._cxtPlot.strokeStyle = _PRIMARY_AXIS_COLOR ;
            this._cxtPlot.lineWidth   = 1 ;
            this._cxtPlot.stroke();
            
            this._cxtPlot.beginPath() ;
            this._cxtPlot.strokeStyle = _AXIS_COLOR ;
            this._cxtPlot.moveTo(this._geom.PLOT_X_MIN, yLabelOffset + _LABEL_FONT_SIZE) ;
            this._cxtPlot.lineTo(this._geom.PLOT_X_MAX, yLabelOffset + _LABEL_FONT_SIZE) ;
            this._cxtPlot.lineWidth = 1 ;
            this._cxtPlot.stroke();

            this._cxtPlot.font      = _LABEL_FONT ;
            this._cxtPlot.fillStyle = 'black' ;
            this._cxtPlot.textAlign = 'center';

            this._cxtPlot.beginPath() ;
            
            for (var step = 0, x = this._geom.PLOT_X_MIN + xStepSize ;      // no tick at position 0
                     step < xFormattedLabels.length ;                       // no tick at the end
                   ++step,     x += xStepSize)
            {
                if (step == 25) {
                    this._cxtPlot.moveTo(x, yLabelOffset + _LABEL_FONT_SIZE) ;
                    this._cxtPlot.lineTo(x, yLabelOffset + _LABEL_FONT_SIZE + _TICK_SIZE) ;
                    // Note that the starting point of the text is
                    // depends on the alignment. If the text is center aligned then
                    // the central position fo the label text will be consider for text placement .
                    this._cxtPlot.fillText('2015-06-10', x, yLabelOffset + 3 * _LABEL_FONT_SIZE) ;
                }
            }
            this._cxtPlot.lineWidth = 1 ;
            this._cxtPlot.strokeStyle = _AXIS_COLOR ;
            this._cxtPlot.stroke();
            
            // ATTENTION: the region also include the plot area. This may change
            //            in the future.
            this._region.X[0] = {
                xMin: xMin,                  xMax: xMax ,
                yMin: this._geom.PLOT_Y_MIN, yMax: yMax
            } ;
        } ;
        this._drawAxisY = function () {
            for (var i = 0; i < this._series.length; i++) {
                this._drawAxisYofSeries(i) ;
            }
            this._drawGridY() ;
        } ;
        this._drawAxisYofSeries = function (i, active) {

            var series = this._series[i] ;

            var xMin = !i ? this._geom.X_MIN : this._geom.PLOT_LABEL_X[i-1] ,
                xMax = this._geom.PLOT_LABEL_X[i] ,
                yMin = this._geom.PLOT_Y_MIN - 2 * _LABEL_FONT_SIZE ,
                yMax = this._geom.PLOT_Y_MAX + 5 * _LABEL_FONT_SIZE + _TICK_SIZE ,
                width  = xMax - xMin ,
                height = yMax - yMin ;

            // Start position of labels is one simbol left.

            var  xLabelOffset = xMax - _LABEL_FONT_SIZE ;

            // Draw/clear the area background depenidng on its present status

            if (active) {
                this._cxtPlot.fillStyle = _ACTIVE_AXES_COLOR ;
                this._cxtPlot.fillRect(xMin+1,  1, width-1, yMax-1) ;
//                this._cxtPlot.fillRect(xMin+1,  yMin+1, width-1, height-1) ;
            } else {
                this._cxtPlot.clearRect(xMin+1, 0,   width-1, yMax) ;
//                this._cxtPlot.clearRect(xMin+1, yMin,   width-1, height) ;
            }
            
            // Draw the axes itself

            this._cxtPlot.beginPath() ;
            this._cxtPlot.lineWidth   = i == this._series.length - 1 ?                 1.25 :           1 ;
            this._cxtPlot.strokeStyle = i == this._series.length - 1 ? _PRIMARY_AXIS_COLOR : _AXIS_COLOR ;
//            this._cxtPlot.moveTo(xMax, yMin) ;
//            this._cxtPlot.lineTo(xMax, yMax) ;
            this._cxtPlot.moveTo(xMax, 0) ;
            this._cxtPlot.lineTo(xMax, yMax) ;
            this._cxtPlot.stroke();

            // Draw ticks and labels

            this._cxtPlot.beginPath() ;
            this._cxtPlot.lineWidth = 1 ;

            this._cxtPlot.fillStyle = series.color ;   // using the same color as the one of the plotted values
            this._cxtPlot.textAlign = 'right';

            var yFormattedLabels = series.yLabels.pretty_formatted() ;

            for (var step = 0, yTick = this._geom.PLOT_Y_BEGIN[i];
                     step < series.yLabels.get().length ;
                   ++step,     yTick -= this._geom.PLOT_Y_STEP_SIZE[i])
            {
                // Tick

                this._cxtPlot.moveTo (xMax - _TICK_SIZE, yTick) ;
                this._cxtPlot.lineTo (xMax,              yTick) ;

                // Label
                //
                // Note that the starting point of the text is
                // depends on the alignment. If the text is rigth aligned then
                // the position will be counted from the very right (from the vertical axis).

                switch (series.scale) {
                    case 'linear':
                        var label = yFormattedLabels[step].text ;
                        this._cxtPlot.font = _LABEL_FONT ;
                        this._cxtPlot.fillText (label, xLabelOffset, yTick + _LABEL_FONT_SIZE / 2) ;
                        break ;
                    case 'log10':
                        var label = yFormattedLabels[step] ;
                        this._cxtPlot.font = _LABEL_FONT ;
                        this._cxtPlot.fillText (label.base_text, xLabelOffset - _LABEL_FONT_SIZE / 2, yTick + _LABEL_FONT_SIZE / 2) ;
                        this._cxtPlot.font = _LABEL_EXPONENT_FONT ;
                        this._cxtPlot.fillText (label.exponent_text, xLabelOffset - _LABEL_FONT_SIZE / 2 + _LABEL_FONT_SIZE, yTick - 0.6 * _LABEL_FONT_SIZE) ;
                        break ;
                }
            }
            this._cxtPlot.stroke();

            // Put the lock flag on top of the axes to make the user
            // aware that teh dynamic range of the range is not computed automatically.

            if ((series.yRange.min !== series.yOriginalRange.min) ||
                (series.yRange.max !== series.yOriginalRange.max)) {
                this._cxtPlot.drawImage(_LOCK_IMAGE, xMin + (xMax - xMin) / 2 - 10, yMin, 20, 20);
            }

            // Remember the position of this axes region
            this._region.Y[i] = {
                xMin: xMin, xMax: xMax ,
                yMin: yMin, yMax: yMax
            } ;
        } ;

        this._drawGridY = function () {

            // Grid lines for the specifid Y range if requested. Otherwise
            // do this fo rthe last last axis.

            var sid = this._activeRegion && this._activeRegion.direction == 'Y' ? this._activeRegion.position : this._series.length - 1 ,
                series = this._series[sid] ;


            var yLabels = series.yLabels.get() ;
            for (var ystep = 0, y = this._geom.PLOT_Y_BEGIN[sid], nSteps = yLabels.length ; // no line at position 0
                     ystep < nSteps + 1 ;                                                   // one extra line on top
                   ++ystep,     y -= this._geom.PLOT_Y_STEP_SIZE[sid])
            {
                this._cxtPlot.beginPath() ;
                this._cxtPlot.lineWidth = 1 ;

                // Emphasize '0' grid line if the value at the absolute value at
                // the current grid point is less than the 'epsilon' (some small number)
                // This should cover three cases:
                //
                // - absolute zero (0.)
                // - nearly zero positive number s (very small values which are
                //   smaller than 'epsilon')
                // - negative zeroes (which are actually small negative numbers whose
                //   absolute value is smaller than 'epsilon')

                this._cxtPlot.strokeStyle = (ystep > 0) && (ystep < nSteps) && Math.abs(yLabels[ystep].value) < 10e-9 ? 
                    _ZERO_GRID_COLOR :
                    series.scale === 'log10' ? _AXIS_COLOR : _GRID_COLOR ;

                this._cxtPlot.moveTo(this._geom.PLOT_X_MIN + 2, y) ;    // +2 to avoid overwriting the axes
                this._cxtPlot.lineTo(this._geom.PLOT_X_MAX,     y) ;

                this._cxtPlot.stroke();
                
                // Plot 10 intermediate lines if in the log10 scale mode
                if (series.scale === 'log10') {
                    this._cxtPlot.beginPath() ;
                    this._cxtPlot.lineWidth = 1 ;
                    this._cxtPlot.strokeStyle = _GRID_COLOR ;
                    for (var i = 2; i <= 9 ; ++i) {
                        var yAt = y + this._geom.PLOT_Y_STEP_SIZE[sid] - Math.log10(i) * this._geom.PLOT_Y_STEP_SIZE[sid] ;
                        for (var x = this._geom.PLOT_X_MIN + 2 * _TICK_SIZE / 2, xStep = 4 * _TICK_SIZE / 2;
                                 x < this._geom.PLOT_X_MAX - 2 * _TICK_SIZE / 2 ;
                                 x += xStep) {
                            this._cxtPlot.moveTo(x - _TICK_SIZE / 2, yAt) ;
                            this._cxtPlot.lineTo(x + _TICK_SIZE / 2, yAt) ;
                        }
                        this._cxtPlot.stroke();
                    }
                }
            }

        } ;
        this._clearDataPlotArea = function (deltaX) {

            this._cxtPlot.clearRect (
                this._geom.PLOT_X_MIN+1 ,
                0 ,
                this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN - 1 ,
                this._geom.PLOT_Y_MAX - 1) ;

            if (deltaX) {
                
                // Show the "no data yet" space in the X drag mode

                this._cxtPlot.fillStyle = '#f8f8f8' ;
                this._cxtPlot.fillRect (
                    deltaX < 0 ? this._geom.PLOT_X_MAX : this._geom.PLOT_X_MIN + 1 ,
                    this._geom.PLOT_Y_MIN ,
                    deltaX ,
                    this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN - 1) ;

            }
        } ;
        this._plotData = function (deltaX) {

            for (var sid = 0; sid < this._series.length; ++sid) {
                var series = this._series[sid] ;

                this._cxtPlot.beginPath() ;
                this._cxtPlot.lineWidth = 1;
                this._cxtPlot.strokeStyle = series.color ;

                if (!(!this._activeRegion || !(this._activeRegion.direction === 'Y') || (this._activeRegion.position === sid))) {
                    this._cxtPlot.globalAlpha = 0.25 ; 
                }
                switch (series.scale) {
                    case 'linear': this._plotDataLinear(deltaX, series, sid) ; break ;
                    case 'log10':  this._plotDataLog10 (deltaX, series, sid) ; break ;
                }
                this._cxtPlot.stroke ();
                this._cxtPlot.globalAlpha = 1. ;
                
                // Show the name of the series if in the X axes selection mode
                
                if (this._activeRegion && (this._activeRegion.direction === 'Y') && (this._activeRegion.position === sid)) {
                    this._cxtPlot.font      = _SEREIS_NAME_FONT ;
                    this._cxtPlot.fillStyle = series.color ;   // using the same color as the one of the plotted values
                    this._cxtPlot.textAlign = 'right';
                    this._cxtPlot.fillText (
                        series.name ,
                        this._geom.PLOT_X_MAX ,
                        this._geom.PLOT_Y_MIN + _SEREIS_NAME_FONT_SIZE) ;
                }
            }
        } ;
        this._plotDataLinear = function (deltaX, series, sid) {

            console.log("_plotDataLinear: "+series.name+' series.yLabels.min: '+series.yLabels.min) ;

            var xDelta = this._xRange.max - this._xRange.min ,
                x0     = this._geom.PLOT_X_MIN + (deltaX               ? deltaX : 0) ,
                xmin   = this._geom.PLOT_X_MIN ,
                xmax   = this._geom.PLOT_X_MAX + (deltaX && deltaX < 0 ? deltaX : 0) ,
                yPrev  = null ;

            series.positions = [] ;

            for (var i = 0; i < series.points.length; ++i) {
                var p = series.points[i] ;

                var xVal = p[0] ,
                    yVal = p[1] ;

                var x = x0 + this._geom.PLOT_WIDTH  * ((xVal - this._xRange.min) / xDelta) ;
                if (x < xmin) continue ;
                if (x > xmax) continue ;

                var y = this._geom.PLOT_Y_BEGIN[sid] - (this._geom.PLOT_Y_STEP_SIZE[sid] / series.yLabels.step) * (yVal - series.yLabels.min) ;
                if (y > this._geom.PLOT_Y_MAX - 1) y = this._geom.PLOT_Y_MAX - 1 ;
                if (y < this._geom.PLOT_Y_MIN) y = this._geom.PLOT_Y_MIN ;
                if (_.isNull(yPrev)) {
                    this._cxtPlot.moveTo (x, y) ;
                } else {
                    this._cxtPlot.lineTo (x, yPrev) ;
                    this._cxtPlot.lineTo (x, y) ;
                }
                yPrev = y ;
                
                series.positions.push([x, y]) ;
            }
        } ;
        this._plotDataLog10 = function (deltaX,series, sid) {

            console.log("_plotDataLog10: series: "+series.name+' yLabels.min: '+series.yLabels.min+' yLabels.max: '+series.yLabels.max+' yLabels.get().length: '+series.yLabels.get().length) ;

            var xDelta = this._xRange.max - this._xRange.min ,
                x0     = this._geom.PLOT_X_MIN + (deltaX               ? deltaX : 0) ,
                xmin   = this._geom.PLOT_X_MIN ,
                xmax   = this._geom.PLOT_X_MAX + (deltaX && deltaX < 0 ? deltaX : 0) ,
                yPrev  = null ,
                epsilon = 10e-9 ;

            series.positions = [] ;

            for (var j in series.points) {
                var p = series.points[j] ;

                var xVal = p[0] ,
                    yVal = p[1] ;

                var x = x0 + this._geom.PLOT_WIDTH  * ((xVal - this._xRange.min) / xDelta) ;
                if (x < xmin) continue ;
                if (x > xmax) continue ;

                // TEMPORARY RESTRICTION: avoid displaying signal values
                // to which the log10(v) operation won't apply. This will be taken
                // care off later.
                if (yVal <= epsilon) continue ;
                var y = this._geom.PLOT_Y_BEGIN[sid] -
                    (Math.log10(yVal) - Math.log10(series.yLabels.min)) *
                    ((this._geom.PLOT_Y_STEP_SIZE[sid] * (series.yLabels.get().length - 1)) / (Math.log10(series.yLabels.max) - Math.log10(series.yLabels.min))) ;


                if (y > this._geom.PLOT_Y_MAX - 1) y = this._geom.PLOT_Y_MAX - 1 ;
                if (y < this._geom.PLOT_Y_MIN) y = this._geom.PLOT_Y_MIN ;
                if (_.isNull(yPrev)) {
                    this._cxtPlot.moveTo (x, y) ;
                } else {
                    this._cxtPlot.lineTo (x, yPrev) ;
                    this._cxtPlot.lineTo (x, y) ;
                }
                yPrev = y ;

                series.positions.push([x, y]) ;
            }
        } ;
        this._drawRulers = function (xAt, yAt) {
            this._clearGrid() ;
            if (this._activeRegion) {
                this._cxtGrid.beginPath() ;
                this._cxtGrid.lineWidth = 1;
                this._cxtGrid.strokeStyle = "maroon" ; // _AXIS_COLOR
                this._cxtGrid.globalAlpha = 0.5 ;
                switch (this._activeRegion.direction) {
                    case 'X':
                        for (var y = 2 * _TICK_SIZE, yStep = 4 * _TICK_SIZE;
                                 y < this._CANVAS_HEIGHT - 2 * _TICK_SIZE;
                                 y += yStep) {
                            this._cxtGrid.moveTo(xAt, y - _TICK_SIZE) ;
                            this._cxtGrid.lineTo(xAt, y + _TICK_SIZE) ;
                        }
                    case 'Y':
                        for (var x = 2 * _TICK_SIZE, xStep = 4 * _TICK_SIZE;
                                 x < this._CANVAS_WIDTH - 2 * _TICK_SIZE;
                                 x += xStep) {
                            this._cxtGrid.moveTo(x - _TICK_SIZE, yAt) ;
                            this._cxtGrid.lineTo(x + _TICK_SIZE, yAt) ;
                        }
                        break ;
                }
                this._cxtGrid.stroke ();
                this._cxtGrid.globalAlpha = 1. ;
            }
        } ;
        this._highlightRulerSelection = function (sid, xPrev, xCurr, y ) {
            this._cxtGrid.fillStyle = this._series[sid].color ;
            //this._cxtGrid.globalAlpha = 0.5 ;
            this._cxtGrid.fillRect(xPrev, y - 2, xCurr - xPrev, 4) ; 
        } ;
    }
    Class.define_class (TimeSeriesPlotN, Display, {}, {}) ;

    return TimeSeriesPlotN ;
}) ;
