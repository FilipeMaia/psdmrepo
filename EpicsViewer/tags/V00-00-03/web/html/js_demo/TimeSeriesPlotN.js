define ([
    'webfwk/Class', 'webfwk/Widget' ,
    'EpicsViewer/LabelGenerator', 'EpicsViewer/Log10LabelGenerator'] ,

function (
    Class, Widget ,
    LabelGenerator, Log10LabelGenerator) {


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
        'y_range_change'
    ] ;
    
    var _LOCK_IMAGE = document.getElementById('lock') ;

    /**
     * The 2-column tabular widget representing properties and their values
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
        Widget.Widget.call(this) ;

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
            var yMin = this._series[i].yRange.min ,
                yMax = this._series[i].yRange.max ;
            var deltaY = shift ? shift.dy * (yMax - yMin) / shift.ybins : (yMax - yMin) / 10. ;
            this._series[i].yRange.min -= deltaY ;
            this._series[i].yRange.max -= deltaY ;
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_move_up = function (i, shift) {
            var yMin = this._series[i].yRange.min ,
                yMax = this._series[i].yRange.max ;
            var deltaY = shift ? shift.dy * (yMax - yMin) / shift.ybins : (yMax - yMin) / 10. ;
            this._series[i].yRange.min += deltaY ;
            this._series[i].yRange.max += deltaY ;
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_zoom_in = function (i) {
            console.log('_y_zoom_in: series='+i) ;
            var yMin = this._series[i].yRange.min ,
                yMax = this._series[i].yRange.max ;
            var deltaY = (yMax - yMin) / 10. ;
            this._series[i].yRange.min += deltaY ;
            this._series[i].yRange.max -= deltaY ;
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_zoom_out = function (i) {
            console.log('_y_zoom_out: series='+i) ;
            var yMin = this._series[i].yRange.min ,
                yMax = this._series[i].yRange.max ;
            var deltaY = (yMax - yMin) / 10. ;
            this._series[i].yRange.min -= deltaY ;
            this._series[i].yRange.max += deltaY ;
            this._display() ;
            this._report_y_range_change(i) ;
        } ;
        this._y_reset_or_lock = function (i) {
            console.log('_y_reset_or_lock: series='+i) ;
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
            console.log('_y_lock: series='+i) ;
            this._y_move_up(i, {
                dy: 1 ,
                ybins: 1e9
            }) ;
        } ;

        // Rendering is done only once
        this._is_rendered = false ;

        this._canvas = null ;
        this._cxt = null ;
        this._geom = {} ;
        this._CANVAS_WIDTH  = 0 ;
        this._CANVAS_HEIGHT = 0 ;

        /**
         * Implement the widget rendering protocol
         *
         * @returns {undefined}
         */
        this.render = function () {
            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this.reset() ;

            // Initialize the plot with default settings. Also draw axes
            // just to show a user the geometry of the canvas as we don't have
            // any real data at this point.

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
            
            // No data or empty data - no display
            if (!this._series.length) return ;

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

            if (!this._canvas) {
                this._canvas = this.container ;
                
                // Resizing canvas's HTML element to fit into the visible window
                // and to prevent it from clipping or leaving extra space at the bottom
                // of the wndow.
                //
                // TODO: this algorithms assumes that canvas is located at the very
                //       bottom of the application's window. This may change if
                //       the application gets a different layout

                this._canvas.css('height', (window.innerHeight-this._canvas.offset().top - 30)+'px') ;
                $(window).resize (function () {
                    _that._canvas.css('height', (window.innerHeight-_that._canvas.offset().top - 30)+'px') ;
                    _that._display() ;  // redisplay is needed to prevent plots
                                        // from being scaled.
                }) ;

                // Track mouse position in order to intersept and process events
                // generated by a keyboard and mouse scroll (real mouse pointer - for
                // mouse-based inputs, or touch screen zoom in/outs for tables
                // and Apple laptops).

                this._inCanvas = false ;
                this._canvas.mouseover(function (e) {
                    e.preventDefault() ;
                    // This is needed for canva's operation's
                    _that._inCanvas = true ;
                }) ;
                this._canvas.mouseout(function (e) {
                    e.preventDefault() ;
                    _that._inCanvas = false ;
                    _that._trackRegion() ;
                }) ;
                this._canvas.mousedown(function (e) {
                    e.preventDefault() ;

                    console.log('mousedown', e) ;

                    // Store the present position of the mouse for range change
                    // operations performed when the mouse will be up.
                    _that._mouseDownPosition = {
                        x: e.offsetX ,
                        y: e.offsetY
                    };
                }) ;
                this._canvas.mousemove(function (e) {
                    e.preventDefault() ;

                    // Skip this if in the mouse drag mode

                    if (_that._mouseDownPosition) {
                        _that._mouseMoved = true ;

                        // Find a region where the mouse is in
                        var x = e.offsetX ,
                            y = e.offsetY ,
                            deltaX = x - _that._mouseDownPosition.x ,
                            deltaY = y - _that._mouseDownPosition.y ;

                        console.log('mousemove: (dx,dy)=('+deltaX+','+deltaY+')') ;

                        if (_that._activeRegion) {
                            switch (_that._activeRegion.direction) {
                                case 'Y':
                                    if (deltaY) {
                                        if (deltaY > 0) {
                                            if (_that._y_move_up)
                                                _that._y_move_up (_that._activeRegion.position, {
                                                    ybins: _that._geom.PLOT_HEIGHT ,
                                                    dy:    deltaY
                                                }) ;
                                        } else {
                                            if (_that._y_move_down)
                                                _that._y_move_down (_that._activeRegion.position, {
                                                    ybins: _that._geom.PLOT_HEIGHT ,
                                                    dy:    Math.abs(deltaY)
                                                }) ;
                                        }
                                    } else {

                                        /* This should never happen in this context
                                         */
                                        ;
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
                            console.log('mouse in region X['+i+']') ;
                            _that._trackRegion({
                                direction: 'X' ,
                                position: +i
                            }) ;
                            return ;
                        }
                    }
                    for (var i in _that._region.Y) {
                        var reg = _that._region.Y[i] ;
                        if (reg.xMin < x && x < reg.xMax &&
                            reg.yMin < y && y < reg.yMax) {
                            console.log('mouse in region Y['+i+']') ;
                            _that._trackRegion({
                                direction: 'Y' ,
                                position: +i
                            }) ;
                            return ;
                        }
                    }
//                    console.log('mouse not in any known region') ;
               
                }) ;
                this._canvas.mouseup(function (e) {
                    e.preventDefault() ;

                    // Find a region where the mouse is in
                    var x = e.offsetX ,
                        y = e.offsetY ,
                        deltaX = x - _that._mouseDownPosition.x ,
                        deltaY = y - _that._mouseDownPosition.y ;

                    console.log('mouseup: (dx,dy)=('+deltaX+','+deltaY+') mouseMoved='+(_that._mouseMoved ? 'true' : 'false')) ;

                    if (_that._activeRegion) {
                        switch (_that._activeRegion.direction) {
                            case 'X':
                                if (deltaX) {
                                    if (deltaX > 0) {
                                        if (_that._config.x_move_left)
                                            _that._config.x_move_left ({
                                                xbins: _that._geom.PLOT_WIDTH ,
                                                dx:    deltaX
                                            }) ;
                                    } else {
                                        if (_that._config.x_move_right)
                                            _that._config.x_move_right ({
                                                xbins: _that._geom.PLOT_WIDTH ,
                                                dx:    Math.abs(deltaX)
                                            }) ;
                                    }
                                }
                                break ;

                            case 'Y':
                                if (deltaY) {
                                    if (deltaY > 0) {
                                        if (_that._y_move_up)
                                            _that._y_move_up (_that._activeRegion.position, {
                                                ybins: _that._geom.PLOT_HEIGHT ,
                                                dy:    deltaY
                                            }) ;
                                    } else {
                                        if (_that._y_move_down)
                                            _that._y_move_down (_that._activeRegion.position, {
                                                ybins: _that._geom.PLOT_HEIGHT ,
                                                dy:    Math.abs(deltaY)
                                            }) ;
                                    }
                                } else {
                                    if (_that._y_reset_or_lock && !_that._mouseMoved)
                                        _that._y_reset_or_lock(_that._activeRegion.position) ;
                                }
                                break ;
                        }
                    }
                                            
                    // Always reset this
                    _that._mouseDownPosition = null ;
                    _that._mouseMoved = false ;
                }) ;
                this._canvas.mousewheel(function (e) {

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
            }
            
            // IMPORTANT: Make sure that canvas' attributes always stay in sync
            //            with the HTML element's CSS geometry. In this case
            //            we'll be guaranteed 1:1 mapping between window's pixels
            //            and canvas' pixels. Otherwise an unpleasant scaling will
            //            occure each time the window gets resized.

            this._CANVAS_WIDTH  = this._canvas.css('width') .substr(0, this._canvas.css('width') .length - 2) ,  // remove 'px'
            this._CANVAS_HEIGHT = this._canvas.css('height').substr(0, this._canvas.css('height').length - 2) ;  // remove 'px'

            this._canvas.attr('width',  this._CANVAS_WIDTH) ;
            this._canvas.attr('height', this._CANVAS_HEIGHT) ;

            // Get and tune up the 2D drawing context
            this._cxt = this._canvas.get(0).getContext('2d') ;
            this._cxt.translate(0.5, 0.5) ;     // half-step translation is needed
                                                // to prevent line blurring

            this._initGeometry() ;
            this._clearPlot() ;
        } ;

        this._initGeometry = function () {    

            // Calculate total space in X dimension which would be required to
            // draw all Y labels on vertical axes.

            this._geom.X_MIN      = 2 * _LABEL_FONT_SIZE ;
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
                                this._cxt.measureText(label.text).width) ;
                            break ;
                        case 'log10':
                            yLabelWidth = Math.max (
                                yLabelWidth ,
                                this._cxt.measureText(label.base_text + ' ' + label.exponent_text).width) ;
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
                var stepSize = this._geom.PLOT_HEIGHT / (series.yLabels.get().length + 1) ;
                this._geom.PLOT_Y_STEP_SIZE.push(stepSize) ;
                this._geom.PLOT_Y_BEGIN.push(this._geom.PLOT_Y_MAX - stepSize) ;
            }
            console.log('this._geom', this._geom) ;
        } ;

        this._clearPlot = function () {
            this._cxt.clearRect(0, 0, this._CANVAS_WIDTH, this._CANVAS_HEIGHT) ;
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
                this._cxt.fillStyle = _ACTIVE_AXES_COLOR ;
                this._cxt.fillRect(xMin+1,  yMin+1, width-1, height-1) ;
            } else {
                this._cxt.clearRect(xMin+1, yMin,   width-1, height) ;
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
                var xLabelWidth = this._cxt.measureText(format).width ;
                var xDesiredTicks = 10 * Math.max(1, Math.floor(this._geom.PLOT_WIDTH / (1.5 * xLabelWidth) / 10)) ;
                var xLabelsGenerator = new LabelGenerator.forMinutes() ;
                var xLabels = xLabelsGenerator.search(xRangeMin, xRangeMax, xDesiredTicks) ;
                xFormattedLabels = xLabels.empty_duplicates(xLabels.pretty_formatted_timestamps()[format]) ;
            } else {
                var format = 'YYYY-MM-DD' ;
                var xLabelWidth = this._cxt.measureText(format).width ;
                var xDesiredTicks = 2 * Math.max(1, Math.floor(this._geom.PLOT_WIDTH / (1.5 * xLabelWidth) / 2)) ;
                var xLabelsGenerator = new LabelGenerator.forDays() ;
                var xLabels = xLabelsGenerator.search(xRangeMin, xRangeMax, xDesiredTicks) ;
                xFormattedLabels = xLabels.empty_duplicates(xLabels.pretty_formatted_timestamps()[format]) ;
            }

            var xStepSize = this._geom.PLOT_WIDTH / (xFormattedLabels.length + 1) ;

            var y = this._geom.PLOT_Y_MAX ,
                yLabelOffset = y + 2 * _LABEL_FONT_SIZE ;    // start position of labels is 2 simbols below.

            this._cxt.beginPath() ;
            this._cxt.font      = _LABEL_FONT ;
            this._cxt.fillStyle = _LABEL_COLOR ;
            this._cxt.textAlign = 'center';
            this._cxt.moveTo(this._geom.PLOT_X_MIN, y) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX, y) ;
          
            for (var step = 0, x = this._geom.PLOT_X_MIN + xStepSize ;  // no tick at position 0
                     step < xFormattedLabels.length ;                   // no tick at the end
                   ++step,     x += xStepSize)
            {
                this._cxt.moveTo(x, y) ;
                this._cxt.lineTo(x, y + _TICK_SIZE) ;
                var label = ''+xFormattedLabels[step].text ;
                // Note that the starting point of the text is
                // depends on the alignment. If the text is center aligned then
                // the central position fo the label text will be consider for text placement .
                this._cxt.fillText(label, x, yLabelOffset) ;
            }
            this._cxt.strokeStyle = _PRIMARY_AXIS_COLOR ;
            this._cxt.lineWidth   = 1 ;
            this._cxt.stroke();
            
            this._cxt.beginPath() ;
            this._cxt.strokeStyle = _AXIS_COLOR ;
            this._cxt.moveTo(this._geom.PLOT_X_MIN, yLabelOffset + _LABEL_FONT_SIZE) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX, yLabelOffset + _LABEL_FONT_SIZE) ;
            this._cxt.lineWidth = 1 ;
            this._cxt.stroke();

            this._cxt.font      = _LABEL_FONT ;
            this._cxt.fillStyle = 'black' ;
            this._cxt.textAlign = 'center';

            this._cxt.beginPath() ;
            
            for (var step = 0, x = this._geom.PLOT_X_MIN + xStepSize ;      // no tick at position 0
                     step < xFormattedLabels.length ;                       // no tick at the end
                   ++step,     x += xStepSize)
            {
                if (step == 25) {
                    this._cxt.moveTo(x, yLabelOffset + _LABEL_FONT_SIZE) ;
                    this._cxt.lineTo(x, yLabelOffset + _LABEL_FONT_SIZE + _TICK_SIZE) ;
                    // Note that the starting point of the text is
                    // depends on the alignment. If the text is center aligned then
                    // the central position fo the label text will be consider for text placement .
                    this._cxt.fillText('2015-06-10', x, yLabelOffset + 3 * _LABEL_FONT_SIZE) ;
                }
            }
            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = _AXIS_COLOR ;
            this._cxt.stroke();
            
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
                this._cxt.fillStyle = _ACTIVE_AXES_COLOR ;
                this._cxt.fillRect(xMin+1,  yMin+1, width-1, height-1) ;
            } else {
                this._cxt.clearRect(xMin+1, yMin,   width-1, height) ;
            }
            
            // Draw the axes itself

            this._cxt.beginPath() ;
            this._cxt.lineWidth   = i == this._series.length - 1 ?                 1.25 :           1 ;
            this._cxt.strokeStyle = i == this._series.length - 1 ? _PRIMARY_AXIS_COLOR : _AXIS_COLOR ;
            this._cxt.moveTo(xMax, yMin) ;
            this._cxt.lineTo(xMax, yMax) ;
            this._cxt.stroke();

            // Draw ticks and labels

            this._cxt.beginPath() ;
            this._cxt.lineWidth = 1 ;

            this._cxt.fillStyle = series.color ;   // using the same color as the one of the plotted values
            this._cxt.textAlign = 'right';

            var yFormattedLabels = series.yLabels.pretty_formatted() ,
                yTick = this._geom.PLOT_Y_BEGIN[i] ;                // no tick at position 0
            for (var step = 0; step < series.yLabels.get().length ; // no tick at the end
                   ++step, yTick -= this._geom.PLOT_Y_STEP_SIZE[i])
            {
                // Tick

                this._cxt.moveTo (xMax - _TICK_SIZE, yTick) ;
                this._cxt.lineTo (xMax,              yTick) ;

                // Label
                //
                // Note that the starting point of the text is
                // depends on the alignment. If the text is rigth aligned then
                // the position will be counted from the very right (from the vertical axis).

                switch (series.scale) {
                    case 'linear':
                        var label = yFormattedLabels[step].text ;
                        this._cxt.font = _LABEL_FONT ;
                        this._cxt.fillText (label, xLabelOffset, yTick + _LABEL_FONT_SIZE / 2) ;
                        break ;
                    case 'log10':
                        var label = yFormattedLabels[step] ;
                        this._cxt.font = _LABEL_FONT ;
                        this._cxt.fillText (label.base_text, xLabelOffset - _LABEL_FONT_SIZE / 2, yTick + _LABEL_FONT_SIZE / 2) ;
                        this._cxt.font = _LABEL_EXPONENT_FONT ;
                        this._cxt.fillText (label.exponent_text, xLabelOffset - _LABEL_FONT_SIZE / 2 + _LABEL_FONT_SIZE, yTick - 0.6 * _LABEL_FONT_SIZE) ;
                        break ;
                }
            }
            this._cxt.stroke();

            // Put the lock flag on top of the axes to make the user
            // aware that teh dynamic range of the range is not computed automatically.

            if ((series.yRange.min !== series.yOriginalRange.min) ||
                (series.yRange.max !== series.yOriginalRange.max)) {
                this._cxt.drawImage(_LOCK_IMAGE, xMin + (xMax - xMin) / 2 - 10, yMin, 20, 20);
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
                this._cxt.beginPath() ;
                this._cxt.lineWidth = 1 ;

                // Emphasize '0' grid line if the value at the absolute value at
                // the current grid point is less than the 'epsilon' (some small number)
                // This should cover three cases:
                //
                // - absolute zero (0.)
                // - nearly zero positive number s (very small values which are
                //   smaller than 'epsilon')
                // - negative zeroes (which are actually small negative numbers whose
                //   absolute value is smaller than 'epsilon')

                this._cxt.strokeStyle = (ystep > 0) && (ystep < nSteps) && Math.abs(yLabels[ystep].value) < 10e-9 ? 
                    _ZERO_GRID_COLOR :
                    _GRID_COLOR ;

                this._cxt.moveTo(this._geom.PLOT_X_MIN + 2, y) ;    // +2 to avoid overwriting the axes
                this._cxt.lineTo(this._geom.PLOT_X_MAX,     y) ;

                this._cxt.stroke();
            }

        } ;
        this._clearDataPlotArea = function (deltaX) {

            this._cxt.clearRect (
                this._geom.PLOT_X_MIN+1 ,
                0 ,
                this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN - 1 ,
                this._geom.PLOT_Y_MAX - 1) ;

            if (deltaX) {
                
                // Show the "no data yet" space in the X drag mode

                this._cxt.fillStyle = '#f8f8f8' ;
                this._cxt.fillRect (
                    deltaX < 0 ? this._geom.PLOT_X_MAX : this._geom.PLOT_X_MIN + 1 ,
                    this._geom.PLOT_Y_MIN ,
                    deltaX ,
                    this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN - 1) ;

            }
        } ;
        this._plotData = function (deltaX) {

            for (var sid = 0; sid < this._series.length; ++sid) {
                var series = this._series[sid] ;

                this._cxt.beginPath() ;
                this._cxt.lineWidth = 1;
                this._cxt.strokeStyle = series.color ;

                if (!(!this._activeRegion || !(this._activeRegion.direction === 'Y') || (this._activeRegion.position === sid))) {
                    this._cxt.globalAlpha = 0.25 ; 
                }
                switch (series.scale) {
                    case 'linear': this._plotDataLinear(deltaX, series, sid) ; break ;
                    case 'log10':  this._plotDataLog10 (deltaX, series, sid) ; break ;
                }
                this._cxt.stroke ();
                this._cxt.globalAlpha = 1. ;
                
                // Show the name of the series if in the X axes selection mode
                
                if (this._activeRegion && (this._activeRegion.direction === 'Y') && (this._activeRegion.position === sid)) {
                    this._cxt.font      = _SEREIS_NAME_FONT ;
                    this._cxt.fillStyle = series.color ;   // using the same color as the one of the plotted values
                    this._cxt.textAlign = 'right';
                    this._cxt.fillText (
                        series.name ,
                        this._geom.PLOT_X_MAX ,
                        this._geom.PLOT_Y_MIN + _SEREIS_NAME_FONT_SIZE) ;
                }
            }
        } ;
        this._plotDataLinear = function (deltaX, series, sid) {

            console.log("_plotDataLinear: "+series.name) ;

            var xDelta = this._xRange.max - this._xRange.min ,
                x0     = this._geom.PLOT_X_MIN + (deltaX               ? deltaX : 0) ,
                xmin   = this._geom.PLOT_X_MIN ,
                xmax   = this._geom.PLOT_X_MAX + (deltaX && deltaX < 0 ? deltaX : 0) ,
                yPrev  = null ;

            for (var j in series.points) {
                var p = series.points[j] ;

                var xVal = p[0] ,
                    yVal = p[1] ;

                var x = x0 + this._geom.PLOT_WIDTH  * ((xVal - this._xRange.min) / xDelta) ;
                if (x < xmin) continue ;
                if (x > xmax) continue ;

                var y = this._geom.PLOT_Y_BEGIN[sid] - (this._geom.PLOT_Y_STEP_SIZE[sid] / series.yLabels.step) * (yVal - series.yLabels.min) ;
                if (y > this._geom.PLOT_Y_MAX - 1) y = this._geom.PLOT_Y_MAX - 1 ;
                if (y < this._geom.PLOT_Y_MIN) y = this._geom.PLOT_Y_MIN ;
                if (_.isNull(yPrev)) {
                    this._cxt.moveTo (x, y) ;
                } else {
                    this._cxt.lineTo (x, yPrev) ;
                    this._cxt.lineTo (x, y) ;
                }
                yPrev = y ;
            }
        } ;
        this._plotDataLog10 = function (deltaX,series, sid) {

            console.log("_plotDataLog10: "+series.name) ;

                var xDelta = this._xRange.max - this._xRange.min ,
                    x0     = this._geom.PLOT_X_MIN + (deltaX               ? deltaX : 0) ,
                    xmin   = this._geom.PLOT_X_MIN ,
                    xmax   = this._geom.PLOT_X_MAX + (deltaX && deltaX < 0 ? deltaX : 0) ,
                    yPrev  = null ,
                    epsilon = 10e-9 ;

                for (var j in series.points) {
                    var p = series.points[j] ;

                    var xVal = p[0] ,
                        yVal = p[1] ;

                    var x = x0 + this._geom.PLOT_WIDTH  * ((xVal - this._xRange.min) / xDelta) ;
                    if (x < xmin) continue ;
                    if (x > xmax) continue ;

//                console.log("_plotDataLog10: Math.log10(yVal) - Math.log10(series.yLabels.min)", Math.log10(yVal) - Math.log10(series.yLabels.min)) ;
//                console.log("_plotDataLog10: Math.log10(series.yLabels.max) - Math.log10(series.yLabels.min)", Math.log10(series.yLabels.max) - Math.log10(series.yLabels.min)) ;
//                console.log("_plotDataLog10: this._geom.PLOT_Y_STEP_SIZE[sid] * series.yLabels.length", this._geom.PLOT_Y_STEP_SIZE[sid] * series.yLabels.get().length) ;
//
                    // TEMPORARY RESTRICTION: avoid displaying signal values
                    // to which the log10(v) operation won't apply. This will be taken
                    // care off later.
                    if (yVal <= epsilon) continue ;
                    var y = this._geom.PLOT_Y_BEGIN[sid] -
                        (Math.log10(yVal) - Math.log10(series.yLabels.min)) /
                        ((Math.log10(series.yLabels.max) - Math.log10(series.yLabels.min)) / (this._geom.PLOT_Y_STEP_SIZE[sid] * series.yLabels.get().length)) ;

                    console.log("_plotDataLog10: "+y) ;


                    if (y > this._geom.PLOT_Y_MAX - 1) y = this._geom.PLOT_Y_MAX - 1 ;
                    if (y < this._geom.PLOT_Y_MIN) y = this._geom.PLOT_Y_MIN ;
                    if (_.isNull(yPrev)) {
                        this._cxt.moveTo (x, y) ;
                    } else {
                        this._cxt.lineTo (x, yPrev) ;
                        this._cxt.lineTo (x, y) ;
                    }
                    yPrev = y ;
                }
        } ;
    }
    Class.define_class (TimeSeriesPlotN, Widget.Widget, {}, {}) ;

    return TimeSeriesPlotN ;
}) ;
