define ([
    'webfwk/Class', 'webfwk/Widget' ,
    'EpicsViewer/LabelGenerator'] ,

function (
    Class, Widget ,
    LabelGenerator) {


    // Plot parameteres
    //
    // TODO: consider external customization via
    //       widget's parameters.
    
    var _LABEL_FONT_SIZE =  9 ;
    var _LABEL_FONT      = _LABEL_FONT_SIZE+'pt Calibri' ;
    var _TICK_SIZE       =  4 ;

    var _AXIS_COLOR  = '#b0b0b0' ;
    var _LABEL_COLOR = '#a0a0a0' ;
    var _GRID_COLOR  = '#f0f0f0' ;
    var _PLOT_COLOR  = '#0071bc' ;

    var _KEY_LEFT_ARROW  = 37 ,
        _KEY_RIGHT_ARROW = 39 ,
        _KEY_UP_ARROW    = 38 ,
        _KEY_DOWN_ARROW  = 40 ;

    var _EVENT_HANDLER = [
        'time_move_left' ,
        'time_move_right' ,
        'time_zoom_in' ,
        'time_zoom_out'
    ] ;

    /**
     * The 2-column tabular widget representing properties and their values
     *
     * USAGE:
     * 
     *   TO BE COMPLETED...
     *
     * @returns {TimeSeriesPlot}
     */
    function TimeSeriesPlot (config) {

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
            
            // Initialize the plot with default settings. Also draw axes
            // just to show a user the geometry of the canvas as we don't have
            // any real data at this point.

            this._initDrawingContext() ;
            this._prepareAxes() ;
        } ;

        this._xRange = {
            min: (new Date()) / 1000. - 7 * 24 * 3600. ,    // 7 days ago
            max: (new Date()) / 1000.                       // right now
        } ;
        this._series = [{
            yLabels: null ,
            yRange: {
                min: 0 ,
                max: 1.
            } ,
            points: null
        }] ;

        this.load = function (xRange, series) {
            this._xRange = xRange ;
            this._series[0].yRange = series.yRange ;

            // add a little to the range to preven the lockup in
            // the widget implementation.
            //
            // TODO: consider a more reliable way of sanitizing
            //       the implementation.
            if (this._series[0].yRange.min === this._series[0].yRange.max) {
                console.log('TimeSeries.load() trap yRange.min === yRange.max', this._series[0].yRange.max) ;
                this._series[0].yRange.max += !this._series[0].yRange.max ? 
                    1. :
                    Math.abs(this._series[0].yRange.max / 2) ;
            }
            this._series[0].points = series.points ;
            
            this._display() ;
        } ;
        this.load_many = function (xRange, many_series) {
            this._xRange = xRange ;
            this._series = [] ;
            for (var i in many_series) {
                var s = {
                    yLabels: null ,
                    yRange:  many_series[i].yRange ,
                    points:  many_series[i].points
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
                }
                this._series.push(s) ;
            }
            this._display() ;
        } ;
        this._display = function () {
            
            // No data or empty data - no display
            if (!(this._series[0].points &&
                 (this._series[0].points.length > 1) &&
                 (this._xRange.max - this._xRange.min))) return ;

            this._initDrawingContext() ;
            this._prepareAxes() ;
            this._plotData() ;
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

                this._in_canvas = false ;
                this._canvas.mouseover(function (e) {
                    e.preventDefault() ;
                    _that._in_canvas = true ;
                }) ;
                this._canvas.mouseout(function (e) {
                    e.preventDefault() ;
                    _that._in_canvas = false ;
                }) ;
                

                $(document).keydown(function (e) {
                    
                    // Intercepting _specific_ keyboard events and _only_ when
                    // the mouse pointer is within the canvas. This won't affect
                    // other operations.

                    if (_that._in_canvas) {
                        switch (e.keyCode) {

                            case _KEY_LEFT_ARROW:
                                e.preventDefault() ;
                                if (_that._config.time_move_left)
                                    _that._config.time_move_left({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;

                            case _KEY_RIGHT_ARROW:
                                e.preventDefault() ;
                                if (_that._config.time_move_right)
                                    _that._config.time_move_right({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;

                            case _KEY_UP_ARROW:
                                e.preventDefault() ;
                                if (_that._config.time_zoom_in)
                                    _that._config.time_zoom_in({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;

                            case _KEY_DOWN_ARROW:
                                e.preventDefault() ;
                                if (_that._config.time_zoom_out)
                                    _that._config.time_zoom_out({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;
                        }
                    }
                }) ;
                this._canvas.mousewheel(function (e) {

                    // TODO: These are some ideas on how to do the so called "zoom math"
                    //
                    //   http://stackoverflow.com/questions/2916081/zoom-in-on-a-point-using-scale-and-translate
                    //   http://stackoverflow.com/questions/6775168/zooming-with-canvas

                    e.preventDefault() ;
                    if (e.deltaY > 0) {
                        if (_that._config.time_zoom_in)
                            _that._config.time_zoom_in({xbins: _that._geom.PLOT_WIDTH}) ;
                    } else {
                        if (_that._config.time_zoom_out)
                            _that._config.time_zoom_out({xbins: _that._geom.PLOT_WIDTH}) ;
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


            // Set up the label generator for the Y axis here becase we need
            // to calculate the left offset for the plot.

            // Calculate a desired number of ticks based on the size of
            // the canvas and font size of the labels.

            var y_labels_generator = new LabelGenerator.base10() ;
            this._series[0].yLabels = y_labels_generator.search (
                this._series[0].yRange.min ,
                this._series[0].yRange.max ,
                Math.max (
                    2 ,         // at least
                    Math.min (
                        Math.floor(this._CANVAS_HEIGHT / (4 * _LABEL_FONT_SIZE)) ,
                        20      // at most
                    )
                )
            ) ;
            
            // Calculate the maximum width of the formatted labels
            // and use this to define the left offset of he plot
            // so that the labels would fit in there.

            var yLabelWidth = 0 ;
            var yFormattedLabels = this._series[0].yLabels.pretty_formatted() ;
                    
            for (var i in yFormattedLabels) {
                var label = yFormattedLabels[i] ;
                yLabelWidth = Math.max (yLabelWidth, this._cxt.measureText(label).width) ;
            }
            console.log('yLabelWidth', yLabelWidth) ;

            this._geom.PLOT_X_MIN  = yLabelWidth          + 2 * _LABEL_FONT_SIZE ;
            this._geom.PLOT_X_MAX  = this._CANVAS_WIDTH   - 2 * _LABEL_FONT_SIZE ;
            this._geom.PLOT_Y_MIN  =                       20 ;
            this._geom.PLOT_Y_MAX  = this._CANVAS_HEIGHT - 60 ;
            this._geom.PLOT_WIDTH  = this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN ;
            this._geom.PLOT_HEIGHT = this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN ;

            // Actual limits for plotting in the Y dimension
            // 
            //   PLOT_Y_STEP_SIZE:  plotted interval between ticks
            //   PLOT_Y_BEGIN:      the position of the first tick
            //
            this._geom.PLOT_Y_STEP_SIZE = this._geom.PLOT_HEIGHT / (this._series[0].yLabels.get().length + 1) ;
            this._geom.PLOT_Y_BEGIN = this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_STEP_SIZE ;

            console.log('this._geom', this._geom) ;
        } ;

        this._clearPlot = function () {
            this._cxt.clearRect(0, 0, this._CANVAS_WIDTH, this._CANVAS_HEIGHT) ;
        } ;
        this._prepareAxes = function () {
            this._drawAxisX() ;
            this._drawAxisY() ;
        } ;
        this._drawAxisX = function () {

            var now_sec = (new Date()) / 1000. ;
            var xRangeMin = this._xRange ? this._xRange.min : now_sec - 7 * 24 * 3600. ,
                xRangeMax = this._xRange ? this._xRange.max : now_sec ,
                xDelta    = xRangeMax - xRangeMin ;

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
          
            for (var step = 0, x = this._geom.PLOT_X_MIN + xStepSize ;      // no tick at position 0
                     step < xFormattedLabels.length ;            // no tick at the end
                   ++step,     x += xStepSize)
            {
                this._cxt.moveTo(x, y) ;
                this._cxt.lineTo(x, y - _TICK_SIZE) ;
                var label = ''+xFormattedLabels[step] ;
                // Note that the starting point of the text is
                // depends on the alignment. If the text is center aligned then
                // the central position fo the label text will be consider for text placement .
                this._cxt.fillText(label, x, yLabelOffset) ;
            }
            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = _AXIS_COLOR ;
            this._cxt.stroke();
            
            this._cxt.beginPath() ;
            this._cxt.font      = _LABEL_FONT ;
            this._cxt.fillStyle = 'black' ;
            this._cxt.textAlign = 'center';
            this._cxt.moveTo(this._geom.PLOT_X_MIN, yLabelOffset + _LABEL_FONT_SIZE) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX, yLabelOffset + _LABEL_FONT_SIZE) ;
            
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
        } ;
        this._drawAxisY = function () {

            var x = this._geom.PLOT_X_MIN ,
                xLabelOffset = x - _LABEL_FONT_SIZE ;    // start position of labels is one simbol left.

            this._cxt.beginPath() ;

            this._cxt.font      = _LABEL_FONT ;
            this._cxt.fillStyle = _PLOT_COLOR ;     // using the same color as the one of the plotted values
            this._cxt.textAlign = 'right';

            this._cxt.moveTo(x, this._geom.PLOT_Y_MAX + 5 * _LABEL_FONT_SIZE + _TICK_SIZE) ;
            this._cxt.lineTo(x, this._geom.PLOT_Y_MIN) ;
            
            var yFormattedLabels = this._series[0].yLabels.pretty_formatted() ;
            for (var step = 0, y = this._geom.PLOT_Y_BEGIN ;            // no tick at position 0
                     step < this._series[0].yLabels.get().length ;      // no tick at the end
                   ++step,     y -= this._geom.PLOT_Y_STEP_SIZE)
            {
                // Tick

                this._cxt.moveTo (x - _TICK_SIZE, y) ;
                this._cxt.lineTo (x, y) ;

                // Label
                //
                // Note that the starting point of the text is
                // depends on the alignment. If the text is rigth aligned then
                // the position will be counted from the very right (from the vertical axis).

                var label = ''+yFormattedLabels[step] ;
                this._cxt.fillText (label, xLabelOffset, y + _LABEL_FONT_SIZE / 2) ;
            }
            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = _AXIS_COLOR ;
            this._cxt.stroke();


            // Grid lines

            this._cxt.beginPath() ;

            for (var ystep = 0, y = this._geom.PLOT_Y_BEGIN ;           // no line at position 0
                     ystep < this._series[0].yLabels.get().length ;     // no line at the end
                   ++ystep,     y -= this._geom.PLOT_Y_STEP_SIZE)
            {
                this._cxt.moveTo(this._geom.PLOT_X_MIN, y) ;
                this._cxt.lineTo(this._geom.PLOT_X_MAX, y) ;
            }
            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = _GRID_COLOR ;
            this._cxt.stroke();

        } ;
        this._plotData = function () {
            var xDelta = this._xRange.max - this._xRange.min ;
            if (!(xDelta && this._series[0].points.length > 1)) {
                this._cxt.font = _LABEL_FONT ;
                this._cxt.fillStyle = 'maroon';
                this._cxt.fillText (
                    'Not enough measuremets to plot' ,
                    this._geom.PLOT_X_MIN + Math.floor (this._xLabels.get().length          / 2 - 1) * this._xLabels.step ,
                    this._geom.PLOT_Y_MIN + Math.floor (this._series[0].yLabels.get().length / 2)     * this._series[0].yLabels.step) ;
                return ;
            }

            var yPrev = null ;

            this._cxt.lineWidth = 1;
            this._cxt.strokeStyle = _PLOT_COLOR ;
            this._cxt.beginPath() ;
            for (var i in this._series[0].points) {
                var p = this._series[0].points[i] ;
                var x = this._geom.PLOT_X_MIN + this._geom.PLOT_WIDTH  * ((p[0] - this._xRange.min) / xDelta) ,
                    y = this._geom.PLOT_Y_BEGIN -
                    (this._geom.PLOT_Y_STEP_SIZE / this._series[0].yLabels.step) * (p[1] - this._series[0].yLabels.min) ;
                if (_.isNull(yPrev)) {
                    this._cxt.moveTo (x, y) ;
                } else {
                    this._cxt.lineTo (x, yPrev) ;
                    this._cxt.lineTo (x, y) ;
                }
                yPrev = y ;
            }
            this._cxt.stroke ();
        } ;
    }
    Class.define_class (TimeSeriesPlot, Widget.Widget, {}, {}) ;

    return TimeSeriesPlot ;
}) ;
