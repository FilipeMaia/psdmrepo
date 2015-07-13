define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget' ,
    'EpicsViewer/LabelGenerator'] ,

function (
    cssloader, Class, Widget ,
    LabelGenerator) {

//    cssloader.load('../EpicsViewer/css/TimeSeriesPlot.css') ;


    // Plot parameteres
    //
    // TODO: Consider external customization via widget's parameters
    
    var _LABEL_FONT_SIZE =  9 ;
    var _LABEL_FONT      = _LABEL_FONT_SIZE+'pt Calibri' ;
    var _TICK_SIZE       =  4 ;

    var _AXIS_COLOR  = '#b0b0b0' ;
    var _LABEL_COLOR = '#a0a0a0' ;
    var _GRID_COLOR  = '#f0f0f0' ;
    var _PLOT_COLOR  = '#0071bc' ;

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
            for (var i in _EVENT_HANDLER) {
                var handler = _EVENT_HANDLER[i] ;
                if (_.has(config, handler)) {
                    var func = config[handler] ;
                    if (_.isFunction(func)) this._config[handler] = func ;
                }
            }
        }
        console.log('TimeSeriesPlot._config', this._config) ;
        
        // Rendering is done only once
        this._is_rendered = false ;

        this._canvas = null ;
        this._cxt = null ;

        this._yLabels = null ;

        /**
         * Implement the widget rendering protocol
         *
         * @returns {undefined}
         */
        this.render = function () {
            if (this._is_rendered) return ;
            this._is_rendered = true ;
            
            // Initialize the plot with default settings
            this._init_drawing_context() ;
            this._clear() ;
            this._prepareAxes() ;
        } ;

        this._xRange = null ;
        this._yRange = null ;
        this._points  = null ;

        this.load = function (x_range, y_range, points) {
            this._xRange = x_range ;
            this._yRange = y_range ;
            this._points  = points ;
            this._display() ;
        } ;

        this._cxt  = null ;
        this._geom = null ;

        this._init_drawing_context = function () {
            
            console.log('TimeSeriesPlot._init_drawing_context') ;

            if (!this._canvas) {
                this._canvas = this.container ;
                this._canvas.css('height', (window.innerHeight-this._canvas.offset().top - 30)+'px') ;
                $(window).resize (function () {
                    _that._canvas.css('height', (window.innerHeight-_that._canvas.offset().top - 30)+'px') ;
                    // redisplay needed to prevent plots from being scaled.
                    _that._display() ;
                }) ;
                this._in_canvas = false ;
                this._canvas.mouseover(function (e) {
                    console.log('canvas.mouseover') ;
                    e.preventDefault() ;
                    _that._in_canvas = true ;
                }) ;
                this._canvas.mouseout(function (e) {
                    console.log('canvas.mouseout') ;
                    e.preventDefault() ;
                    _that._in_canvas = false ;
                }) ;
                $(document).keydown(function (e) {
                    console.log('document.keydown: _in_canvas:', _that._in_canvas) ;
                    if (_that._in_canvas) {
                        console.log('document.keydown - event intercepted in canvas', e) ;
                        e.preventDefault() ;
                        switch (e.keyCode) {
                            case 37:
                                if (_that._config.time_move_left)
                                    _that._config.time_move_left({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;
                            case 39:
                                if (_that._config.time_move_right)
                                    _that._config.time_move_right({xbins: _that._geom.PLOT_WIDTH}) ;
                                break ;
                        }
                    }
                }) ;
                this._canvas.mousewheel(function (e) {
                    // TODO: These are some ideas on how to do the so called "zoom math"
                    //
                    //   http://stackoverflow.com/questions/2916081/zoom-in-on-a-point-using-scale-and-translate
                    //   http://stackoverflow.com/questions/6775168/zooming-with-canvas
                    //
                    console.log('canvas.mousewheel', e) ;
                    e.preventDefault() ;
                    if (e.deltaY < 0) {
                        if (_that._config.time_zoom_in)
                            _that._config.time_zoom_in({xbins: _that._geom.PLOT_WIDTH}) ;
                    } else {
                        if (_that._config.time_zoom_out)
                            _that._config.time_zoom_out({xbins: _that._geom.PLOT_WIDTH}) ;
                    }
                }) ;
            }
            var css_width  = this._canvas.css('width') .substr(0, this._canvas.css('width') .length - 2) ,
                css_height = this._canvas.css('height').substr(0, this._canvas.css('height').length - 2) ;

            this._canvas.attr('width',  css_width) ;
            this._canvas.attr('height', css_height) ;

            this._cxt = this._canvas.get(0).getContext('2d') ;

            // half-step translation to prevent line blurring
            this._cxt.translate(0.5, 0.5) ;

            // -------------------------------------------
            //         Set up the geometry of plots
            // -------------------------------------------

            this._geom = {} ;
            this._geom.WIDTH  = this._canvas.width() ;
            this._geom.HEIGHT = this._canvas.height() ;


            // Set up the label generator for the Y axis here becase we need
            // to calculate the left offset for the plot.


            // Calculate a desired number of ticks based on the size of
            // the canvas and font size of the labels.

            var y_labels_generator = new LabelGenerator.base10() ;
            this._yLabels = y_labels_generator.search (
                this._yRange ? this._yRange.min : 0. ,
                this._yRange ? this._yRange.max : 1. ,
                Math.max (
                    2 ,         // at least
                    Math.min (
                        Math.floor(this._geom.HEIGHT / (4 * _LABEL_FONT_SIZE)) ,
                        20      // at most
                    )
                )
            ) ;
            
            // Calculate the maximum width of the formatted labels
            // and use this to define the left offset of he plot
            // so that the labels would fit in there.

            var yLabelWidth = 0 ;
            var yFormattedLabels = this._yLabels.pretty_formatted() ;
                    
            for (var i in yFormattedLabels) {
                var label = yFormattedLabels[i] ;
                yLabelWidth = Math.max (yLabelWidth, this._cxt.measureText(label).width) ;
            }
            console.log('yLabelWidth', yLabelWidth) ;

            this._geom.PLOT_X_MIN  = yLabelWidth       + 2 * _LABEL_FONT_SIZE ;
            this._geom.PLOT_X_MAX  = this._geom.WIDTH  - 2 * _LABEL_FONT_SIZE ;
            this._geom.PLOT_Y_MIN  =                     20 ;
            this._geom.PLOT_Y_MAX  = this._geom.HEIGHT - 60 ;
            this._geom.PLOT_WIDTH  = this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN ;
            this._geom.PLOT_HEIGHT = this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN ;

            console.log('this._geom', this._geom) ;
        } ;

        this._clear = function () {
            this._cxt.clearRect(0, 0, this._geom.WIDTH, this._geom.HEIGHT) ;
        } ;
        this._display = function () {
            
            // No data or empty data - no display
            if (!(this._points &&
                 (this._points.length > 1) &&
                 (this._xRange.max - this._xRange.min))) return ;

            this._init_drawing_context() ;
            this._clear() ;
            this._prepareAxes() ;
            this._plotData() ;
        } ;

        this._prepareAxes = function () {

            // ---------------------
            //        X Axis
            // ---------------------

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
                     step < xFormattedLabels.length ;            // no tick at the end
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

            // ---------------------
            //        Y Axis
            // ---------------------

            // Plotted interval between ticks
            var yStepSize = this._geom.PLOT_HEIGHT / (this._yLabels.get().length + 1) ;

            var x = this._geom.PLOT_X_MIN ,
                xLabelOffset = x - _LABEL_FONT_SIZE ;    // start position of labels is one simbol left.

            this._cxt.beginPath() ;

            this._cxt.font      = _LABEL_FONT ;
            this._cxt.fillStyle = _PLOT_COLOR ;     // using the same color as teh one of the plotted values
            this._cxt.textAlign = 'right';

            this._cxt.moveTo(x, this._geom.PLOT_Y_MAX + 5 * _LABEL_FONT_SIZE + _TICK_SIZE) ;
            this._cxt.lineTo(x, this._geom.PLOT_Y_MIN) ;
            
            var yFormattedLabels = this._yLabels.pretty_formatted() ;
            for (var step = 0, y = this._geom.PLOT_Y_MAX - yStepSize ;      // no tick at position 0
                     step < this._yLabels.get().length ;                    // no tick at the end
                   ++step,     y -= yStepSize)
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


            // ---------------------
            //       Grid lines
            // ---------------------

            this._cxt.beginPath() ;

            for (var ystep = 0, y = this._geom.PLOT_Y_MAX - yStepSize ;       // no line at position 0
                     ystep < this._yLabels.get().length ;                      // no line at the end
                   ++ystep,     y -= yStepSize)
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
            if (!(xDelta && this._points.length > 1)) {
                this._cxt.font = _LABEL_FONT ;
                this._cxt.fillStyle = 'maroon';
                this._cxt.fillText (
                    'Not enough measuremets to plot' ,
                    this._geom.PLOT_X_MIN + Math.floor (this._xLabels.get().length / 2 - 1) * this._xLabels.step ,
                    this._geom.PLOT_Y_MIN + Math.floor (this._yLabels.get().length / 2)     * this._yLabels.step) ;
                return ;
            }

            var yDelta = this._yRange.max - this._yRange.min ;
            var yPrev = null ;

            this._cxt.lineWidth = 1;
            this._cxt.strokeStyle = _PLOT_COLOR ;
            this._cxt.beginPath() ;
            for (var i in this._points) {
                var p = this._points[i] ;
                var x = this._geom.PLOT_X_MIN + this._geom.PLOT_WIDTH  * ((p[0] - this._xRange.min) / xDelta) ,
                    y = this._geom.PLOT_Y_MAX - this._geom.PLOT_HEIGHT * ((p[1] - this._yRange.min) / yDelta) ;
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
