define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget' ,
    'EpicsViewer/LabelGenerator'] ,

function (
    cssloader, Class, Widget ,
    LabelGenerator) {

//    cssloader.load('../EpicsViewer/css/WaveformPlot.css') ;

    var _LABEL_SIZE = 12 ;
    var _LABEL_FONT = _LABEL_SIZE+'pt Calibri' ;

    /**
     * The 2-column tabular widget representing properties and their values
     *
     * USAGE:
     * 
     *   TO BE COMPLETED...
     *
     * @returns {WaveformPlot}
     */
    function WaveformPlot () {

        var _that = this ;

        // Always call the c-tor of the base class
        Widget.Widget.call(this) ;

        // Rendering is done only once
        this._is_rendered = false ;

        this._canvas = null ;
        this._cxt = null ;

        this._y_labels = null ;

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

        this._t_range = null ;
        this._v_range = null ;
        this._elementCount = null ;
        this._points = null ;
        this._nextPoint = null ;
        this._redisplay_last_frame = false ;

        this.load = function (t_range, v_range, elementCount, points) {
            this._stopUpdateTimer() ;
            this._t_range = t_range ;
            this._v_range = v_range ;
            this._elementCount = elementCount ;
            this._points = points ;
            this._display() ;
        } ;

        this._cxt  = null ;
        this._geom = null ;

        this._init_drawing_context = function () {

            if (!this._canvas) {
                this._canvas = this.container ;
                this._canvas.css('height', (window.innerHeight-this._canvas.offset().top - 30)+'px') ;
                $(window).resize (function () {
                    _that._canvas.css('height', (window.innerHeight-_that._canvas.offset().top - 30)+'px') ;
                    // redisplay needed to prevent plots from being scaled.
                    _that._redisplay_last_frame = true ;
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
                }) ;
            }
            var css_width  = this._canvas.css('width') .substr(0, this._canvas.css('width') .length - 2) ,
                css_height = this._canvas.css('height').substr(0, this._canvas.css('height').length - 2) ;

            this._canvas.attr('width',  css_width) ;
            this._canvas.attr('height', css_height) ;

            this._cxt = this._canvas.get(0).getContext('2d') ;

            // half-step translation to prevent line blurring
            this._cxt.translate(0.5, 0.5) ;

            // geometry of plots
            this._geom = {} ;
            this._geom.WIDTH  = this._canvas.width() ;
            this._geom.HEIGHT = this._canvas.height() ;
            this._geom.WIDTH_OFFSET = Math.floor(this._geom.WIDTH  / 20.) ;
            this._geom.HEIGH_OFFSET = Math.floor(this._geom.HEIGHT / 20.) ;
            this._geom.PLOT_X_MIN   =                     40 ;
            this._geom.PLOT_X_MAX   = this._geom.WIDTH  - 40 ;
            this._geom.PLOT_Y_MIN   =                     40 ;
            this._geom.PLOT_Y_MAX   = this._geom.HEIGHT - 40 ;
            this._geom.PLOT_WIDTH   = this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN ;
            this._geom.PLOT_HEIGHT  = this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN ;

            this._geom.X_TICKS = 10 ;

            // Calculate a desired number of ticks based on the size of
            // the viewing area and font size of teh labels.

            var labels_generator = new LabelGenerator.base10() ;
            this._y_labels = labels_generator.search (
                this._v_range ? this._v_range.min : 0. ,
                this._v_range ? this._v_range.max : 1. ,
                Math.max (
                    2 ,
                    Math.min (
                        Math.floor(this._geom.PLOT_HEIGHT / (2 * _LABEL_SIZE)) ,
                        20
                    )
                )
            ) ;
            console.log('y labels:' , {
                min:   this._y_labels.min ,
                max:   this._y_labels.max ,
                step:  this._y_labels.step ,
                ticks: this._y_labels.get()}) ;
            
            this._geom.Y_TICKS = this._y_labels.get().length ;

            this._geom.X_TICK_STEP = Math.floor((this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN) / this._geom.X_TICKS) ;
            this._geom.Y_TICK_STEP = Math.floor((this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN) / this._geom.Y_TICKS) ;
            this._geom.X_TICK_HALF_HEIGHT = 4 ;
            this._geom.Y_TICK_HALF_WIDTH  = 4 ;

            console.log('this._geom', this._geom) ;
        } ;

        this._clear = function () {
            this._cxt.clearRect(0, 0, this._geom.WIDTH, this._geom.HEIGHT) ;
        } ;
        this._display = function () {

            // Noting to display if data haven't loaded yet
            if (!this._points) return ;

            // Do not display if there is an on-going display
            if (this._updateInProgress()) {
                console.log('_display: updateInProgress') ;
                return ;
            }

            // Initiate the plot
            this._init_drawing_context() ;
            this._clear() ;
            this._prepareAxes() ;
            this._plotData() ;
        } ;

        this._prepareAxes = function () {

            var color = '#b0b0b0' ;

            // X Axis
            this._cxt.beginPath() ;
            this._cxt.moveTo(this._geom.PLOT_X_MIN, this._geom.PLOT_Y_MAX) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX, this._geom.PLOT_Y_MAX) ;
            for (var x = this._geom.PLOT_X_MIN + this._geom.X_TICK_STEP, step = 0; step < this._geom.X_TICKS-1 ; ++step, x += this._geom.X_TICK_STEP) {
                this._cxt.moveTo(x, this._geom.PLOT_Y_MAX + this._geom.X_TICK_HALF_HEIGHT) ;
                this._cxt.lineTo(x, this._geom.PLOT_Y_MAX - this._geom.X_TICK_HALF_HEIGHT) ;
            }
            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = color ;
            this._cxt.stroke();
            /*
             * Arrow ending
             * 
            this._cxt.beginPath() ;
            this._cxt.moveTo(this._geom.PLOT_X_MAX - this._geom.Y_TICK_HALF_WIDTH, this._geom.PLOT_Y_MAX - this._geom.X_TICK_HALF_HEIGHT) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX,                                this._geom.PLOT_Y_MAX) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX - this._geom.Y_TICK_HALF_WIDTH, this._geom.PLOT_Y_MAX + this._geom.X_TICK_HALF_HEIGHT) ;
            this._cxt.lineTo(this._geom.PLOT_X_MAX - this._geom.Y_TICK_HALF_WIDTH, this._geom.PLOT_Y_MAX - this._geom.X_TICK_HALF_HEIGHT) ;
            this._cxt.lineWidth = 1;
            this._cxt.fillStyle = color ;
            this._cxt.fill();
            this._cxt.stroke();
*/

            // Y Axis
            this._cxt.beginPath() ;
            this._cxt.moveTo(this._geom.PLOT_X_MIN, this._geom.PLOT_Y_MAX) ;
            this._cxt.lineTo(this._geom.PLOT_X_MIN, this._geom.PLOT_Y_MIN) ;
            for (var y = this._geom.PLOT_Y_MAX - this._geom.Y_TICK_STEP, step = 0; step < this._geom.Y_TICKS-1 ; ++step,  y -= this._geom.Y_TICK_STEP) {
                this._cxt.moveTo(this._geom.PLOT_X_MIN - this._geom.Y_TICK_HALF_WIDTH, y) ;
                this._cxt.lineTo(this._geom.PLOT_X_MIN + this._geom.Y_TICK_HALF_WIDTH, y) ;
            }
            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = color ;
            this._cxt.stroke();
            /*
             * Arrow ending
             * 
            this._cxt.beginPath() ;
            this._cxt.moveTo(this._geom.PLOT_X_MIN - this._geom.Y_TICK_HALF_WIDTH, this._geom.PLOT_Y_MIN + this._geom.X_TICK_HALF_HEIGHT) ;
            this._cxt.lineTo(this._geom.PLOT_X_MIN,                               this._geom.PLOT_Y_MIN) ;
            this._cxt.lineTo(this._geom.PLOT_X_MIN + this._geom.Y_TICK_HALF_WIDTH, this._geom.PLOT_Y_MIN + this._geom.X_TICK_HALF_HEIGHT) ;
            this._cxt.lineTo(this._geom.PLOT_X_MIN - this._geom.Y_TICK_HALF_WIDTH, this._geom.PLOT_Y_MIN + this._geom.X_TICK_HALF_HEIGHT) ;
            this._cxt.lineWidth = 1;
            this._cxt.fillStyle = color ;
            this._cxt.fill();
            this._cxt.stroke();
            */
        } ;


        var TIMER_INTERVAL = 20 ;

        this._timer = null ;

        this._updateInProgress = function () {
            return !_.isNull(this._timer) ;
        } ;
        this._startUpdateTimer = function () {
            this._timer = setTimeout (
                function () {
                    _that._plotStep() ;
                } ,
                TIMER_INTERVAL
            ) ;
        } ;
        this._stopUpdateTimer = function () {
            if (this._updateInProgress()) {
                clearTimeout(this._timer) ;
                this._timer = null ;
            }
        } ;
        this._plotData = function () {
            this._nextPoint = 0 ;
            this._redisplay_last_frame = false ;
            if (this._points.length) this._startUpdateTimer() ;
        } ;
        this._plotStep = function () {
 
            var y_prev = null ;

            // Clear display area and do not touch axeses
            this._clear() ;
            this._prepareAxes() ;

            /*
            this._cxt.clearRect (this._geom.PLOT_X_MIN ,
                                 this._geom.PLOT_Y_MIN ,
                                 this._geom.PLOT_X_MAX - this._geom.PLOT_X_MIN ,
                                 this._geom.PLOT_Y_MAX - this._geom.PLOT_Y_MIN) ;
            */

            this._cxt.lineWidth = 1 ;
            this._cxt.strokeStyle = 'blue' ;
            this._cxt.beginPath() ;

            var v_delta = this._v_range.max - this._v_range.min ;

            var point = this._points[this._nextPoint] ;
            var t = point[0] ;
            var p = point[1] ;
            for (var j in p) {
                var v = p[j] ;
                var x = this._geom.PLOT_X_MIN + this._geom.PLOT_WIDTH  * (j / this._elementCount) ,
                    y = this._geom.PLOT_Y_MAX - this._geom.PLOT_HEIGHT * ((v - this._v_range.min) / v_delta) ;
                if (_.isNull(y_prev)) {
                    this._cxt.moveTo(x, y) ;
                } else {
                    this._cxt.lineTo(x, y_prev) ;
                    this._cxt.lineTo(x, y) ;
                }
                y_prev = y ;
            }
            this._cxt.stroke();
            this._cxt.clearRect (this._geom.PLOT_X_MAX - this._geom.X_TICK_STEP * 4 ,
                                 this._geom.PLOT_Y_MIN ,
                                 this._geom.X_TICK_STEP * 4 ,
                                 this._geom.Y_TICK_STEP * -1) ;

            this._cxt.font      = _LABEL_FONT ;
            this._cxt.fillStyle = 'maroon' ;
            this._cxt.fillText (t ,
                                this._geom.PLOT_X_MAX - this._geom.X_TICK_STEP * 4 ,
                                this._geom.PLOT_Y_MIN) ;

            this._cxt.fillText ('timestep '+(this._nextPoint + 1)+' of '+this._points.length ,
                                this._geom.PLOT_X_MAX - this._geom.X_TICK_STEP * 4 ,
                                this._geom.PLOT_Y_MIN + 16) ;

            this._nextPoint++ ;
            if (this._nextPoint < this._points.length) {
                this._startUpdateTimer() ;
            } else {
                // In case if there was a request to refresh the last frame
                if (this._redisplay_last_frame) {
                    this._redisplay_last_frame = false ;
                    this._init_drawing_context() ;
                    this._clear() ;
                    this._prepareAxes() ;
                    this._nextPoint = this._points.length - 1 ;
                    this._startUpdateTimer() ;
                } else {
                    this._stopUpdateTimer() ;
                }
            }
        } ;
    }
    Class.define_class(WaveformPlot, Widget.Widget, {}, {}) ;

    return WaveformPlot ;
}) ;