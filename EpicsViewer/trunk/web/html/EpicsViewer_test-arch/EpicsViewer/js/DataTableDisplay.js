define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class' ,
    'webfwk/Widget' ,
    'EpicsViewer/Interval'] ,

function (
    CSSLoader ,
    Class ,
    Widget ,
    Interval) {

    CSSLoader.load('../EpicsViewer/css/DataTableDisplay.css') ;

    function _Display () {

        var _that = this ;

        // Always call the c-tor of the base class
        Widget.Widget.call(this) ;

        this._isRendered = false ;  // initial rendering is done only once
        this._xRange = null ;
        this._series = null ;

        this.on_activate = function () {
            this._display() ;
        } ;

        this.load = function (xRange, series) {
            this._xRange = {
                min: xRange.min ,
                max: xRange.max
            } ;
            this._series = series ;
            this._display() ;
        } ;
        this._display = function () {

            this.resize() ;

            if (!this._isRendered) return ;
            if (_.isNull(this._xRange)) return ;

            var nextIdx = [] ;

            var html =
'<table> ' +
  '<thead> ' + 
    '<tr> ' +
      '<td>time</td> ' ;
            for (var i = 0; i < this._series.length; ++i) {
                var series = this._series[i] ;
                nextIdx.push(series.points.length ? 0 : -1) ;
                html +=
      '<td>'+series.name+'</td> ' ;
            }
            html +=
    '</tr> ' +
  '</thead> ' + 
  '<tbody> ' ;
            
            while (true) {
                var times  = [] ,
                    values = [] ,
                    found  = 0 ;
                
                for (var i = 0; i < nextIdx.length; ++i) {
                    var idx = nextIdx[i] ;
                    var points = this._series[i].points ;
                    if (idx < points.length) {
                        var p = points[idx] ;
                        times.push(Math.floor(p[0])) ;
                        values.push(p[1]) ;
                        found++ ;
                    } else {
                        times.push(+Infinity) ;
                        values.push(0) ;
                    }
                }
                if (!found) break ; // run out of points in all series
                
                var minTime = Math.min.apply(null ,times) ;

                var htmlRow =
    '<tr> ' +
      '<td>'+Interval.time2htmlUTC(new Date(minTime * 1000))+'</td> ' ;
                for (var i = 0; i < nextIdx.length; ++i) {
                    if (times[i] === minTime) {
                        htmlRow +=
      '<td><div style="color:'+this._series[i].color+';" >'+values[i]+'</div></td> ' ;
                        nextIdx[i]++ ;
                    } else {
                        htmlRow +=
      '<td>&nbsp;</td> ' ;
                    }
                }
                htmlRow +=
    '</tr> ' ;
                html += htmlRow ;
            }
            html +=
  '</tbody> ' ;
'</table> ' ;
            this.container.html(html) ;
        } ;

        this.resize = function () {
            if (!this._isRendered) return ;
            this.container.css('height', (window.innerHeight - this.container.offset().top - 70)+'px') ;            
        } ;

       /**
         * Implement the widget rendering protocol
         *
         * @returns {undefined}
         */
        this.render = function () {

            if (this._isRendered) return ;
            this._isRendered = true ;

            this.container.addClass('datatable-disp') ;

            this.resize() ;
            $(window).resize(function () {
                _that._display() ;  // redisplay is needed to prevent plots
                                    // from being scaled.
            }) ;
            this._display() ;
        } ;
    }
    Class.define_class (_Display, Widget.Widget, {}, {}) ;
    
    return _Display ;
}) ;