define ([
    'CSSLoader' ,
    'Class' ,
    'Display' ,
    'Interval'] ,

function (
    CSSLoader ,
    Class ,
    Display ,
    Interval) {

    CSSLoader.load('css/DataTableDisplay.css') ;

    function _DataTableDisplay () {

        var _that = this ;

        // Always call the c-tor of the base class
        Display.call(this) ;

        this._isRendered = false ;  // initial rendering is done only once

        this._xRange = null ;
        this._series = null ;

        // Metods implementing the Display contract
        this.on_activate   = function () { this._display() ; } ;
        this.on_deactivate = function () { } ;
        this.on_resize     = function () { this._resize() ; } ;

        this._resize = function () {
            if (!this._isRendered) return ;
            this.container.css('height', (window.innerHeight - this.container.offset().top - 70)+'px') ;            
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

            // Display is not active - no display
            if (!this.active) return ;

            this._resize() ;

            if (!this._isRendered) return ;
            if (_.isNull(this._xRange)) return ;

            var nextIdx = [] ;

            var html =
'<div id="download" > ' +
  '<a href="#" ' +
     'target="_blank" ' +
     'data="Download PV data in the CSV format" ' +
     'download="data.csv" ' +
     '><img src="img/download-32-000000.png" ></a> ' +
'</div> ' +
'<table> ' +
  '<thead> ' + 
    '<tr> ' +
      '<th>&nbsp;</th> ' ;
            for (var sid = 0; sid < this._series.length; ++sid) {
                var series = this._series[sid] ;
                nextIdx.push(series.points.length ? series.points.length - 1 : -1) ;
                html +=
      '<th>'+series.name+'</th> ' ;
            }
            html +=
    '</tr> ' +
  '</thead> ' + 
  '<tbody> ' ;

            while (true) {
                var times  = [] ,
                    values = [] ,
                    found  = 0 ;
                
                for (var sid = 0; sid < nextIdx.length; ++sid) {
                    var idx = nextIdx[sid] ;
                    var points = this._series[sid].points ;
                    if (idx >= 0) {
                        var p = points[idx] ;
                        times.push(Math.floor(p[0])) ;
                        values.push(p[1]) ;
                        found++ ;
                    } else {
                        times.push(-Infinity) ;
                        values.push(0) ;
                    }
                }
                if (!found) break ; // run out of points in all series
                
                var maxTime = Math.max.apply(null ,times) ;

                var htmlRow =
    '<tr> ' +
      '<td>'+Interval.time2htmlLocal(new Date(maxTime * 1000))+'</td> ' ;
                for (var sid = 0; sid < nextIdx.length; ++sid) {
                    if (times[sid] === maxTime) {
                        htmlRow +=
      '<td><div style="color:'+this._series[sid].color+';" >'+values[sid]+'</div></td> ' ;
                        nextIdx[sid]-- ;
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
            this.container.find('#download > a').click(function () {
                var csv = '' ;
                for (var sid = 0; sid < _that._series.length; ++sid) {
                    var series = _that._series[sid] ;
                    for (var i = 0, points = series.points ; i < points.length; ++i) {
                        var p = points[i] ,
                            sec  = Math.floor(p[0]) ,
                            nsec = Math.floor(10e9 * (p[0] - sec)) ,
                            row  = '"' + series.name + '",' + sec +',' + nsec + ',"' + p[1] + '"\n' ;
                        csv += row ;
                    }
                }
                this.href = 'data:text/comma-separated-values;charset=utf-8,' + encodeURIComponent(csv) ;
            }).mouseover(function () {
                $(this).children('img').prop('src', 'img/download-32-ff0000.png') ;
            }).mouseout(function () {
                $(this).children('img').prop('src', 'img/download-32-000000.png') ;
            }) ;
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

            this._resize() ;
            $(window).resize(function () {
                _that._display() ;  // redisplay is needed to prevent plots
                                    // from being scaled.
            }) ;
            this._display() ;
        } ;
    }
    Class.define_class (_DataTableDisplay, Display, {}, {}) ;
    
    return _DataTableDisplay ;
}) ;