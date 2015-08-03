/**
 * Timeline interval definitions
 */
define ([
    'RadioBox'] ,

function (RadioBox) {

    function _pad (n) {
        return (n < 10 ? '0' : '') + n ;
    }
    function _time2htmlUTC (t) {
        return  t.getUTCFullYear() +
            '-' + _pad(t.getUTCMonth() + 1) + 
            '-' + _pad(t.getUTCDate()) +
            '&nbsp;&nbsp;<span style="font-weight:bold;">' +
            _pad(t.getUTCHours()) +
            ':' +
            _pad(t.getUTCMinutes()) +
            ':' +
            _pad(t.getUTCSeconds()) +
            '</span>' ;
    }
    function _time2htmlLocal (t) {
        return  t.getFullYear() +
            '-' + _pad(t.getMonth() + 1) + 
            '-' + _pad(t.getDate()) +
            '&nbsp;&nbsp;<span style="font-weight:bold;">' +
            _pad(t.getHours()) +
            ':' +
            _pad(t.getMinutes()) +
            ':' +
            _pad(t.getSeconds()) +
            '</span>' ;
    }
    function _date2YmdLocal (d) {
        return d.getFullYear() +
                '-' +
                _pad(d.getMonth() + 1) +
                '-' +
                _pad(d.getDate()) ;
    }
    if (!Date.prototype.toISOString) {
        // Polyfill for 
        (function () {

            Date.prototype.toISOString = function () {
                return this.getUTCFullYear() +
                    '-' + _pad(this.getUTCMonth() + 1) +
                    '-' + _pad(this.getUTCDate()) +
                    'T' + _pad(this.getUTCHours()) +
                    ':' + _pad(this.getUTCMinutes()) +
                    ':' + _pad(this.getUTCSeconds()) +
                    '.' + (this.getUTCMilliseconds() / 1000).toFixed(3).slice(2, 5) +
                    'Z' ;
            } ;
        } ()) ;
    }

    /**
     * Utility class representing a window in the timeline
     *
     * @returns {_Interval}
     */
    function _Interval () {}
    
    /**
     * Window definitions
     */
    _Interval.WINDOW_DEFS = [
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
    
    _Interval.maxZoomOut = _Interval.WINDOW_DEFS[0].name ;
    _Interval.minZoomIn  = _Interval.WINDOW_DEFS[_Interval.WINDOW_DEFS.length - 1].name ;

    /**
     * Find the name of the previous more narrow window if the one is available.
     * Return the input name if it was the first one. The function returns
     * undefined if the wrong window name is passed as the parameter.
     * 
     * @param {String} name
     * @returns {String|undefined}
     */
    _Interval.zoomIn = function  (name) {
        for (var i = 0, num = _Interval.WINDOW_DEFS.length; i < num; ++i) {
            var w = _Interval.WINDOW_DEFS[i] ;
            if (w.name === name) {
                var wPrev = _Interval.WINDOW_DEFS[i+1] ;
                return _.isUndefined(wPrev) ? name : wPrev.name ;
            }
        }
        console.log('_Interval.zoomIn: unknown window: '+name) ;
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
    _Interval.zoomOut = function  (name) {
        for (var i = 0, num = _Interval.WINDOW_DEFS.length; i < num; ++i) {
            var w = _Interval.WINDOW_DEFS[i] ;
            if (w.name === name) {
                var wNext = _Interval.WINDOW_DEFS[i-1] ;
                return _.isUndefined(wNext) ? name : wNext.name ;
            }
        }
        console.log('_Interval.zoomOut: unknown window: '+name) ;
        return undefined ;
    } ;



    var _AUTO_TRACK_INTERVAL_SEC = 0.9 ,
        _AUTO_TRACK_MAX_ZOOM = 600 ;

    function _IntervalUI (options) {

        var _that = this ;

        this._options = {
            on_change: _.isUndefined(options.on_change) ?
                function () {} :
                options.on_change ,
            changes_allowed: _.isUndefined(options.changes_allowed) ?
                function () { return true ; } :
                options.changes_allowed
        } ;
        this._on_change = function (xbins) {
            this._options.on_change(xbins) ;
        } ;

        // using the smallest time window to initialize the interval
        this.to   = new Date() ,
        this.from = new Date(+this.to - _Interval.minZoomIn * 1000) ;

        this._interval = new RadioBox (
            _Interval.WINDOW_DEFS ,
            function (zoom) { _that._timeZoom(zoom) ; } ,
            {activate: _Interval.minZoomIn }
        ) ;
        this._interval.display($('#interval')) ;

        var now = new Date() ;

        this._end_ymd = $('#end_ymd > input')
            .datepicker({
                changeMonth: true,
                changeYear: true})
            .datepicker('option', 'dateFormat', 'yy-mm-dd')
            .datepicker('setDate', _date2YmdLocal(now))
            .change(function () {
                _that._end_time_changed() ;
        }) ;
        this._end_hh = $('#end_hh > input')
            .val(_pad(now.getHours()))
            .change(function () {
                var v = parseInt($(this).val()) || 0 ;
                // Validate before triggering further actions.
                if (v > 23 || v < 0) {
                    $(this).val(_pad(_that.to.getHours())) ;
                    return ;
                }
                _that._end_time_changed() ;
        }) ;
        this._end_mm = $('#end_mm > input')
            .val(_pad(now.getMinutes()))
            .change(function () {
                var v = parseInt($(this).val()) || 0 ;
                // Validate before triggering further actions.
                if (v > 59 || v < 0) {
                    $(this).val(_pad(_that.to.getMinutes())) ;
                    return ;
                }
                _that._end_time_changed() ;
        }) ;
        this._end_ss = $('#end_ss > input')
            .val(_pad(now.getSeconds()))
            .change(function () {
                var v = parseInt($(this).val()) || 0 ;
                // Validate before triggering further actions.
                if (v > 59 || v < 0) {
                    $(this).val(_pad(_that.to.getSeconds())) ;
                    return ;
                }
                _that._end_time_changed() ;
        }) ;
        this._end_now = $('#end_now > button').button().click(function () {
            _that._end_time_changed(new Date()) ;
        }) ;
        this._end_left = $('#end_left > button').button().click(function () {
            _that.moveLeft() ;
        }) ;
        this._end_right = $('#end_right > button').button().click(function () {
            _that.moveRight() ;
        }) ;
        
        // ---------------------------------------------------------------------
        // Auto-tracking mode if enabled would automatically update plots in
        // the very end of the timeline after resetting the end time to
        // the present time.
        //
        // Note that a choice of the allowed interval sizes will also be limited
        // to a few shortest ones. This is done to prevent overloading the Web
        // services.


        this.inAutoTrackMode = function () { return this._autoTrackMode ; }

        this._autoTrackMode = false ;
        this._end_track_start = $('#end_track').find('button[name="start"]').button().click(function () {
            _that._autoTrack(true) ;
        }) ;
        this._end_track_stop = $('#end_track').find('button[name="stop"]').button().click(function () {
            _that._autoTrack(false) ;
        }) ;
        this._autoTrack = function (yes) {
            this._autoTrackMode = yes ;
            if (this._autoTrackMode) {
                this._end_track_start.closest('.auto-track-visible').removeClass('auto-track-visible').addClass('auto-track-hidden') ;
                this._end_track_stop .closest('.auto-track-hidden') .removeClass('auto-track-hidden') .addClass('auto-track-visible') ;
                
                // Reset end time to the present time
                this._end_time_changed(new Date()) ;

                // Reset zoom to the shortest one                
                this._timeZoom(_Interval.minZoomIn) ;

                // Disable all but a few shortest zoom modes
                var force = true ;
                this.disable(true, force) ;
                for (var i = 0, num = _Interval.WINDOW_DEFS.length; i < num; ++i) {
                    var w = _Interval.WINDOW_DEFS[i] ;
                    if (+w.name <= _AUTO_TRACK_MAX_ZOOM) this._interval.disable(w.name, false) ;
                }

                // Start a chain of timers
                this._startTrackTimer() ;

            } else {
                this._end_track_start.closest('.auto-track-hidden') .removeClass('auto-track-hidden').addClass('auto-track-visible') ;
                this._end_track_stop .closest('.auto-track-visible').removeClass('auto-track-visible').addClass('auto-track-hidden') ;
                this.disable(false) ;
            }
        } ;
        this._startTrackTimer = function () {
            setTimeout(function () {
                if (_that._autoTrackMode) {
                    _that._end_time_changed(new Date()) ;
                    _that._startTrackTimer() ;
                }
            } , _AUTO_TRACK_INTERVAL_SEC * 1000) ;
        } ;


        this.zoomIn = function (xbins) {
            if (!this._options.changes_allowed()) return ;

            // Stop when reaching the minimal zoom to prevent plot jittering
            var prevZoom = this._interval.active() ,
                zoom     = _Interval.zoomIn(prevZoom) ;

            if (prevZoom !== zoom) {
                this.from = new Date(+this.to - zoom * 1000) ;

                this._interval.activate(zoom) ;
                this._on_change(xbins) ;
            }
        } ;
        this.zoomOut = function (xbins) {
            if (!this._options.changes_allowed()) return ;

            // Stop when reaching the minimal zoom to prevent plot jittering
            var prevZoom = this._interval.active() ,
                zoom     = _Interval.zoomOut(prevZoom) ;

            if (prevZoom !== zoom) {
                this.from = new Date(+this.to - zoom * 1000) ;

                this._interval.activate(zoom) ;
                this._on_change(xbins) ;
            }
        } ;
        this._timeZoom = function (zoom) {
            if (!this._options.changes_allowed()) return ;

            this.from = new Date(+this.to - zoom * 1000) ;

            this._interval.activate(zoom) ;
            this._on_change() ;
        } ;
        this._timeline_change = function (range) {
            if (!this._options.changes_allowed()) return ;

            console.log('range.deltaZoom:', range.deltaZoom) ;
            var zoom = this._interval.active() ,
                to = Math.min(+(new Date()), +this.to + Math.round(zoom * 1000 * range.deltaZoom)) ;

            this.from = new Date(to - zoom * 1000) ;
            this.to   = new Date(to) ;

            // Update controls accordingly
            this._end_ymd.datepicker('setDate', _date2YmdLocal(this.to)) ;
            this._end_hh.val(_pad(this.to.getHours())) ;
            this._end_mm.val(_pad(this.to.getMinutes())) ;
            this._end_ss.val(_pad(this.to.getSeconds())) ;

            this._on_change(range.xbins) ;
        } ;
        this.moveLeft = function (xbins, dx) {
            this._timeline_change ({
                deltaZoom: -(dx ? dx / xbins : 1.) ,
                xbins: xbins
            }) ;
        } ;
        this.moveRight = function (xbins, dx) {
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
                this._end_ymd.val(_date2YmdLocal(end)) ;
                this._end_hh. val(_pad(end.getHours())) ;
                this._end_mm. val(_pad(end.getMinutes())) ;
                this._end_ss. val(_pad(end.getSeconds())) ;
            }
            var deltaMS = end - this.to ;
            this.from = new Date(+this.from + deltaMS) ;
            this.to   = new Date(+this.to   + deltaMS) ;
            this._on_change() ;
        } ;
        this.disable = function (yes, force) {
            
            // Prevent unlocking controls in the automatic tracking mode unless forced
            // to do so. The controls are managed in a special way in this mode.
            if (this._autoTrackMode && !force) return ;

            this._interval.disableAll(yes) ;

            this._end_ymd.datepicker(yes ? 'disable' : 'enable') ;

            this._end_hh.prop('disabled', yes) ;
            this._end_mm.prop('disabled', yes) ;
            this._end_ss.prop('disabled', yes) ;

            this._end_now  .button(yes ? 'disable' : 'enable') ;
            this._end_left .button(yes ? 'disable' : 'enable') ;
            this._end_right.button(yes ? 'disable' : 'enable') ;
        } ;
    }

    return {
        Interval:       _IntervalUI ,
        time2htmlUTC:   _time2htmlUTC ,
        time2htmlLocal: _time2htmlLocal
    }
}) ;