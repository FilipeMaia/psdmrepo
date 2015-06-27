define ([] ,

function () {

    /**
     * The tick labels generator for axes.
     *
     * Reference:
     * [1] An Extension of Wilkinson's Algorithm for positioning Tick Labels on Axes
     * (Justin Talbot, Sharon Lin, Pat Hanrahan)
     * Ahmet Engin Karahan, revised implementation 20.04.2015
     * 
     * See: http://vis.stanford.edu/files/2010-TickLabels-InfoVis.pdf
     */
    var Integer = {
        MAX_VALUE: Math.pow(2, 31) - 1
    } ;

    // Polyfill for environments where the function
    // is not supported.

    Math.log10 = Math.log10 || function (x) {
        return Math.log(x) / Math.LN10 ;
    } ;

    /**
     * The class which represent a result of a search for a collection
     * of labels.
     */
    function _Labels () {

        this.min ;
        this.max ;
        this.step ;
        this.score ;

        this._list = null ;
        this.get = function () {
            if (!this._list) {
                this._list = [] ;
                for (var i = this.min; i < this.max; i += this.step) {
                    this._list.push(i) ;
                }
            }
            return this._list ;
        } ;
        function _log10s (v) {
            var whole = Math.floor(Math.abs(v)) ,
                fract = Math.abs(v) - whole ;

            var whole_log10 = whole ? Math.max(1,Math.ceil(Math.log10(whole)))   : 0 ,
                fract_log10 = fract ? Math.max(2,Math.ceil(Math.log10(1/fract))) : 0 ;

            return {
                whole: {val: whole, log10: whole_log10} ,
                fract: {val: fract, log10: fract_log10}} ;
        }
        this.pretty_formatted = function () {
            if (!this._formatted) {
                this._formatted = [] ;

                // Calculate the number of fixed points
                var list_fixedPoints = [] ;
                var maxFixedPoints = 0 ;
                var list = this.get() ;
                for (var i in list) {
                    var label = list[i] ;
                    var fixedPoints = _log10s(label).fract.log10 ;
                    list_fixedPoints.push(fixedPoints) ;
                    // Prevent very large log10s around 0.
                    if (Math.abs(fixedPoints) < 16) {
                        maxFixedPoints = Math.max(maxFixedPoints, fixedPoints) ;
                    }
                }

                // Make a few iterations over pretty formatting if duplicate
                // labels are detected. Note, this only done for cases when
                // printing in the fixed point mode.

                // Generate pretty formatted labels and watch for twins
                var unique_labels = {} ,
                    twins_detected = false ;
                for (var i in list) {
                    var label = list[i];
                    var label_str = label.toFixed(maxFixedPoints) ;
                    this._formatted.push(label_str) ;
                    if (unique_labels[label_str] === undefined) {
                        unique_labels[label_str] = 1 ;
                    } else {
                        twins_detected = true ;
                        break ;
                    }
                }

                // Increase the number of fixed points and make another
                // attempst to generate labels if twins were detected.
                while (twins_detected && maxFixedPoints && maxFixedPoints < 17) {
                    maxFixedPoints++ ;
                    unique_labels = {} ;
                    twins_detected = false ;
                    this._formatted = [] ;
                    for (var i in list) {
                        var label = list[i];
                        var label_str = label.toFixed(maxFixedPoints) ;
                        this._formatted.push(label_str) ;
                        if (unique_labels[label_str] === undefined) {
                            unique_labels[label_str] = 1 ;
                        } else {
                            twins_detected = true ;
                            break ;
                        }
                    }
                }
            }
            return this._formatted ;
        } ;

        function pad (number) {
            if (number < 10) {
                return '0' + number ;
            }
            return number ;
        }
        this.pretty_formatted_timestamps = function () {
            if (!this._formatted_timestamps) {
                this._formatted_timestamps = {
                    ':SS'              : [] ,
                    ':MM:SS'           : [] ,
                    'HH:MM:SS'         : [] ,
                    'HH:MM'            : [] ,
                    'YYYY-MM-DD HH:MM' : [] ,
                    'YYYY-MM-DD'       : [] ,
                    'YYYY-MM'          : [] ,
                    'YYYY'             : []
                } ;

                var delta = this.max - this.min ;

                var list = this.get() ;
                for (var i in list) {
                    var msec = Math.round(1000 * list[i]) ;
                    var date = new Date(msec) ;
                    var SS               = ':' + pad(date.getUTCSeconds()) ,
                        MM               = ':' + pad(date.getUTCMinutes()) ,
                        MM_SS            = MM + SS ,
                        HH_MM            = pad(date.getUTCHours()) + MM ,
                        HH_MM_SS         = HH_MM + SS ,
                        YYYY             = pad(date.getUTCFullYear()) ,
                        YYYY_MM          = YYYY       + '-' + pad(date.getUTCMonth() + 1) ,
                        YYYY_MM_DD       = YYYY_MM    + '-' + pad(date.getUTCDate()) ,
                        YYYY_MM_DD_HH_MM = YYYY_MM_DD + ' ' + HH_MM ;
                    
                    this._formatted_timestamps[':SS']             .push(SS) ;
                    this._formatted_timestamps[':MM:SS']          .push(MM_SS) ;
                    this._formatted_timestamps['HH:MM']           .push(HH_MM) ;
                    this._formatted_timestamps['HH:MM:SS']        .push(HH_MM_SS) ;
                    this._formatted_timestamps['YYYY-MM-DD HH:MM'].push(YYYY_MM_DD_HH_MM) ;
                    this._formatted_timestamps['YYYY-MM-DD']      .push(YYYY_MM_DD) ;
                    this._formatted_timestamps['YYYY-MM']         .push(YYYY_MM) ;
                    this._formatted_timestamps['YYYY']            .push(YYYY) ;
                }
            }
            return this._formatted_timestamps ;
        } ;
        this.timestamp_formats = function () {
            return ['YYYY', 'YYYY-MM', 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM', 'HH:MM', 'HH:MM:SS', ':MM:SS', ':SS'] ;
        } ;
        this.has_duplicates = function (list) {
            var vPrev ;
            for (var i in list) {
                var v = list[i] ;
                if (v === vPrev) return true ;
                vPrev = v ;
            }
            return false ;
        } ;
        this.empty_duplicates = function (list) {
            var vPrev ;
            var cleared_list = [] ;
            for (var i in list) {
                var v = list[i] ;
                cleared_list.push(v === vPrev ? '' : v) ;                
                vPrev = v ;
            }
            return cleared_list ;
        } ;
    }

    /**
     * Constructor
     *
     * double[]  Q      Initial step sizes which we use as seed of generator
     * double    base   Number base used to calculate logarithms
     * double[]  w      scale-goodness weights for simplicity, coverage, density, legibility
     * double    eps    Can be injected via c'tor depending on your application, default is 1e-10
     *
     */
    function _Generator (Q, base, w, eps) {

        // Parameters of the object
        this._Q    = Q ;
        this._base = base ;
        this._w    = w === undefined ? [0.25, 0.2, 0.5, 0.05] : w ;
        this._eps  = eps === undefined ? 1e-10 : eps ;

        this.loose = true ;    // Loose flag (set externally)

        /**
         * Calculation of scale-goodness
         * 
         * double  s
         * double  c
         * double  d
         * double  l
         *
         * returns: double[]
         */
        this.w = function (s, c, d, l) {
            return this._w[0] * s + this._w[1] * c + this._w[2] * d + this._w[3] * l ;
        } ;

        /**
         * double  a
         * returns: double
         */
        this.logB = function (a) {
            return Math.log(a) / Math.log(this._base) ;
        } ;

        /**
         * a mod b for float numbers (reminder of a/b)
         *
         * double  a
         * double  n
         *
         * returns: double
         */
        this.flooredMod = function (a, n) {
            return a - n * Math.floor(a / n) ;
        } ;


        /**
         * double  min
         * double  max
         * double  step
         *
         * returns: double
         */
        this.v = function (min, max, step) {
            return (this.flooredMod(min, step) < this._eps && min <= 0 && max >= 0) ? 1 : 0 ;
        } ;

        /**
         * int     i
         * int     j 
         * double  min  
         * double  max  
         * double  step
         * returns: double
         */
        this.simplicity = function (i, j, min, max, step) {
            if (this._Q.length > 1) {
                return 1. - i / (this._Q.length - 1) - j + this.v(min, max, step) ;
            } else {
                return 1 - j + this.v(min, max, step) ;
            }
        } ;

        /**
         * int     i
         * int     j 
         * returns: double
         */
        this.simplicity_max = function (i, j) {
            if (this._Q.length > 1) {
                return 1. - i / (this._Q.length - 1) - j + 1.0 ;
            } else {
                return 1 - j + 1.0 ;
            }
        } ;

        /**
         * double  dmin
         * double  dmax
         * double  lmin
         * double  lmax
         *
         * returns: double
         */
        this.coverage = function (dmin, dmax, lmin, lmax) {
            var a = dmax - lmax ;
            var b = dmin - lmin ;
            var c = 0.1 * (dmax - dmin) ;
            return 1. - 0.5 * ((a * a + b * b) / (c * c)) ;
        } ;

        /**
         * double  dmin
         * double  dmax
         * double  span
         *
         * returns: double
         */
        this.coverage_max = function(dmin, dmax, span) {
            var range = dmax - dmin ;
            if (span > range) {
                var half = (span - range) / 2 ;
                var r = 0.1 * range ;
                return 1 - half * half / (r * r) ;
            } else {
                return 1.0 ;
            }
        } ;

        /*
         * Calculate density
         *
         * int     k        number of labels
         * int     m        number of desired labels
         * double  dmin     data range minimum
         * double  dmax     data range maximum
         * double  lmin     label range minimum
         * double  lmax     label range maximum
         *
         * returns: double
         * 
         * k-1 number of intervals between labels
         * m-1 number of intervals between desired number of labels
         * r   label interval length/label range
         * rt  desired label interval length/actual range
         */
        this.density = function (k, m, dmin, dmax, lmin, lmax) {
            var r = (k - 1) / (lmax - lmin) ;
            var rt = (m - 1) / (Math.max(lmax, dmax) - Math.min(lmin, dmin)) ;
            return 2 - Math.max(r / rt, rt / r) ;  // return 1-Math.max(r/rt, rt/r); (paper is wrong) 
        } ;

        /*
         * Calculate maximum density
         *
         * int  k   number of labels
         * int  m   number of desired labels
         *
         * returns: double
         */
        this.density_max = function (k, m) {
            if (k >= m) {
                return 2 - (k - 1) / (m - 1) ;      // return 2-(k-1)/(m-1); (paper is wrong)
            } else {
                return 1 ;
            }
        } ;

        /**
         * double  min
         * double  max
         * double step
         *
         * returns: double
         */
        this.legibility = function (min, max, step) {
            return 1 ;  // Maybe later more... 
        } ;

        /**
         *
         * double  dmin     data range min
         * double  dmax     data range max
         * int     m        desired number of labels
         *
         * returns: _Labels
         */
        this.search = function (dmin, dmax, m) {
            var best = new _Labels() ;
            var bestScore = -2 ;        // double
            var sm, dm, cm, delta ;     // double
            var j = 1 ;                 // int

            main_loop :
            while (j < Integer.MAX_VALUE) {
                for (var _i = 0; _i < this._Q.length; _i++) {
                    var i = _i + 1 ;        // int
                    var q = this._Q[_i] ;   // double
                    var sm = this.simplicity_max(i, j) ;
                    if (this.w(sm, 1, 1, 1) < bestScore) {
                        break main_loop ;
                    }
                    var k = 2 ;  // int
                    while (k < Integer.MAX_VALUE) {
                        dm = this.density_max(k, m) ;
                        if (this.w(sm, 1, dm, 1) < bestScore) {
                            break ;
                        }
                        delta = (dmax - dmin) / (k + 1) / (j * q) ;
                        var z = Math.ceil(this.logB(delta)) ;   // int
                        while (z < Integer.MAX_VALUE) {
                            var step = j * q * Math.pow(this._base, z) ;    // double
                            cm = this.coverage_max(dmin, dmax, step * (k - 1)) ;
                            if (this.w(sm, cm, dm, 1) < bestScore) {
                                break ;
                            }
                            var min_start = Math.floor(dmax / step - (k - 1)) * j ; // int
                            var max_start = Math.ceil(dmin / step) * j ;            // int

                            for (var start = min_start; start <= max_start; start++) {
                                var lmin = start * step / j ;                           // double
                                var lmax = lmin + step * (k - 1) ;                      // double
                                var c = this.coverage(dmin, dmax, lmin, lmax) ;         // double
                                var s = this.simplicity(i, j, lmin, lmax, step) ;       // double
                                var d = this.density(k, m, dmin, dmax, lmin, lmax) ;    // double
                                var l = this.legibility(lmin, lmax, step) ;             // double
                                var score = this.w(s, c, d, l) ;                        // double

                                // later legibility logic can be implemented hier

                                if (score > bestScore && (!this.loose || (lmin <= dmin && lmax >= dmax))) {
                                    best.min = lmin ;
                                    best.max = lmax ;
                                    best.step = step ;
                                    best.score = score ;
                                    bestScore = score ;
                                }
                            }
                            z = z + 1 ;
                        }
                        k = k + 1 ;
                    }
                }
                j = j + 1 ;
            }
            return best ;
        } ;
    }

    // Static factory methods
    _Generator.of = function (Q, base) { return new _Generator(Q, base) ; } ;

    // Static factory methods for vertical axes
    _Generator.base10 = function ()     { return _Generator.of([1, 5, 2, 2.5, 4, 3], 10) ; } ;
    _Generator.base2  = function ()     { return _Generator.of([1],                   2) ; }
    _Generator.base16 = function ()     { return _Generator.of([1, 2, 4, 8],         16) ; }

    // Static factory methods that may be useful for time-axis implementations 	
    _Generator.forSeconds = function () { return _Generator.of([1, 2, 3, 5, 10, 15, 20, 30], 60) ; }
    _Generator.forMinutes = function () { return _Generator.of([1, 2, 3, 5, 10, 15, 20, 30], 60) ; }
    _Generator.forHours24 = function () { return _Generator.of([1, 2, 3, 4, 6, 8, 12],       24) ; }
    _Generator.forHours12 = function () { return _Generator.of([1, 2, 3, 4, 6],              12) ; }
    _Generator.forDays    = function () { return _Generator.of([1, 2],                        7) ; }
    _Generator.forWeeks   = function () { return _Generator.of([1, 2, 4, 13, 26],            52) ; }
    _Generator.forMonths  = function () { return _Generator.of([1, 2, 3, 4, 6],              12) ; }
    _Generator.forYears   = function () { return _Generator.of([1, 2, 5],                    10) ; }

    return _Generator ;
}) ;

