define ([] ,

function () {

    // Polyfill for environments where the function
    // is not supported.

    Math.log10 = Math.log10 || function (x) {
        return Math.log(x) / Math.LN10 ;
    } ;

    /**
     * The class which represent a result of a search for a collection
     * of labels.
     */
    function _Labels (step) {

        this.min = null ;
        this.max = null ;

        this.step = step ;

        this.maxNegative = null ;
        this.minPositive = null ;

        this.list = [] ;
        this.push = function (tick) {

            this.list.push(tick) ;

            var v = tick.value ;

            this.min = Math.min(_.isNull(this.min) ? v : this.min, v) ;
            this.max = Math.max(_.isNull(this.max) ? v : this.max, v) ;

            if (v < 0) this.maxNegative = Math.max(_.isNull(this.maxNegative) ? v : this.maxNegative, v) ;
            else       this.minPositive = Math.max(_.isNull(this.minPositive) ? v : this.minPositive, v) ;
        } ;
        this.pushNegative = function (v) {
            this.push({
                sign: -1 ,
                exponent: v ,
                value: -1*Math.pow(10,v)
            }) ;
        } ;
        this.pushPositive = function (v) {
            this.push({
                sign: 1 ,
                exponent: v ,
                value: Math.pow(10,v)
            }) ;
        } ;
        this._formatted = null ;
        this.pretty_formatted = function () {
            if (!this._formatted) {
                this._formatted = [] ;
                for (var i = 0; i < this.list.length; ++i) {
                    var tick = this.list[i] ;
                    this._formatted.push({
                        exponent: tick.exponent ,
                        value: tick.value ,
                        base_text:     tick.value ? ''+(tick.sign*10) : '0' ,
                        exponent_text: tick.value ? ''+tick.exponent  : ''
                    }) ;
                }
                this._formatted.reverse() ;
            }
            return this._formatted ;
        } ;
        this.get = this.pretty_formatted ;
    }

    function _Generator () {

        this.search = function (min, max, m) {

            var labels = new _Labels(10) ; ;

            var negative_min  = min < 0  ,
                negative_max  = max < 0 ,
                abs_min       = Math.abs(min) ,
                abs_max       = Math.abs(max) ,
                abs_min_log10 = Math.log10(abs_min) ,
                abs_max_log10 = Math.log10(abs_max) ,
                min_log10     = negative_min ? Math.ceil (abs_min_log10) : Math.floor(abs_min_log10) ,
                max_log10     = negative_max ? Math.floor(abs_max_log10) : Math.ceil (abs_max_log10) ;

            if (negative_min) {
                if (negative_max) {    
                    for (var x = max_log10; x <= min_log10; ++x) {
                        labels.pushNegative(x) ;
                    }
                } else {
                    if (max_log10 >= 0) {
                        for (var x = max_log10; x >= 0; --x) {
                            labels.pushPositive(x) ;
                        }
                    }
                    if (max_log10 < 0 || min_log10 < 0) {
                        for (var x = -1; x >= Math.min(max_log10, min_log10); --x) {
                            labels.pushPositive(x) ;
                        }
                    }
                    labels.push(0) ;
                    if (max_log10 < 0 || min_log10 < 0) {
                        for (var x = Math.min(max_log10, min_log10); x <= -1; ++x) {
                            labels.pushNegative(x) ;
                        }
                    }
                    if (min_log10 >=0 )
                    for (var x = 0; x <= min_log10; ++x) {
                        labels.pushNegative(x) ;
                    }
                }
            } else {
                for (var x = max_log10; x >= min_log10; --x) {
                    labels.pushPositive(x) ;
                }
            }
            return labels ;
        } ;
    }
    return _Generator ;
}) ;