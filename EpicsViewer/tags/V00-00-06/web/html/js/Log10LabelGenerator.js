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

        this.list = [] ;
        this.push = function (exponent) {

            var tick = {
                exponent: exponent ,
                value: Math.pow(10,exponent)
            } ;
            this.list.push(tick) ;

            var v = tick.value ;

            this.min = Math.min(_.isNull(this.min) ? v : this.min, v) ;
            this.max = Math.max(_.isNull(this.max) ? v : this.max, v) ;

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
                        base_text:     '10',
                        exponent_text: ''+tick.exponent
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

            var min_log10 = Math.floor(Math.log10(min)) ,
                //min_log10 = Math.ceil(Math.log10(min)) ,
                max_log10 = Math.ceil(Math.log10(max)) ;

            for (var x = max_log10; x >= min_log10; --x) {
                labels.push(x) ;
            }

            return labels ;
        } ;
    }
    return _Generator ;
}) ;