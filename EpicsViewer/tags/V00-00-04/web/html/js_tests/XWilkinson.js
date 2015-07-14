/**
 * Reference:
 * [1] An Extension of Wilkinson's Algorithm for positioning Tick Labels on Axes
 * (Justin Talbot, Sharon Lin, Pat Hanrahan)
 * Ahmet Engin Karahan, revised implementation 20.04.2015
 * 
 * See: http://vis.stanford.edu/files/2010-TickLabels-InfoVis.pdf
 */


var Integer = {
    MAX_VALUE: Math.pow(2, 31) - 1
};

/**
 * The class which represent a result of a search for a collection
 * of labels.
 */
function Labels() {

    this.min;
    this.max;
    this.step;
    this.score;

    this.getList = function () {
        var list = [] ;
        for (var i = this.min; i < this.max; i += this.step) {
            list.push(i);
        }
        return list;
    } ;
    this.getMin   = function () { return this.min; } ;
    this.getMax   = function () { return this.max; } ;
    this.getStep  = function () { return this.step; } ;
    this.getScore = function () { return this.score; } ;
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
function LabelGenerator (Q, base, w, eps) {

    // Parameters of the object
    this._Q    = Q;
    this._base = base;
    this._w    = w === undefined ? [0.25, 0.2, 0.5, 0.05] : w;
    this._eps  = eps === undefined ? 1e-10 : eps;

    this.loose = true;     // Loose flag (set externally)

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
        return this._w[0] * s + this._w[1] * c + this._w[2] * d + this._w[3] * l;
    } ;

    /**
     * double  a
     * returns: double
     */
    this.logB = function (a) {
        return Math.log(a) / Math.log(this._base);
    } ;

    /**
     * a mod b for float numbers (reminder of a/b)
     *
     * double  a
     * double  n
     *
     * returns: double
     */
    this.flooredMod = function(a, n) {
        return a - n * Math.floor(a / n);
    } ;


    /**
     * double  min
     * double  max
     * double  step
     *
     * returns: double
     */
    this.v = function(min, max, step) {
        return (this.flooredMod(min, step) < this._eps && min <= 0 && max >= 0) ? 1 : 0;
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
            return 1. - i / (this._Q.length - 1) - j + this.v(min, max, step);
        } else {
            return 1 - j + this.v(min, max, step);
        }
    } ;

    /**
     * int     i
     * int     j 
     * returns: double
     */
    this.simplicity_max = function (i, j) {
        if (this._Q.length > 1) {
            return 1. - i / (this._Q.length - 1) - j + 1.0;
        } else {
            return 1 - j + 1.0;
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
        var a = dmax - lmax;
        var b = dmin - lmin;
        var c = 0.1 * (dmax - dmin);
        return 1. - 0.5 * ((a * a + b * b) / (c * c));
    } ;

    /**
     * double  dmin
     * double  dmax
     * double  span
     *
     * returns: double
     */
    this.coverage_max = function(dmin, dmax, span) {
        var range = dmax - dmin;
        if (span > range) {
            var half = (span - range) / 2;
            var r = 0.1 * range;
            return 1 - half * half / (r * r);
        } else {
            return 1.0;
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
        var r = (k - 1) / (lmax - lmin);
        var rt = (m - 1) / (Math.max(lmax, dmax) - Math.min(lmin, dmin));
        return 2 - Math.max(r / rt, rt / r);   // return 1-Math.max(r/rt, rt/r); (paper is wrong) 
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
            return 2 - (k - 1) / (m - 1);        // return 2-(k-1)/(m-1); (paper is wrong)
        } else {
            return 1;
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
        return 1; // Maybe later more... 
    } ;

    /**
     *
     * double  dmin     data range min
     * double  dmax     data range max
     * int     m        desired number of labels
     *
     * returns: Labels
     */
    this.search = function (dmin, dmax, m) {
        var best = new Labels();
        var bestScore = -2;         // double
        var sm, dm, cm, delta;      // double
        var j = 1;                  // int

        main_loop:
        while (j < Integer.MAX_VALUE) {
            for (var _i = 0; _i < this._Q.length; _i++) {
                var i = _i + 1;         // int
                var q = this._Q[_i];    // double
                var sm = this.simplicity_max(i, j);
                if (this.w(sm, 1, 1, 1) < bestScore) {
                    break main_loop;
                }
                var k = 2;  // int
                while (k < Integer.MAX_VALUE) {
                    dm = this.density_max(k, m);
                    if (this.w(sm, 1, dm, 1) < bestScore) {
                        break;
                    }
                    delta = (dmax - dmin) / (k + 1) / (j * q);
                    var z = Math.ceil(this.logB(delta));   // int
                    while (z < Integer.MAX_VALUE) {
                        var step = j * q * Math.pow(this._base, z);   // double
                        cm = this.coverage_max(dmin, dmax, step * (k - 1));
                        if (this.w(sm, cm, dm, 1) < bestScore) {
                            break;
                        }
                        var min_start = Math.floor(dmax / step - (k - 1)) * j;  // int
                        var max_start = Math.ceil(dmin / step) * j;             // int

                        for (var start = min_start; start <= max_start; start++) {
                            var lmin = start * step / j;                            // double
                            var lmax = lmin + step * (k - 1);                       // double
                            var c = this.coverage(dmin, dmax, lmin, lmax);          // double
                            var s = this.simplicity(i, j, lmin, lmax, step);        // double
                            var d = this.density(k, m, dmin, dmax, lmin, lmax);     // double
                            var l = this.legibility(lmin, lmax, step);              // double
                            var score = this.w(s, c, d, l);                         // double

                            // later legibility logic can be implemented hier
                            
                            if (score > bestScore && (!this.loose || (lmin <= dmin && lmax >= dmax))) {
                                best.min = lmin;
                                best.max = lmax;
                                best.step = step;
                                best.score = score;
                                bestScore = score;
                            }
                        }
                        z = z + 1;
                    }
                    k = k + 1;
                }
            }
            j = j + 1;
        }
        return best;
    } ;
}

// Static factory methods
LabelGenerator.of = function (Q, base) { return new LabelGenerator(Q, base); } ;

// Static factory methods for vertical axes
LabelGenerator.base10 = function ()     { return LabelGenerator.of([1, 5, 2, 2.5, 4, 3], 10); } ;
LabelGenerator.base2  = function ()     { return LabelGenerator.of([1],                   2); }
LabelGenerator.base16 = function ()     { return LabelGenerator.of([1, 2, 4, 8],         16); }

// Static factory methods that may be useful for time-axis implementations 	
LabelGenerator.forSeconds = function () { return LabelGenerator.of([1, 2, 3, 5, 10, 15, 20, 30], 60); }
LabelGenerator.forMinutes = function () { return LabelGenerator.of([1, 2, 3, 5, 10, 15, 20, 30], 60); }
LabelGenerator.forHours24 = function () { return LabelGenerator.of([1, 2, 3, 4, 6, 8, 12],       24); }
LabelGenerator.forHours12 = function () { return LabelGenerator.of([1, 2, 3, 4, 6],              12); }
LabelGenerator.forDays    = function () { return LabelGenerator.of([1, 2],                        7); }
LabelGenerator.forWeeks   = function () { return LabelGenerator.of([1, 2, 4, 13, 26],            52); }
LabelGenerator.forMonths  = function () { return LabelGenerator.of([1, 2, 3, 4, 6],              12); }
LabelGenerator.forYears   = function () { return LabelGenerator.of([1, 2, 5],                    10); }


function toFixedStr(x, len) {
    var _str = ' '+x;
    var _len = (len === undefined ? 16 : len) - _str.length;
    if (_len > 0) {
        for (var i=0; i < _len; ++i) {
            _str = _str+' ';
        }
    }
    return _str;
}

function print_header() {
    print(toFixedStr('min')           +'|'+toFixedStr('max')           +'|'+toFixedStr('ticks')         +'|'+toFixedStr('loose')         +'|'+'labels');
    print_separator();
}
function print_separator() {
    print(toFixedStr('--------------')+'+'+toFixedStr('--------------')+'+'+toFixedStr('--------------')+'+'+toFixedStr('--------------')+'+'+'--------------');
}
function search_and_print (x, loose, dmin, dmax, num_labels) {
    x.loose = loose;
    var labels = x.search(dmin, dmax, num_labels).getList();
    print(toFixedStr(dmin) +'|'+toFixedStr(dmax) +'|'+toFixedStr(num_labels)+'|'+toFixedStr(loose)  +'|'+' ' +labels);
}
function search (x, dmin, dmax, num_labels) {
    search_and_print (x, true,  dmin, dmax, num_labels);
    search_and_print (x, false, dmin, dmax, num_labels);
    print_separator();
}

// Demo for usage
function main() {

/* THIS ALL WORKS. DISABLED IT TO RUN SOME OTHER TESTS BELOW

    print();
    print("Some additional tests: Testing with base10");
    print();
    print_header();

    var x = LabelGenerator.base10();

    search(x, -130000,     8234567,      10);
    search(x,      -0.01,        0.0221,  5);
    search(x,     -98.0,        18.0,    10);
    search(x,      -1.0,       200.0,     5);
    search(x,     119.0,       178.0,     5);
    search(x,     -31.0,        27.0,     4);
    search(x,     -55.45,      -49.99,    2);
    for (var num_ticks = 2; num_ticks <= 10; ++num_ticks)
        search(x, -10, 100, num_ticks);

    print();
    print("Some additional tests: Testing with base2");
    print();
    print_header();

    x = LabelGenerator.base2();

    search(x, 0, 32, 8);

    print();
    print("Quick experiment with minutes: Check the logic");
    print();
    print_header();

    x = LabelGenerator.forMinutes();
    
    search(x, 0, 240, 16);
    search(x, 0, 240,  9);
*/

//    print("Quick experiment with minutes: Convert values to HH:mm");
//    LocalTime start = LocalTime.now();
//    LocalTime end = start.plusMinutes(245); // add 4 hrs 5 mins (245 mins) to the start
//
//    int dmin = start.toSecondOfDay() / 60;
//    int dmax = end.toSecondOfDay() / 60;
//    if (dmin > dmax) {
//        // if adding 4 hrs exceeds the midnight simply swap the values this is just an
//        // example...
//        int swap = dmin;
//        dmin = dmax;
//        dmax = swap;
//    }
//    print("dmin: " + dmin + " dmax: " + dmax);
//    XWilkinson.Label labels = x.search(dmin, dmax, 15);
//    print("labels");
//    for (double time = labels.getMin(); time < labels.getMax(); time += labels.getStep()) {
//        LocalTime lt = LocalTime.ofSecondOfDay(Double.valueOf(time).intValue() * 60);
//        print(lt);
//    }

    print("Testing pretty print");

    // Polyfill for environments where the function
    // is not supported.

    Math.log10 = Math.log10 || function (x) {
        return Math.log(x) / Math.LN10 ;
    } ;

    function log10s (v) {
        var whole = Math.floor(Math.abs(v)),
            fract = Math.abs(v) - whole;

        var whole_log10 = whole ? Math.max(1,Math.ceil(Math.log10(whole)))   : 0,
            fract_log10 = fract ? Math.max(2,Math.ceil(Math.log10(1/fract))) : 0;
    
        return {
            whole: {val: whole, log10: whole_log10},
            fract: {val: fract, log10: fract_log10}};
    }
    function right_justified(str, max_len) {
        var just = str,
            str_len = str.length;
        for (var i = 0; i < max_len - str_len; ++i) {
            just = ' '+just;
        }
        return just;
    }

    var args = readline().split(' ');
    if (args.length !== 2) {
        print('Usage: <dmin> <dmax>');
        return;
    }
    var drange = [
        +args[0],
        +args[1]];
    print('drange', drange);

    var x = LabelGenerator.base10();

    var labels = x.search(drange[0], drange[1], 10);
    print('labels (min,max,step)', [labels.min,labels.max,labels.step]);

    var list = labels.getList();
    if (list.length) {
        print('list', list);

        var lrange = [list[0], list[list.length-1]];
        print('lrange', lrange);

        var lmax = lrange[1],
            lmin = lrange[0],
            lmax_log10s = log10s(lmax),
            lmin_log10s = log10s(lmin);
        print(
            'lmax:',           lmax,
            '\t(w,f)',        [lmax_log10s.whole.val,   lmax_log10s.fract.val],
            '\tlog10: (w,f)', [lmax_log10s.whole.log10, lmax_log10s.fract.log10]);
        print(
            'lmin:',           lmin,
            '\t(w,f)',        [lmin_log10s.whole.val,   lmin_log10s.fract.val],
            '\tlog10: (w,f)', [lmin_log10s.whole.log10, lmin_log10s.fract.log10]);

        var numFixedPoints = Math.max(lmax_log10s.fract.log10, lmin_log10s.fract.log10);
        print('[calculated] numFixedPoints', numFixedPoints);
        for (var i in list) {
            var label = list[i];
            var label_log10s = log10s(label);
            // Prevent very large log10s around 0.
            if (Math.abs(label_log10s.fract.log10 - numFixedPoints) < 3) {
                numFixedPoints = Math.max(numFixedPoints, label_log10s.fract.log10);
            }
        }
        print('[adjusted]   numFixedPoints', numFixedPoints);
        print('[formatted] labels');
        var label_width = 0;
        for (var i in list) {
            var label = list[i];
            var label_str = label.toFixed(numFixedPoints);
            print(label_str, '\t', label);
            label_width = Math.max(label_width, label_str.length + (label < 0 ? 1 : 0)) ;
        }
        print('[formatted] labels');
        for (var i in list) {
            var label = list[i];
            var label_str = label.toFixed(numFixedPoints);
            print(right_justified(label_str, label_width));
        }        
    }
}
main();

