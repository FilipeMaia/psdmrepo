/* 
 * This is the test application to develop an algorithm for generating
 * log-scale labels.
 */



// Polyfill for environments where the function
// is not supported.

Math.log10 = Math.log10 || function (x) {
    return Math.log(x) / Math.LN10 ;
} ;

function generateLabels (range) {

    var negative_min  = range.min < 0  ,
        negative_max  = range.max < 0 ,
        abs_min       = Math.abs(range.min) ,
        abs_max       = Math.abs(range.max) ,
        abs_min_log10 = Math.log10(abs_min) ,
        abs_max_log10 = Math.log10(abs_max) ,
        min_log10     = negative_min ? Math.ceil (abs_min_log10) : Math.floor(abs_min_log10) ,
        max_log10     = negative_max ? Math.floor(abs_max_log10) : Math.ceil (abs_max_log10) ;

    print('          min: '+range.min) ;
    print('          max: '+range.max) ;
    print('log10 abs min: '+abs_min_log10) ;
    print('          max: '+abs_max_log10) ;
    print('adj log10 min: '+min_log10) ;
    print('adj log10 max: '+max_log10) ;

    print('labels:') ;
    
    if (negative_min) {
        if (negative_max) {    
            for (var x=max_log10; x <= min_log10; ++x) {
                var label = '-10^'+x ;
                print('  '+label) ;
            }
        } else {
            if (max_log10 >= 0) {
                for (var x=max_log10; x >= 0; --x) {
                    var label = ' 10^'+x ;
                    print('  '+label) ;
                }
            }
            if (max_log10 < 0 || min_log10 < 0) {
                for (var x = -1; x >= Math.min(max_log10, min_log10); --x) {
                    var label = ' 10^'+x ;
                    print('  '+label) ;
                }
            }
            var label = '  0' ;
            print('  '+label) ;
            if (max_log10 < 0 || min_log10 < 0) {
                for (var x = Math.min(max_log10, min_log10); x <= -1; ++x) {
                    var label = '-10^'+x ;
                    print('  '+label) ;
                }
            }
            if (min_log10 >=0 )
            for (var x=0; x <= min_log10; ++x) {
                var label = '-10^'+x ;
                print('  '+label) ;
            }
        }
    } else {
        for (var x=max_log10; x >= min_log10; --x) {
            var label = ' 10^'+x ;
            print('  '+label) ;
        }
    }
}


// Read the range from the input

var args = readline().split(' ') ;
if (args.length !== 2) {
    print('Usage: <min> <max>') ;
    exit ;
}
var range = {
    min: +args[0] ,
    max: +args[1]
} ; 
generateLabels(range) ;