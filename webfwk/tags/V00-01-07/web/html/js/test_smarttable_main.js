require.config ({
    baseUrl: '..' ,
    paths: {
        'jquery'        : '/jquery/js/jquery-1.8.2' ,
        'jquery-ui'     : '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'jquery.resize' : '/jquery/js/jquery.resize' ,
        'underscore'    : '/underscore/underscore-min' ,
        'webfwk'        : 'webfwk/js'
    } ,
    shim : {
        'jquery' : {
            exports : '$'
        } ,
        'jquery-ui' : {
            exports : '$' ,
            deps : ['jquery']
        } ,
        'jquery.resize' :  {
            deps : ['jquery']
        } ,
        'underscore' : {
            exports  : '_'
        }
    }
}) ;

require ([
    'webfwk/CSSLoader', 'webfwk/SmartTable' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'jquery.resize', 'underscore'] ,

function (
    cssloader, SmartTable) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    $(function () {

        var num_rows     = 100 ;
        var num_hdr_rows =   2 ;
        var max_hdr_rows =   4 ;
        var num_cols     =  20 ;

        var rows = [] ;
        for (var i=0; i < num_rows; i++) {
            var row = ['<b>'+(num_rows-i)+'</b>'] ;
            for (var j=1; j < num_cols; j++) {
                var val = (j + 1) % 7 ? 100000 * (i % 10 + 1) + j*j*j*j : 'TEXT-FIELD' ;
                row.push(val) ;
            }
            rows.push(row) ;
        }
        var hdr = [
            '<b>Run</b>',
            'AMO:R14:IOC:21:VHS7:CH2:VoltageMeasure',
            'AmoEndstation-0|Opal1000-0',
            'AMO:R14:IOC:10:ao0:out2',
            'FEE mirror LVDT position',
            'AMO:R14:EVR:21:CTRL.DG0D',
            'AMO:HFP:MMS:72.RBV',
            '7-th may not be as long as others',
            'AMO:DIA:STC:07',
            'Gas attenuator calculated transmission',
            'And one more before to finish',
            'AMO:R14:IOC:21:VHS7:CH2:VoltageMeasure',
            'The long one again',
            'AMO:DIA:STC:07',
            '7-th may not be as long as others',
            'gasjet x-position (rel. distance)',
            'AMO:HFP:MMS:72.RBV',
            'And the third one',
            'AMO:DIA:STC:07',
            'etc.'
        ] ;
        var table = new SmartTable(hdr, rows, num_hdr_rows, max_hdr_rows) ;
        table.display($('#smarttable')) ;
    }) ;
}) ;