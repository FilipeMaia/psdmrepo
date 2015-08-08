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


    function test_large (obj) {

        var num_rows     =  5 ;
        var num_hdr_rows =  2 ;
        var max_hdr_rows =  4 ;
        var num_cols     = 12 ;

        var hdr = [
            '<b>Run</b>',
            'AMO:R14:IOC:21:VHS7:CH2:VoltageMeasure',
            'AmoEndstation-0|Opal1000-0',
            'FEE mirror LVDT position',
            '7-th may not be as long as others',
            'AMO:DIA:STC:07',
            'The long one again',
            '7-th may not be as long as others',
            'gasjet x-position (rel. distance)',
            'AMO:HFP:MMS:72.RBV',
            'And the third one',
            'etc.'
        ] ;
        var rows = [] ;
        for (var i=0; i < num_rows; i++) {
            var row = ['<b>'+(num_rows-i)+'</b>'] ;
            for (var j=1; j < num_cols; j++) {
                var val = (j + 1) % 7 ? 100000 * (i % 10 + 1) + j*j*j*j : 'TEXT-FIELD' ;
                row.push(val) ;
            }
            rows.push(row) ;
        }
        var table = new SmartTable(hdr, rows, num_hdr_rows, max_hdr_rows) ;
        table.display(obj) ;
    }

    function test_narrow (obj, num_hdr_rows, max_hdr_rows) {

        var num_rows     =  5 ;
        var num_cols     = 25 ;
        var hdr = [
            '<b>Run</b>',
            'One',
            'AmoEndstation-0|Opal1000-0',
            'LVDT position',
            '7-th is a bit longer than others',
            'AMO:DIA:STC:07',
            'C1',
            'The long one again',
            'empty',
            'gasjet x-position (rel. distance)',
            'C2',
            'C3',
            'AMO:HFP:MMS:72.RBV',
            'C4',
            'And the third one',
            'T1',
            'AMO:HFP:MMS:72.RBV',
            'T2',
            'T3',
            'AMO:HFP:MMS:72.RBV',
            'T4',
            'AMO:HFP:MMS:72.RBV',
            'T5',
            'etc.'
        ] ;
        var rows = [] ;
        for (var i=0; i < num_rows; i++) {
            var row = ['<b>'+(num_rows-i)+'</b>'] ;
            for (var j=1; j < num_cols; j++) {
                var val = (j + 1) % 3 ? '' : '<div style="width:100%; text-align:center; font-size:14px; color:red;">&diams;</div>' ;
                row.push(val) ;
            }
            rows.push(row) ;
        }
        var table = new SmartTable(hdr, rows, num_hdr_rows, max_hdr_rows) ;
        table.display(obj) ;
    }

    $(function () {

        test_large($('#smarttable1')) ;
        test_large($('#smarttable2')) ;

        test_narrow($('#smarttable3'), 10, 10) ;
        test_narrow($('#smarttable4'),  4,  6) ;
        test_narrow($('#smarttable5'),  4,  4) ;
        test_narrow($('#smarttable6'),  4,  2) ;
        test_narrow($('#smarttable7'),  4,  3) ;
        test_narrow($('#smarttable8'),  4,  4) ;
        test_narrow($('#smarttable9'),  4,  5) ;
        test_narrow($('#smarttable10'), 4,  6) ;
        test_narrow($('#smarttable11'), 4,  7) ;
    }) ;
}) ;