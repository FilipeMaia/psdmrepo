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
        'underscore' : {
            exports  : '_'
        }
    }
}) ;

require ([
    'webfwk/CSSLoader', 'webfwk/SimpleTable', 'webfwk/Class' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (
    cssloader, SimpleTable, Class) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    $(function () {

        var table1 = new SimpleTable.constructor (
            'table1' ,
            [
                {   name: '1'} ,
                {   name: '2'} ,
                {   name: '3', sorted: false}
            ] ,
            [
                ['1(1)','2(2)','3(2)'] ,
                ['2(1)','3(2)','4(2)'] ,
                ['3(1)','4(2)','1(2)'] ,
                ['4(1)','1(2)','2(2)']
            ]
        ) ;
        table1.display() ;

        var table2 = new SimpleTable.constructor (
            'table2' ,
            [
                {name: 'id'} ,
                {   name: '2' ,
                    coldef: [
                        {   name: '2.1'} ,
                        {   name: '2.2'}
                    ]} ,
                {name: '3'} ,
                {name: '4'}
            ] ,
            [   ['1', '1(2.1)', '1(2.2)', '1(3)', '1(4)'] ,
                ['2', '2(2.1)', '2(2.2)', '2(3)', '2(4)'] ,
                ['3', '3(2.1)', '3(2.2)', '3(3)', '3(4)'] ,
                ['4', '4(2.1)', '4(2.2)', '4(3)', '4(4)']
            ]
        ) ;
        table2.display() ;

        var table3 = new SimpleTable.constructor (

            'table3' ,

            [
                {   name:   'id'} ,
                {   name:   '2' ,
                    coldef: [
                        {   name:   '2.1' ,
                            coldef: [
                                {   name:   '2.1.1' } ,
                                {   name:   '2.1.2' }]} ,
                        {   name:   '2.2' }]} ,
                {   name:   '3' } ,
                {   name:   '4' }] ,

            [   ['1', '1(2.1.1)', '1(2.1.2)', '1(2.2)', '1(3)', '1(4)'] ,
                ['2', '2(2.1.1)', '2(2.1.2)', '2(2.2)', '2(3)', '2(4)'] ,
                ['3', '3(2.1.1)', '3(2.1.2)', '3(2.2)', '3(3)', '3(4)'] ,
                ['4', '4(2.1.1)', '4(2.1.2)', '4(2.2)', '4(3)', '4(4)']]

        ) ;
        table3.display() ;

        function MyCellType () {
            SimpleTable.CellType.call(this) ;
        }
        Class.define_class (MyCellType, SimpleTable.CellType, {}, {
           to_string     : function (a)   { return '<b>'+a+'</b>' ; } ,
           compare_values: function (a,b) { return this.compare_strings(a,b) ; }}
        ) ;

        var table4 = new SimpleTable.constructor (
            'table4' ,
            [
                {   name: 'Text_URL',   type: SimpleTable.Types.Text_URL} ,
                {   name: 'Number_URL', type: SimpleTable.Types.Number_URL} ,
                {   name: 'MyCellType', type: new MyCellType} ,
                {   name: 'Customized', type: { to_string     : function (a)   { return SimpleTable.html.Button(a.data, {name: a.data, classes: 'my_button'}) ; } ,
                                                compare_values: function (a,b) { return a.data - b.data ; } ,
                                                after_sort    : function ()    { $('.my_button').button().click(function () { alert(this.name) ; }) ; }}}
            ] ,
            null ,
            SimpleTable.Status.Empty
        ) ;
        table4.display() ;

        var data4 = [
            [{text: 'A',         url: 'https://www.slac.stanford.edu'}, {number: 123, url: 'https://www.slac.stanford.edu'}, '3(2)', {data:  3}] ,
            [{text: 'a',         url: 'https://www.slac.stanford.edu'}, {number: -99, url: 'https://www.slac.stanford.edu'}, '4(2)', {data: 11}] ,
            [{text: 'xYz',       url: 'https://www.slac.stanford.edu'}, {number:   3, url: 'https://www.slac.stanford.edu'}, '1(2)', {data: 12}] ,
            [{text: 'let it be', url: 'https://www.slac.stanford.edu'}, {number:   0, url: 'https://www.slac.stanford.edu'}, '2(2)', {data:  1}]
        ] ;
        $('#table4_load') .button().click(function () { table4.load(data4) ; }) ;
        $('#table4_erase').button().click(function () { table4.erase() ; }) ;

        var table5 = new SimpleTable.constructor (
            'table5' ,
            [
                {   name: 'Number', type: SimpleTable.Types.Number} ,
                {   name: 'Text'}
            ]
        ) ;
        table5.display() ;

        $('#table5_load').button().click(function () {
            table5.erase(SimpleTable.Status.Loading) ;
            $.ajax ({
                type: 'GET' ,
                url:  '../webfwk/ws/table_data.php' ,
                data: {
                    rows: 12 ,
                    cols: table5.cols()} ,
                success: function (result) {
                    if (result.status != 'success') {
                        table5.erase(SimpleTable.Status.error(result.message)) ;
                        return ;
                    }
                    table5.load(result.data) ;
                } ,
                error: function () {
                    table5.erase(SimpleTable.Status.error('service is not available')) ;
                } ,
                dataType: 'json'
            }) ;
        }) ;

        var table6 = new SimpleTable.constructor (
            'table6' ,
            [
                {   name: '1'} ,
                {   name: '2'} ,
                {   name: 'hideable', hideable: true} ,
                {   name: '4',                                       align: 'center'} ,
                {   name: '5',        hideable: true, sorted: false, align: 'right'}
            ] ,
            [
                ['1(1)','2(2)','3(3)',     4,    5] ,
                ['2(1)','3(2)','4(3)', 12554,  333] ,
                ['3(1)','4(2)','1(3)',     1,   23] ,
                ['4(1)','1(2)','2(3)',    21,    0] ,
                ['7(1)','8(2)','9(3)',    56, 1999]
            ]
        ) ;
        table6.display() ;

        var table7 = new SimpleTable.constructor (
            'table7' ,
            [
                {   name: '1'} ,
                {   name: 'custom style', style:   'color: red; font-size: 125%'} ,
                {   name: 'last',         hideable: true, sorted: false, align: 'right'}
            ] ,
            [
                ['1(1)',     4,    5] ,
                ['2(1)', 12554,  333] ,
                ['3(1)',     1,   23] ,
                ['4(1)',    21,    0] ,
                ['7(1)',    56, 1999]
            ]
        ) ;
        table7.display() ;
    }) ;
}) ;