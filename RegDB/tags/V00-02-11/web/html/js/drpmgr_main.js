require.config ({

    baseUrl: '..' ,

    waitSeconds: 15,
    urlArgs:     "bust="+new Date().getTime() ,

    paths: {
        'jquery':               '/jquery/js/jquery-1.8.2' ,
        'jquery-ui':            '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'jquery-ui-timepicker': '/jquery/js/jquery-ui-timepicker-addon' ,
        'jquery.form':          '/jquery/js/jquery.form' ,
        'jquery.json':          '/jquery/js/jquery.json' ,
        'jquery.printElement':  '/jquery/js/jquery.printElement' ,
        'jquery.resize':        '/jquery/js/jquery.resize' ,
        'underscore':           '/underscore/underscore-min' ,
        'highcharts':           '/highcharts/js' ,
        'webfwk':               'webfwk/js' ,
        'regdb':                'regdb/js'
    } ,

    shim:  {
        'jquery-ui': {
            exports: '$' ,
            deps: ['jquery']
        } ,
        'jquery-ui-timepicker':  {
            deps : ['jquery-ui']
        } ,
        'jquery.form':  {
            deps: ['jquery-ui']
        } ,
        'jquery.json':  {
            deps: ['jquery']
        } ,
        'jquery.printElement':  {
            deps: ['jquery']
        } ,
        'jquery.resize':  {
            deps: ['jquery']
        } ,
        'underscore': {
            exports: '_'
        } ,
        'highcharts/highcharts': {
            deps:    ['jquery', 'jquery-ui'] ,
            exports: 'Highcharts'
        } ,
        'highcharts/modules/exporting': {
            deps: ['highcharts/highcharts']
        }
    }
}) ;
require ([
    'webfwk/CSSLoader', 'webfwk/Fwk' ,

    'regdb/DRPMgr_Policy',     'regdb/DRPMgr_Exceptions' ,
    'regdb/DRPMgr_Instrument', 'regdb/DRPMgr_Totals' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery-ui',     'jquery-ui',   'jquery-ui-timepicker' ,
    'jquery.form',   'jquery.json', 'jquery.printElement' ,
    'jquery.resize', 'underscore',  'highcharts/highcharts', 'highcharts/modules/exporting'] ,

function (
    cssloader, Fwk ,

    DRPMgr_Policy,     DRPMgr_Exceptions ,
    DRPMgr_Instrument, DRPMgr_Totals) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;
    cssloader.load('/jquery/css/jquery-ui-timepicker-addon.css') ;

    cssloader.load('../regdb/css/drpmgr.css') ;

    $(function() {

        var menus = [] ;
        menus.push ({
            name: 'Policy', menu: [{
                name: 'General',    application: new DRPMgr_Policy    (app_config)} , {
                name: 'Exceptions', application: new DRPMgr_Exceptions(app_config)}]
        }) ;
        for (var i in app_config.instruments) {
            var instr_name = app_config.instruments[i] ;
            menus.push ({
                name: instr_name, menu: [{
                    name: 'Manage', application: new DRPMgr_Instrument(instr_name, app_config)}]
            }) ;
        }
        menus.push ({
            name: 'Totals', menu: [{
                name: 'Storage Usage', application: new DRPMgr_Totals(app_config)}]
        }) ;

        Fwk.build (

            app_config.title ,
            app_config.subtitle ,

            menus ,

            function (text2search) {
                console.log('main() -- no quick search available for this application')
            } ,
            function () {
                
                // Display the staring application

                Fwk.activate(app_config.select_app, app_config.select_app_context1) ;
            }
        ) ;
    }) ;

    // Global entries for applications

    window.global_variable_test = 0 ;

}) ;

