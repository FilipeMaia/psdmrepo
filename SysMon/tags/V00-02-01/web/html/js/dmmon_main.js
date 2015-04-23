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
        'sysmon':               'sysmon/js'
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

    'sysmon/DMMon_Live',      'sysmon/DMMon_History' ,
    'sysmon/DMMon_FS_Usage',  'sysmon/DMMon_FS_Summary' ,
    'sysmon/DMMon_FM_Status', 'sysmon/DMMon_FM_Notify' ,


    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery-ui',     'jquery-ui',   'jquery-ui-timepicker' ,
    'jquery.form',   'jquery.json', 'jquery.printElement' ,
    'jquery.resize', 'underscore',  'highcharts/highcharts', 'highcharts/modules/exporting'] ,

function (
    cssloader, Fwk ,

    DMMon_Live,      DMMon_History ,
    DMMon_FS_Usage,  DMMon_FS_Summary ,
    DMMon_FM_Status, DMMon_FM_Notify) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;
    cssloader.load('/jquery/css/jquery-ui-timepicker-addon.css') ;

    cssloader.load('../sysmon/css/dmmon.css') ;

    $(function() {

        var menus = [] ;
        for (var i in app_config.instruments) {
            var instr_name = app_config.instruments[i] ;
            menus.push ({
                name: instr_name, menu: [{
                    name: 'Live',    application: new DMMon_Live   (instr_name, app_config)}, {
                    name: 'History', application: new DMMon_History(instr_name, app_config)}]
            }) ;
        }
        menus.push ({
            name: 'File Systems', menu: [{
                name: 'Summary', application: new DMMon_FS_Summary(app_config)}, {
                name: 'Usage',   application: new DMMon_FS_Usage  (app_config)}]
        } , {
            name: 'File Migration', menu: [{
                name: 'Status',               application: new DMMon_FM_Status(app_config)}, {
                name: 'E-mail Notifications', application: new DMMon_FM_Notify(app_config)}]
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

