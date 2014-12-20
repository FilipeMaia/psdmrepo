require.config ({

    baseUrl : '..' ,

    paths : {
        'jquery'                : '/jquery/js/jquery-1.8.2' ,
        'jquery-ui'             : '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'jquery-ui-timepicker'  : '/jquery/js/jquery-ui-timepicker-addon' ,
        'jquery.form'           : '/jquery/js/jquery.form' ,
        'jquery.json'           : '/jquery/js/jquery.json' ,
        'jquery.printElement'   : '/jquery/js/jquery.printElement' ,
        'jquery.resize'         : '/jquery/js/jquery.resize' ,
        'underscore'            : '/underscore/underscore-min' ,
        'webfwk'                : 'webfwk/js' ,
        'portal'                : 'portal/js'
    } ,

    shim : {
        'jquery-ui' : {
            exports : '$' ,
            deps : ['jquery']
        } ,
        'jquery-ui-timepicker' :  {
            deps : ['jquery-ui']
        } ,
        'jquery.form' :  {
            deps : ['jquery-ui']
        } ,
        'jquery.json' :  {
            deps : ['jquery']
        } ,
        'jquery.printElement' :  {
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
    'webfwk/CSSLoader',          'webfwk/Fwk' ,
    'portal/ExpSwitch_Station' , 'portal/ExpSwitch_History' , 'portal/ExpSwitch_ELogAccess' ,
    
    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery-ui',    'jquery-ui',   'jquery-ui-timepicker' ,
    'jquery.form',  'jquery.json', 'jquery.printElement' ,
    'jquery.resize'] ,

function (
    cssloader,          Fwk ,
    ExpSwitch_Station , ExpSwitch_History , ExpSwitch_ELogAccess) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;
    cssloader.load('/jquery/css/jquery-ui-timepicker-addon.css') ;

    cssloader.load('../portal/css/common.css') ;

    $(function() {

        var instruments = app_config.instruments ;

        var menus = [] ;
        for (var i in instruments) {

            var instr = instruments[i] ;
            var instr_tab = {
                name: instr.name ,
                menu: []} ;

            for (var station = 0; station < instr.num_stations; ++station)
                instr_tab.menu.push ({
                    name: 'Station '+station ,
                    application: new ExpSwitch_Station(instr.name, station, instr.access_list)}) ;

                instr_tab.menu.push ({
                    name: 'History' ,
                    application: new ExpSwitch_History(instr.name, instr.access_list)
                } , {
                    name: 'e-Log Access - '+instr.operator_uid ,
                    application: new ExpSwitch_ELogAccess(instr.name, instr.operator_uid, instr.access_list)
                }) ;

            menus.push(instr_tab) ;
        }
        Fwk.build (

            app_config.title ,
            app_config.subtitle ,

            menus ,

            null ,  // no quick search for this application

            function () {
                Fwk.activate(app_config.select_app, app_config.select_app_context1) ; }
        ) ;
    }) ;

    // Redirections which may be required by the legacy code generated
    // by Web services.

    window.show_email = function (user, addr) { Fwk.show_email(user, addr) ; } ;

}) ;

