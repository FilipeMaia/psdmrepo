require.config ({
    baseUrl: '..' ,
    paths: {
        underscore: '/underscore/underscore-min' ,
        webfwk: 'webfwk/js' ,
        portal: 'portal/js'
    }
}) ;
require ([
    'webfwk/Fwk' ,
    'portal/ExpSwitch_Station' , 'portal/ExpSwitch_History' , 'portal/ExpSwitch_ELogAccess'
] ,

function (
    Fwk ,
    ExpSwitch_Station , ExpSwitch_History , ExpSwitch_ELogAccess) {

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
}) ;

