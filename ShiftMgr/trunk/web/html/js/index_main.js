require.config ({

    baseUrl: '..' ,

    waitSeconds : 15,
    urlArgs     : "bust="+new Date().getTime() ,

    paths: {
        underscore: '/underscore/underscore-min' ,
        webfwk:     'webfwk/js' ,
        shiftmgr:   'shiftmgr/js'
    }
}) ;
require ([
    'webfwk/Fwk' ,
    'shiftmgr/Reports',     'shiftmgr/History', 'shiftmgr/Notifications' ,
    'shiftmgr/Reports4all', 'shiftmgr/Access',  'shiftmgr/Rules'] ,

function (
    Fwk ,
    Reports,     History, Notifications ,
    Reports4all, Access,  Rules) {

    $(function() {

        var menus = [] ;

        for (var i in app_config.instruments) {
            var instr_name = app_config.instruments[i] ;
            menus.push ({
                name: instr_name,        menu: [{
                    name: 'Reports',              application: new Reports(instr_name, app_config.instr2editor[instr_name])}, {
                    name: 'History',              application: new History(instr_name)}, {
                    name: 'E-mail Notifications', application: new Notifications(instr_name)}]}) ;
        }
        menus.push ({
            name: 'All Hutches',        menu: [{
                name: 'Reports',                  application: new Reports4all()}, {
                name: 'History',                  application: new History()}]}, {
            name: 'Admin',              menu: [{
                name: 'Access Control',           application: new Access()}, {
                name: 'Rules',                    application: new Rules()}]}) ;

        Fwk.build (

            app_config.title ,
            app_config.subtitle ,

            menus ,

            function (text2search) { Fwk.report_info('Search', text2search) ; }
        ) ;
    });
}) ;