require.config ({

    baseUrl: '..' ,

    waitSeconds : 15,
    urlArgs     : "bust="+new Date().getTime() ,

    paths: {
        'jquery'                : '/jquery/js/jquery-1.8.2' ,
        'jquery-ui'             : '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'jquery-ui-timepicker'  : '/jquery/js/jquery-ui-timepicker-addon' ,
        'jquery.form'           : '/jquery/js/jquery.form' ,
        'jquery.json'           : '/jquery/js/jquery.json' ,
        'jquery.printElement'   : '/jquery/js/jquery.printElement' ,
        'jquery.resize'         : '/jquery/js/jquery.resize' ,
        'underscore'            : '/underscore/underscore-min' ,
        'webfwk'                : 'webfwk/js' ,
        'portal'                : 'portal/js' ,
        'shiftmgr'              : 'shiftmgr/js'
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
    'webfwk/CSSLoader', 'webfwk/Fwk' ,

    'portal/Experiment_Info',        'portal/Experiment_Group',    'portal/Experiment_ELogAccess' ,
    'portal/ELog_Live',              'portal/ELog_Post',           'portal/ELog_Search' ,
    'portal/ELog_Shifts',            'portal/ELog_Runs',           'portal/ELog_Attachments' ,
    'portal/ELog_Subscribe',
    'portal/Runtables_Calibrations', 'portal/Runtables_DAQ',
    'portal/Runtables_EPICS',        'portal/Runtables_User',
    'portal/Filemanager_Summary',    'portal/Filemanager_Files',   'portal/Filemanager_Files_USR' ,
    'portal/HDF5_Translator' ,

    'shiftmgr/Reports', 'shiftmgr/History', 'shiftmgr/Notifications' ,
   
    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery-ui',    'jquery-ui',   'jquery-ui-timepicker' ,
    'jquery.form',  'jquery.json', 'jquery.printElement' ,
    'jquery.resize'] ,

function (
    cssloader,              Fwk ,
    Experiment_Info,        Experiment_Group,    Experiment_ELogAccess ,
    ELog_Live,              ELog_Post,           ELog_Search ,
    ELog_Shifts,            ELog_Runs,           ELog_Attachments ,
    ELog_Subscribe,
    Runtables_Calibrations, Runtables_DAQ ,
    Runtables_EPICS,        Runtables_User ,
    Filemanager_Summary,    Filemanager_Files,   Filemanager_Files_USR ,
    HDF5_Translator ,
    Reports,                History,             Notifications) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;
    cssloader.load('/jquery/css/jquery-ui-timepicker-addon.css') ;

    cssloader.load('../portal/css/common.css') ;

    $(function() {

        var experiment  = app_config.experiment ;
        var access_list = app_config.access_list ;

        var menus = [{
            name: 'Experiment',        menu: [{
                name: 'Info',                   application: new Experiment_Info       (experiment, access_list)}, {
                name: 'Group Manager',          application: new Experiment_Group      (experiment, access_list)}, {
                name: 'e-Log Access',           application: new Experiment_ELogAccess (experiment, access_list)}]}] ;

        if (experiment.is_facility) menus.push ({
            name: 'e-Log',              menu: [{
                name: 'Recent (Live)',          application: new ELog_Live        (experiment, access_list)}, {
                name: 'Post',                   application: new ELog_Post        (experiment, access_list)}, {
                name: 'Search',                 application: new ELog_Search      (experiment, access_list)}, {
                name: 'Attachments',            application: new ELog_Attachments (experiment, access_list)}, {
                name: 'Subscribe',              application: new ELog_Subscribe   (experiment, access_list)}]}) ;

        else menus.push ({
            name: 'e-Log',              menu: [{
                name: 'Recent (Live)',          application: new ELog_Live        (experiment, access_list)}, {
                name: 'Post',                   application: new ELog_Post        (experiment, access_list)}, {
                name: 'Search',                 application: new ELog_Search      (experiment, access_list)}, {
                name: 'Shifts',                 application: new ELog_Shifts      (experiment, access_list)}, {
                name: 'Runs',                   application: new ELog_Runs        (experiment, access_list)}, {
                name: 'Attachments',            application: new ELog_Attachments (experiment, access_list)}, {
                name: 'Subscribe',              application: new ELog_Subscribe   (experiment, access_list)}]}, {
            name: 'Run Tables',         menu: [{
                name: 'Calibrations',           application: new Runtables_Calibrations (experiment, access_list)}, {
                name: 'DAQ',                    application: new Runtables_DAQ          (experiment, access_list)}, {
                name: 'EPICS',                  application: new Runtables_EPICS        (experiment, access_list)}, {
                name: 'User',                   application: new Runtables_User         (experiment, access_list)}]}, {
            name: 'File Manager',       menu: [{
                name: 'Summary',                application: new Filemanager_Summary   (experiment, access_list)}, {
                name: 'XTC HDF5',               application: new Filemanager_Files     (experiment, access_list)}, {
                name: 'USR',                    application: new Filemanager_Files_USR (experiment, access_list)}]}, {
            name: 'HDF5 Translation',   menu: [{
                name: 'Standard',               application: new HDF5_Translator('Standard',   experiment, access_list)}, {
                name: 'Monitoring',             application: new HDF5_Translator('Monitoring', experiment, access_list)}]}) ;
        
        if (access_list.shiftmgr.can_edit) menus.push ({
            name: 'Hutch Manager',      menu: [{
                name: 'Reports',                application: new Reports       (experiment.instrument.name, access_list.shiftmgr.can_edit)}, {
                name: 'History',                application: new History       (experiment.instrument.name)}, {
                name: 'E-mail Notifications',   application: new Notifications (experiment.instrument.name)}]}) ;

        Fwk.build (
            app_config.title ,
            app_config.subtitle_url ,
            menus ,
            function (text2search) { global_elog_search_message_by_text (text2search) ; } ,
            function () {
                Fwk.activate (
                    app_config.select_app ,
                    app_config.select_app_context1) ; }) ;
    }) ;
    
    // Redirections in the global scope which may be required by the legacy
    // code generatet by Web services.

    window.show_email = function (user, addr) {
        Fwk.show_email(user, addr) ; } ;

    window.global_elog_search_message_by_text = function (text2search) {
        var application = Fwk.activate('e-Log', 'Search') ;
        if (application) application.search_message_by_text(text2search) ;
        else console.log('global_elog_search_message_by_text(): not implemented') ; } ;

    window.global_elog_search_message_by_id = function (id, show_in_vicinity) {
        var application = Fwk.activate('e-Log', 'Search') ;
        if (application) application.search_message_by_id(id, show_in_vicinity) ;
        else console.log('global_elog_search_message_by_id(): not implemented') ; } ;

    window.global_elog_search_run_by_num = function (num, show_in_vicinity) {
        var application = Fwk.activate('e-Log', 'Search') ;
        if (application) application.search_run_by_num (num, show_in_vicinity) ;
        else console.log('global_elog_search_run_by_num(): not implemented') ; } ;

    window.display_path = function (filepath) { Fwk.show_path(filepath) ; } ;
}) ;

