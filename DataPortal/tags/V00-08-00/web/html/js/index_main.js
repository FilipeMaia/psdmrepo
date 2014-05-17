require.config ({
    baseUrl: '..' ,
    paths: {
        webfwk: 'webfwk/js' ,
        portal: 'portal/js'
    }
}) ;
require ([
    'portal/Experiment_Info' ,
    'portal/Experiment_Group' ,
    'portal/Experiment_ELogAccess' ,

    'portal/ELog_Live' ,
    'portal/ELog_Post' ,
    'portal/ELog_Search' ,
    'portal/ELog_Shifts' ,
    'portal/ELog_Runs' ,
    'portal/ELog_Attachments' ,
    'portal/ELog_Subscribe' ,

    'portal/Runtables_Calibrations' ,
    'portal/Runtables_Detectors' ,
    'portal/Runtables_EPICS' ,
    'portal/Runtables_User' ,

    'portal/Filemanager_Summary' ,
    'portal/Filemanager_Files' ,
    'portal/Filemanager_Files_USR' ,

    'portal/HDF5_Manage'
] ,

function (
    Experiment_Info ,
    Experiment_Group ,
    Experiment_ELogAccess ,

    ELog_Live ,
    ELog_Post ,
    ELog_Search ,
    ELog_Shifts ,
    ELog_Runs ,
    ELog_Attachments ,
    ELog_Subscribe ,

    Runtables_Calibrations ,
    Runtables_Detectors ,
    Runtables_EPICS ,
    Runtables_User ,

    Filemanager_Summary ,
    Filemanager_Files ,
    Filemanager_Files_USR ,

    HDF5_Manage
) {

    $(function() {

        var menus = [{
            name: 'Experiment',  menu: [{
                name: 'Info',                   application: new Experiment_Info       (experiment, access_list)}, {
                name: 'Group Manager',          application: new Experiment_Group      (experiment, access_list)}, {
                name: 'e-Log Access',           application: new Experiment_ELogAccess (experiment, access_list)}]}] ;

        if (experiment.is_facility) menus.push ({
            name: 'e-Log',      menu: [{
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
                name: 'DAQ Detectors',          application: new Runtables_Detectors    (experiment, access_list)}, {
                name: 'EPICS',                  application: new Runtables_EPICS        (experiment, access_list)}, {
                name: 'User',                   application: new Runtables_User         (experiment, access_list)}]}, {
            name: 'File Manager',       menu: [{
                name: 'Summary',                application: new Filemanager_Summary   (experiment, access_list)}, {
                name: 'XTC HDF5',               application: new Filemanager_Files     (experiment, access_list)}, {
                name: 'USR',                    application: new Filemanager_Files_USR (experiment, access_list)}]}, {
            name: 'HDF5 Translation',   menu: [{
                name: 'Manage',                 application: new HDF5_Manage(experiment, access_list)}]}) ;
        
        if (access_list.shiftmgr.can_edit) menus.push ({
            name: 'Hutch Manager',      menu: [{
                name: 'Reports',                application: new Reports       (experiment.instrument.name, access_list.shiftmgr.can_edit), html_container: 'shift-reports-'+experiment.instrument.name}, {
                name: 'History',                application: new History       (experiment.instrument.name),                                html_container: 'shift-history-'+experiment.instrument.name}, {
                name: 'E-mail Notifications',   application: new Notifications (experiment.instrument.namet),                               html_container: 'shift-notifications-'+experiment.instrument.name}]}) ;

        Fwk.build (
            title, subtitle_url,
            menus,
            function (text2search) { global_elog_search_message_by_text (text2search) ; } ,
            function ()            { Fwk.activate(select_app, select_app_context1) ; }) ;
    }) ;
}) ;

