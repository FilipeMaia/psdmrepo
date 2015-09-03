require.config ({

    baseUrl : '..' ,

    waitSeconds : 15,
    urlArgs     : "bust="+new Date().getTime() ,

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
        'irep'                  : 'irep/js'
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

    'irep/Equipment_Inventory',  'irep/Equipment_Add' , 
    'irep/Issue_Search',         'irep/Issue_Report' , 
    'irep/Dictionary_Equipment', 'irep/Dictionary_Location', 'irep/Dictionary_Status' , 
    'irep/Admin_Access',         'irep/Admin_Notify',        'irep/Admin_SLACid' , 

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery-ui',     'jquery-ui',   'jquery-ui-timepicker' ,
    'jquery.form',   'jquery.json', 'jquery.printElement' ,
    'jquery.resize', 'underscore'] ,

function (
    cssloader, Fwk ,

    Equipment_Inventory,  Equipment_Add , 
    Issue_Search,         Issue_Report , 
    Dictionary_Equipment, Dictionary_Location, Dictionary_Status , 
    Admin_Access,         Admin_Notify,        Admin_SLACid) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;
    cssloader.load('/jquery/css/jquery-ui-timepicker-addon.css') ;

    cssloader.load('../irep/css/irep.css') ;

    $(function() {

        var menus = [{
            name: 'Equipment',      menu: [{
                name: 'Inventory',              application: new Equipment_Inventory  (app_config)} , {
                name: 'Add New Equipment',      application: new Equipment_Add        (app_config)}]} , {
            name: 'Issues',         menu: [{
                name: 'Search',                 application: new Issue_Search         (app_config)} , {
                name: 'Reports',                application: new Issue_Report         (app_config)}]} , {
            name: 'Dictionary',     menu: [{
                name: 'Equipment',              application: new Dictionary_Equipment (app_config)} , {
                name: 'Locations',              application: new Dictionary_Location  (app_config)} , {
                name: 'Statuses',               application: new Dictionary_Status    (app_config)}]} , {
            name: 'Admin',          menu: [{
                name: 'Access Control',         application: new Admin_Access         (app_config)} , {
                name: 'E-mail Notifications',   application: new Admin_Notify         (app_config)} , {
                name: 'SLACid Numbers',         application: new Admin_SLACid         (app_config)} ]}] ;

        Fwk.build (

            app_config.title ,
            app_config.subtitle ,

            menus ,

            function (text2search) {
                Fwk.activate('Equipment', 'Inventory').quick_search(text2search) ;
            } ,
            function () {
                
                // Display the staring application

                Fwk.activate(app_config.select_app, app_config.select_app_context1) ;
        
                // preload dictionaries as they may be needed by some applications

                Fwk.get_application('Dictionary', 'Equipment').init() ;
                Fwk.get_application('Dictionary', 'Locations').init() ;
                Fwk.get_application('Dictionary', 'Statuses') .init() ;
            }
        ) ;
    }) ;

    // Global entries for applications

    window.global_equipment_inventory = function () { return Fwk.get_application('Equipment', 'Inventory') ; }

    // Redirections which may be required by the legacy code generated
    // by Web services.

    window.global_get_editors = function () {
        var editors = Fwk.get_application('Admin', 'Access Control').editors() ;
        if (editors) return editors ;
        return app_config.editors ;
    } ;
    window.global_simple_search = function ()   {
        alert('global_simple_search') ;
        Fwk.get_application('Equipment', 'Inventory').simple_search($('#p-search-text').val()) ;
    } ;
    window.global_search_equipment_by_id           = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by(id) ; } ;
    window.global_search_equipment_by_location     = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_location(id) ; } ;
    window.global_search_equipment_by_room         = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_room(id) ; } ;
    window.global_search_equipment_by_manufacturer = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_manufacturer(id) ; } ;
    window.global_search_equipment_by_model        = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_model(id) ; } ;
    window.global_search_equipment_by_slacid_range = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_slacid_range(id) ; } ;
    window.global_search_equipment_by_status       = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_status(id) ; } ;
    window.global_search_equipment_by_status2      = function (id) { Fwk.get_application('Equipment', 'Inventory').search_equipment_by_status2(id) ; } ;

    window.global_export_equipment = function (search_params, outformat) {
        search_params.format = outformat ;
        var html = '<img src="../webfwk/img/loading.gif" />' ;
        var dialog = Fwk.report_action('Generating Document: '+outformat,html) ;
        var jqXHR = $.get (
            '../irep/ws/equipment_inventory_search.php', search_params ,
            function(data) {
                if (data.status != 'success') {
                    Fwk.report_error(data.message) ;
                    dialog.dialog('close') ;
                    return ;
                }
                var html = 'Document is ready to be downloaded from this location: <a class="link" href="'+data.url+'" target="_blank" >'+data.name+'</a>' ;
                dialog.html(html) ;
            },
            'JSON'
        ).error(function () {
            Fwk.report_error('failed because of: '+jqXHR.statusText) ;
            dialog.dialog('close') ; }
        ).complete(function () {}) ;
    } ;

    window.global_equipment_status2rank = function (status) {
        switch(status) {
            case 'Unknown': return 0 ;
        }
        return -1 ;
    } ;
    window.global_equipment_sorter_by_status       = function (a,b) { return window.global_equipment_status2rank(a.status) - window.global_equipment_status2rank(b.status) ; } ;
    window.sort_as_text                            = function (a,b) { return a == b ? 0 : (a < b ? -1 : 1) ; } ;
    window.global_equipment_sorter_by_manufacturer = function (a,b) { return sort_as_text(a.manufacturer, b.manufacturer) ; } ;
    window.global_equipment_sorter_by_model        = function (a,b) { return sort_as_text(a.model,        b.model) ; } ;
    window.global_equipment_sorter_by_location     = function (a,b) { return sort_as_text(a.location,     b.location) ; } ;
    window.global_equipment_sorter_by_modified     = function (a,b) { return a.modified.time_64 - b.modified.time_64 ; } ;

    window.global_dict_find_model_by_id = function (id)                  { return Fwk.get_application('Dictionary', 'Equipment').find_model_by_id(id) ; } ;
    window.global_dict_find_model       = function (manufacturer, model) { return Fwk.get_application('Dictionary', 'Equipment').find_model      (manufacturer, model) ; } ;

}) ;

