require.config ({
    baseUrl: '..' ,
    paths: {
        'jquery'     : '/jquery/js/jquery-1.8.2' ,
        'jquery-ui'  : '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'underscore' : '/underscore/underscore-min' ,
        'webfwk'     : 'webfwk/js'
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
    'webfwk/CSSLoader', 'webfwk/CheckTable' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (
    cssloader, CheckTable) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    var next_id = 0 ;

    function make_row (role) {
        var id = next_id++ ;
        var row = {
            notify:  id % 3 ? true : false ,
            user:    'user_'+id ,
            role:    role ? role : (Math.floor(id / 4) ? 'Member of instrument support group ps-amo' : 'Data Administrator') ,
            _locked: (Math.floor(id % 5) ? false : true)
        } ;
        return row ;
    }

    function report_result (msg) { $('#result').html(msg) ;  }

    var checktable = null ;

    $(function() {

        var coldef = [
            {name: 'notify', text: 'Notify'} ,
            {name: 'user',   text: 'User'} ,
            {name: 'role',   text: 'Role'}
        ] ;
        var rows = [] ;
        rows.push(make_row('Experiment PI')) ;
        for (var i = 0; i < 20; i++)
            rows.push(make_row()) ;

        checktable = new CheckTable(coldef, rows) ;
        checktable.display($('#checktable')) ;
    
        $('button').button().click(function () {
            var row = make_row() ;
            switch (this.name) {

                case 'insert_front'     : checktable.insert_front(row) ; break ;
                case 'append'           : checktable.append      (row) ; break ;
                case 'remove_by_role'   :
                    var rows = checktable.remove(function (row) {
                        return row.role === 'Experiment PI' ;
                    }) ;
                    report_result(rows.length+' rows') ;
                    break ;
                case 'remove_unchecked' :
                    var rows = checktable.remove(function (row) {
                        return row.notify ? false : true 
                    }) ;
                    report_result(rows.length+' rows') ;
                    break ;
                case 'remove_all' :
                    var rows = checktable.remove_all() ;
                    report_result(rows.length+' rows') ;
                    break ;

                case 'find_checked' :
                    var rows = checktable.find_checked() ;
                    report_result(rows.length+' rows') ;
                    break ;
                case 'find_not_checked' :
                    var rows = checktable.find_not_checked() ;
                    report_result(rows.length+' rows') ;
                    break ;
                case 'check' :
                    checktable.check_all() ;
                    break ;
                case 'uncheck' :
                    checktable.uncheck_all() ;
                    break ;

                case 'find_locked' :
                    var rows = checktable.find_locked() ;
                    report_result(rows.length+' rows') ;
                    break ;
                case 'find_not_locked' :
                    var rows = checktable.find_not_locked() ;
                    report_result(rows.length+' rows') ;
                    break ;
                case 'lock' :
                    checktable.lock_all() ;
                    break ;
                case 'unlock' :
                    checktable.unlock_all() ;
                    break ;
            }
        }) ;
    }) ;
}) ;