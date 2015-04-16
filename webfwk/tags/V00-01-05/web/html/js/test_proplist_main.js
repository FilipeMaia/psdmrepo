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
    'webfwk/CSSLoader', 'webfwk/RadioBox', 'webfwk/PropList' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (
    cssloader, RadioBox, PropList) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    $(function () {

        var proplist = new PropList ([{
            name:       "first",    text:   "First" ,
            edit_mode:  true,       editor: "text" ,
            value:      "",         style:  "color:red;" ,
            title:      "this is my first row with the inline suggestion"
        } , {
            name:       "second",   text:   "Second" ,
            value:      "The default value"
        } , {
            name:       "third",    text:   "Something else"
        } , {
            name:       "last",     text:   "Finally" ,
            edit_mode:  true,       editor: "checkbox" ,
            value:      "1"
        }]);
        proplist.display($('#proplist')) ;

        var rb = new RadioBox (
            [   {name: "edit_first", text: "EDIT FIRST", title: "Press to edit a value of the first property" } ,
                {name: "edit_last",  text: "EDIT LAST",  title: "Press to edit a value of the last property" } ,
                {name: "view",       text: "VIEW MODE",  title: "Press to turn all proeprty editors back into the viewing mode"}
            ] ,

            function (name) {
                switch (name) {
                    case 'edit_first':
                        proplist.edit_value('first') ;
                        break ;
                    case 'edit_last':
                        proplist.edit_value('last') ;
                        break ;
                    case 'view':
                        proplist.view_value('first') ;
                        proplist.view_value('last') ;
                        break ;
                }
            } ,
            {   activate: "view"}
        ) ;
        rb.display($('#radiobox')) ;


    }) ;
}) ;