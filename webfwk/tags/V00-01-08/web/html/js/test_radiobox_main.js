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
    'webfwk/CSSLoader', 'webfwk/RadioBox' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (cssloader, RadioBox) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    $(function () {
        var rb = new RadioBox (
              [   {name: "first",  text: "First",   title: "Press the First button" } ,
                  {name: "second", text: "Second",  title: "Press the Second button"} ,
                  {name: "third",  text: "Press me"} ,
                  {name: "next",   text: "Delete",  style: "color:red;"}
              ] ,

              function (name) {
                  console.log('pressed button: '+name)
              } ,
              {   activate: "second"}
          ) ;
          rb.display($('#radiobox')) ;
    }) ;
}) ;


