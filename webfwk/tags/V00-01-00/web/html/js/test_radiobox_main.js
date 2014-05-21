require.config ({
    baseUrl: '..' ,
    paths: {
        underscore: '/underscore/underscore-min' ,
        webfwk: 'webfwk/js'
    }
}) ;

require ([
    'webfwk/RadioBox'
] ,

function (RadioBox) {

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


