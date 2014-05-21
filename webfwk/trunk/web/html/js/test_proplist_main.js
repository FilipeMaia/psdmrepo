require.config ({
    baseUrl: '..' ,
    paths: {
        underscore: '/underscore/underscore-min' ,
        webfwk: 'webfwk/js'
    }
}) ;

require ([
    'webfwk/PropList'
] ,

function (PropList) {

    $(function () {
        var proplist = new PropList (
            [   {name: "first",  text: "First",   value: "" } ,
                {name: "second", text: "Second",  value: "The default value"} ,
                {name: "third",  text: "Something else"} ,
                {name: "next",   text: "Finally", style: "color:red;"}
            ]
        ) ;
        proplist.display($('#proplist')) ;
    }) ;
}) ;