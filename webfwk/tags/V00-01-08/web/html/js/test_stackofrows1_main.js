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
    'webfwk/CSSLoader', 'webfwk/StackOfRows' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (
    cssloader, StackOfRows) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    $(function () {

        var stack1 = new StackOfRows.StackOfRows(null, null, {
            expand_buttons: false ,
            theme: 'stack-theme-aliceblue' ,
            hidden_header: false ,
            effect_on_insert: function (hdr_cont) {
                hdr_cont.stop(true,true).effect('highlight', {color:'#ff6666'}, 30000) ;
            }
        }) ;
        for (var i = 0; i < 100; i++)
            stack1.add_row({
                title: '<b>row '+i+'</b>' ,
                body:  'Here be the body of this row'}) ;

        stack1.display($('#stack1')) ;

        var ctrl = $('#ctrl') ;

        var row_cont  = ctrl.find('input[name="row"]') ;
        var body_cont = ctrl.find('input[name="body"]') ;

        ctrl.find('button[name="update"]').button().click(function () {
            var row = parseInt(row_cont.val()) ;
            var body = body_cont.val() ;
            stack1.update_row(row , {
                title: '<b>row '+row+'</b> (updated)' ,
                body:  body
            }) ;
        }) ;
        function expand_or_collapse(expand, focus) {
            var row = parseInt(row_cont.val()) ;
            stack1.expand_or_collapse_row(row, expand, focus ? $('body') : null) ;
        }
        ctrl.find('button[name="expand"]'  ).button().click(function () { expand_or_collapse(true) ; }) ;
        ctrl.find('button[name="collapse"]').button().click(function () { expand_or_collapse(false) ; }) ;

        ctrl.find('button[name="test"]').button().click(function () {
            var row = parseInt(row_cont.val()) ;
            alert(stack1.is_expanded(row) ? 'expanded' : 'collapsed') ;
        }) ;
        ctrl.find('button[name="front"]').button().click(function () {
            var body = body_cont.val() ;
            stack1.insert_front({
                title: '<b>new row' ,
                body:  body
            }) ;
        }) ;
        ctrl.find('button[name="back"]').button().click(function () {
            var body = body_cont.val() ;
            stack1.append({
                title: '<b>new row' ,
                body:  body
            }) ;
        }) ;
        ctrl.find('button[name="delete"]').button().click(function () {
            var row = parseInt(row_cont.val()) ;
            stack1.delete_row(row) ;
        }) ;
        ctrl.find('button[name="focus"]').button().click(function () {
            expand_or_collapse(true, true) ;
        }) ;
        $(document).bind('scroll', function () {
            console.log('scroll detected') ;
            if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight) {
                console.log("you're at the bottom of the page") ;
                stack1.append({
                    title: '<b>new row' ,
                    body:  'row body is here'
                }) ;
            }
        });
    }) ;
}) ;


