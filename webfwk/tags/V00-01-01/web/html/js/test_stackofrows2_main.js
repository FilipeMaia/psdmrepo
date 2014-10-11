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
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget', 'webfwk/StackOfRows' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (
    cssloader, Class, Widget, StackOfRows) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    $(function () {

        /* Very simple stack of static text fields with no layers */

        var stack1 = new StackOfRows.StackOfRows(null, null, {
            expand_buttons: true
        }) ;
        for (var i = 0; i < 4; i++)
            stack1.add_row({
                title: '<b>row '+i+'</b>' ,
                body:  'Here be the body of this row'}) ;

        stack1.display($('#stack1')) ;

        /* Nested stack of 2 layers, the bottom one is made of
         * simple stacks of static text fields with no further layers. */

        var stack2 = new StackOfRows.StackOfRows(null, null, {
            expand_buttons: true
        }) ;
        for (var i = 0; i < 4; i++) {

            var stack1 = new StackOfRows.StackOfRows(null, null, {
                expand_buttons: true
            }) ;
            for (var j = 0; j < 3; j++)
                stack1.add_row({
                    title: '<b>row '+i+'.'+j+'</b>' ,
                    body:  'Here be the body of this row'}) ;

            stack2.add_row({
                title: '<b>row '+i+'</b>' ,
                body:  stack1}) ;
        }
        stack2.display($('#stack2')) ;

        /* Nested stack of 3 layers, the bottom one is made of
         * simple stacks of static text fields with no further layers. */

        var stack3 = new StackOfRows.StackOfRows(null, null, {
            expand_buttons:   true ,
            expand_propagate: true
        }) ;
        for (var i = 0; i < 3; i++) {

            var stack2 = new StackOfRows.StackOfRows(null, null, {
                expand_buttons: true ,
                expand_propagate: true
            }) ;
            for (var j = 0; j < 4; j++) {

                var stack1 = new StackOfRows.StackOfRows() ;
                for (var k = 0; k < 3; k++)
                    stack1.add_row({
                        title: '<b>row '+i+'.'+j+'.'+k+'</b>' ,
                        body:  'Here be the body of this row'}) ;

                stack2.add_row({
                    title: '<b>row '+i+'.'+j+'</b>' ,
                    body:  stack1}) ;
            }
            stack3.add_row({
                title: '<b>row '+i+'</b>' ,
                body:  stack2}) ;
        }
        stack3.display($('#stack3')) ;

        /* Simple stack with the header. */

        var stack4 = new StackOfRows.StackOfRows([
            {id: 'id',      title: 'Id',      width: 20} ,
            {id: 'date',    title: 'Date',    width: 80} ,
            {id: 'uid',     title: 'User',    width: 100} ,
            {id: '|' } ,
            {id: 'comment', title: 'Comment', width: 320} ,
            {id: '|' } ,
            {id: 'valid',   title: 'Is Valid?', width: 50}] ,
            null, {
            expand_buttons: false
        }) ;
        for (var i = 0; i < 4; i++)
            stack4.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    date:    '2013-08-10' ,
                    uid:     'Igor Gaponenko' ,
                    comment: 'Here be my message...' ,
                    valid:   i % 2 ? 'Yes' : 'No'
                } ,
                body:  'Here be the body of this row'}) ;

        stack4.display($('#stack4')) ;



        /* Single-level stack with theme customization */

        var stack5 = new StackOfRows.StackOfRows([
            {id: 'id',      title: 'Id',      width: 20} ,
            {id: '|' } ,
            {id: 'comment', title: 'Comment', width: 320}
        ], null, {
            expand_buttons: true ,
            theme: 'stack-theme-large16'
        }) ;
        for (var i = 0; i < 4; i++)
            stack5.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    comment: 'Here be my comment...'
                } ,
                body:  'Here be the body of this row'}) ;

        stack5.display($('#stack5')) ;





        /* Nested stack of 3 layers, the bottom one is made of
         * simple stacks of static text fields with no further layers. */

        var stack6 = new StackOfRows.StackOfRows(null, null, {
            expand_buttons:   true ,
        }) ;
        for (var i = 0; i < 5; i++) {

            var stack2 = new StackOfRows.StackOfRows(null, null, {
                expand_buttons: true ,
                theme: 'stack-theme-aliceblue'
            }) ;
            for (var j = 0; j < 4; j++) {

                var stack1 = new StackOfRows.StackOfRows() ;
                for (var k = 0; k < 3; k++)
                    stack1.add_row({
                        title: '<b>row '+i+'.'+j+'.'+k+'</b>' ,
                        body:  'Here be the body of this row'}) ;

                stack2.add_row({
                    title: '<b>row '+i+'.'+j+'</b>' ,
                    body:  stack1}) ;
            }
            stack6.add_row({
                title: '<b>row '+i+'</b>' ,
                body:  stack2}) ;
        }
        stack6.display($('#stack6')) ;


        /* Both size and color themes applied */

        var stack7 = new StackOfRows.StackOfRows([
            {id: 'id',      title: 'Id',      width: 20} ,
            {id: '|' } ,
            {id: 'comment', title: 'Comment', width: 320}
        ], null, {
            expand_buttons: true ,
            theme: 'stack-theme-large16 stack-theme-aliceblue'
        }) ;
        for (var i = 0; i < 4; i++)
            stack7.add_row({
                title: {
                    id : '<b>'+i+'</b>' ,
                    comment: 'Here be my message...'
                } ,
                body:  'Here be the body of this row'
            }) ;

        stack7.display($('#stack7')) ;

        /* Implementing non-trivial row content by subclassing the Widget */

        function CustomBody(text) {
            this.text = text ;
        }
        Class.define_class(CustomBody, Widget.Widget, {}, {
        render : function () {
            this.container.html('<div>'+this.text+'</div>') ;
        }
        });
        var stack8 = new StackOfRows.StackOfRows([
            {id: 'id',      title: 'Id',      width: 20} ,
            {id: 'date',    title: 'Date',    width: 80} ,
            {id: 'uid',     title: 'User',    width: 100} ,
            {id: '|' } ,
            {id: 'comment', title: 'Comment', width: 320} ,
            {id: '|' } ,
            {id: 'valid',   title: 'Is Valid?', width: 50}] ,
            null, {
            expand_buttons: false ,
            theme: 'stack-theme-aliceblue' ,
            hidden_header: true
        }) ;
        for (var i = 0; i < 4; i++)
            stack8.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    date:    '2013-08-10' ,
                    uid:     'Igor Gaponenko' ,
                    comment: 'Here be my message...' ,
                    valid:   i % 2 ? 'Yes' : 'No'
                } ,
                body:  new CustomBody('Here be the body of this row')
            }) ;

        stack8.display($('#stack8')) ;


        /* Color theme tests */

        var hdr = [
            {id: 'id',      title: 'Id',      width: 20} ,
            {id: '|' } ,
            {id: 'comment', title: 'Comment', width: 320}
        ] ;
        var stack9 = new StackOfRows.StackOfRows(hdr, null, {
            expand_buttons: true ,
            theme: 'stack-theme-mustard'
        }) ;
        for (var i = 0; i < 4; i++)
            stack9.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    comment: 'Here be my comment...'
                } ,
                body:  'Here be the body of this row'}) ;
        stack9.display($('#stack9')) ;

        var stack10 = new StackOfRows.StackOfRows(hdr, null, {
            expand_buttons: true ,
            theme: 'stack-theme-green'
        }) ;
        for (var i = 0; i < 4; i++)
            stack10.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    comment: 'Here be my comment...'
                } ,
                body:  'Here be the body of this row'}) ;
        stack10.display($('#stack10')) ;

        var stack11 = new StackOfRows.StackOfRows(hdr, null, {
            expand_buttons: true ,
            theme: 'stack-theme-brown'
        }) ;
        for (var i = 0; i < 4; i++)
            stack11.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    comment: 'Here be my comment...'
                } ,
                body:  'Here be the body of this row'}) ;
        stack11.display($('#stack11')) ;

        /* Applying different color themes to different rows of the same table */

        var hdr = [
            {id: 'id',      title: 'Id',      width: 20} ,
            {id: 'comment', title: 'Comment', width: 320}
        ] ;
        var stack12 = new StackOfRows.StackOfRows(hdr, null, {
            expand_buttons: true
        }) ;
        for (var i = 0; i < 20; i++)
            stack12.add_row({
                title: {
                    id:      '<b>'+i+'</b>' ,
                    comment: 'Here be my comment...'
                } ,
                body:        'Here be the body of this row' ,
                color_theme: !(i % 4) ? 'stack-theme-green' : null}) ;
        stack12.display($('#stack12')) ;

    }) ;
}) ;
