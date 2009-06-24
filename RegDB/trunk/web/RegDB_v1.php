<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <title>Experiment Registry Database</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">

        <!-- Standard reset, fonts and grids -->
        <link rel="stylesheet" type="text/css" href="/yui/build/reset-fonts-grids/reset-fonts-grids.css">

        <!-- CSS for YUI -->
        <link rel="stylesheet" type="text/css" href="/yui/build/menu/assets/skins/sam/menu.css">
        <link rel="stylesheet" type="text/css" href="/yui/build/paginator/assets/skins/sam/paginator.css" />
        <link rel="stylesheet" type="text/css" href="/yui/build/datatable/assets/skins/sam/datatable.css" />
        <link rel="stylesheet" type="text/css" href="/yui/build/button/assets/skins/sam/button.css" />

        <!-- Custom CSS -->
        <link rel="stylesheet" type="text/css" href="RegDBTest.css" />

        <!-- Custom JavaScript -->
        <script type="text/javascript" src="Loader.js"></script>

        <!-- Page-specific styles -->
        <style type="text/css">
            div.yui-b p {
                margin: 0 0 .5em 0;
                color: #999;
            }
            div.yui-b p strong {
                font-weight: bold;
                color: #000;
            }
            div.yui-b p em {
                color: #000;
            }
            h1 {
                font-weight: bold;
                margin: 0 0 1em 0;
                padding: .25em .5em;
                background-color: #ccc;
            }
            #application_title {
                font-family: "Times", serif;
                font-size:42px;
                background-color:#d0d0d0;
                border:solid 2px transparent;
                border-left-width:16px;
            }
            #menubar {
                margin: 0 0 10px 0;
            }
            #context {
                margin-top:20px;
                margin-left:10px;
                margin-right:20px;
                margin-bottom:0px;
                /*font-family: "Serif", "Times New Roman";*/
                /*font-weight: bold;*/
                font-size:16px;
                /*color:#0071bc;*/
                /*background-color:#d0d0d0;*/
                border:solid 4px transparent;
                border-left-width:16px;
                text-align:left;
            }
            #workarea {
                margin-left:40px;
                margin-right:40px;
            }
            #experiment_info,
            #instrument_info {
                margin-top:0px;
                margin-left:4px;
            }
            #workarea_table_container table,
            #params_table_container   table {
                /*margin-left:auto;*/
                /*margin-right:auto;*/
            }
            #workarea_table_paginator,
            #params_table_page,
            #instrument_params_table_page {
                margin-left:auto;
                margin-right:auto;
            }
            #workarea_table_container,
            #workarea_table_container .yui-dt-loading,
            #params_table_container,
            #params_table_container .yui-dt-loading {
                text-align:center;
                background-color:transparent;
            }
            #params_actions_container,
            #actions_container {
                margin-top:24px;
                margin-left:0px;
                text-align:left;
            }

        </style>

        <!-- Dependency source files -->
        <script type="text/javascript" src="/yui/build/yahoo-dom-event/yahoo-dom-event.js"></script>
        <script type="text/javascript" src="/yui/build/animation/animation.js"></script>
        <script type="text/javascript" src="/yui/build/container/container_core.js"></script>

        <!-- Menu source file -->
        <script type="text/javascript" src="/yui/build/menu/menu.js"></script>
        <script type="text/javascript" src="/yui/build/connection/connection-min.js"></script>
        <script type="text/javascript" src="/yui/build/json/json-min.js"></script>
        <script type="text/javascript" src="/yui/build/element/element-min.js"></script>
        <script type="text/javascript" src="/yui/build/paginator/paginator-min.js"></script>
        <script type="text/javascript" src="/yui/build/datasource/datasource-min.js"></script>
        <script type="text/javascript" src="/yui/build/datatable/datatable-min.js"></script>
        <script type="text/javascript" src="/yui/build/button/button-min.js"></script>

        <script type="text/javascript" src="/yui/build/yahoo/yahoo-min.js"></script>
        <script type="text/javascript" src="/yui/build/dom/dom-min.js"></script>


<!-- Page-specific script -->
<script type="text/javascript">

YAHOO.util.Event.onContentReady("menubar", function () {

    var oMenuBar =
        new YAHOO.widget.MenuBar("menubar", {
            autosubmenudisplay: true,
            hidedelay: 750,
            lazyload: true });

    var aSubmenuData = [
        {   id: "applications",
            itemdata: [
                { text: "Experiment Registry Database", url: "javascript:leave_current_app()" },
                { text: "Electronic Log Book", url: "javascript:leave_current_app()" } ] },
        { },
        {   id: "experiments",
            itemdata: [
                { text: "Select..", url: "javascript:list_experiments()" },
                { text: "Create New..", url: "javascript:create_experiment()" } ] },
        {   id: "instruments",
            itemdata: [
                { text: "Select..", url: "javascript:list_instruments()" },
                { text: "Create New..", url: "javascript:create_instrument()" } ] },
        { },
        { },
        {   id: "help",
            itemdata: [
                { text: "Help contents...", url: "#" },
                { text: "Help with the current page...", url: "#" },
                { text: "About the application", url: "#" } ] } ];

    var ua = YAHOO.env.ua,
        oAnim;  // Animation instance

    function onSubmenuBeforeShow(p_sType, p_sArgs) {

        var oBody,
            oElement,
            oShadow,
            oUL;

        if (this.parent) {

            oElement = this.element;

            oShadow = oElement.lastChild;
            oShadow.style.height = "0px";

            if (oAnim && oAnim.isAnimated()) {
                oAnim.stop();
                oAnim = null;
            }
            oBody = this.body;

            //  Check if the menu is a submenu of a submenu.
            if (this.parent &&
                !(this.parent instanceof YAHOO.widget.MenuBarItem)) {

                if (ua.gecko || ua.opera) {
                    oBody.style.width = oBody.clientWidth + "px";
                }
                if (ua.ie == 7) {
                    oElement.style.width = oElement.clientWidth + "px";
                }
            }
            oBody.style.overflow = "hidden";

            oUL = oBody.getElementsByTagName("ul")[0];
            oUL.style.marginTop = ("-" + oUL.offsetHeight + "px");
        }
    }
    function onTween(p_sType, p_aArgs, p_oShadow) {

        if (this.cfg.getProperty("iframe")) {
            this.syncIframe();
        }
        if (p_oShadow) {
            p_oShadow.style.height = this.element.offsetHeight + "px";
        }
    }
    function onAnimationComplete(p_sType, p_aArgs, p_oShadow) {

        var oBody = this.body,
            oUL = oBody.getElementsByTagName("ul")[0];

        if (p_oShadow) {
            p_oShadow.style.height = this.element.offsetHeight + "px";
        }
        oUL.style.marginTop = "";
        oBody.style.overflow = "";

        //  Check if the menu is a submenu of a submenu.

        if (this.parent &&
            !(this.parent instanceof YAHOO.widget.MenuBarItem)) {

            // Clear widths set by the "beforeshow" event handler

            if (ua.gecko || ua.opera) {
                oBody.style.width = "";
            }
            if (ua.ie == 7) {
                this.element.style.width = "";
            }
        }
    }
    function onSubmenuShow(p_sType, p_sArgs) {

        var oElement,
            oShadow,
            oUL;

        if (this.parent) {

            oElement = this.element;
            oShadow = oElement.lastChild;
            oUL = this.body.getElementsByTagName("ul")[0];

            oAnim = new YAHOO.util.Anim(oUL,
                { marginTop: { to: 0 } },
                .5, YAHOO.util.Easing.easeOut);

            oAnim.onStart.subscribe(function () {
                oShadow.style.height = "100%";
            });
            oAnim.animate();

            if (YAHOO.env.ua.ie) {
                oShadow.style.height = oElement.offsetHeight + "px";
                oAnim.onTween.subscribe(onTween, oShadow, this);
            }
            oAnim.onComplete.subscribe(onAnimationComplete, oShadow, this);
        }
    }
    oMenuBar.subscribe("beforeRender", function () {

        var nSubmenus = aSubmenuData.length,
            i;

        if (this.getRoot() == this) {
            for (i = 0; i < nSubmenus; i++) {
                this.getItem(i).cfg.setProperty("submenu", aSubmenuData[i]);
            }
        }
    });
    oMenuBar.subscribe("beforeShow", onSubmenuBeforeShow);
    oMenuBar.subscribe("show", onSubmenuShow);
    oMenuBar.render();
});

</script>

<?php

/* Initial action dispatcher's generator
 */
echo <<<HERE
<script type="text/javascript">
function init() {
HERE;
if( isset( $_GET['action'] )) {
    $action = trim( $_GET['action'] );
    if( $action == 'list_experiments' ) {
        echo "  list_experiments();";
    } else if( $action == 'view_experiment' ) {
        $id   = $_GET['id'];
        $name = $_GET['name'];
        echo "  view_experiment( {$id}, '{$name}' );";
    } else if( $action == 'edit_experiment' ) {
        $id   = $_GET['id'];
        $name = $_GET['name'];
        echo "  edit_experiment( {$id}, '{$name}' );";
    } else if( $action == 'create_experiment' ) {
        echo "  create_experiment();";
    } else if( $action == 'list_instruments' ) {
        echo "  list_instruments();";
    } else if( $action == 'view_instrument' ) {
        $id   = $_GET['id'];
        $name = $_GET['name'];
        echo "  view_instrument( {$id}, '{$name}' );";
    } else if( $action == 'edit_instrument' ) {
        $id   = $_GET['id'];
        $name = $_GET['name'];
        echo "  edit_instrument( {$id}, '{$name}' );";
    } else if( $action == 'create_instrument' ) {
        echo "  create_instrument();";
    } else if( $action == 'list_groups' ) {
        echo "  list_groups();";
    } else {
        echo "  alert( 'unsupported action: {$action}' );";
    }
} else {
    echo <<<HERE
  set_context( 'Home >' );
  load( 'Welcome.php', 'workarea' );
HERE;
}
echo <<<HERE
}
</script>
HERE;
?>

<script type="text/javascript">

function leave_current_app() {
    alert (
        "Warning! You're about to leave the current application. "+
        "All currently oopen connections will be closed."+
        "And all unsaved data will be lost!\n"+
        "Click 'Yes' if you sure you want to proceed. Click 'CAncel' otherwise." );
}

function set_context( context ) {
    document.getElementById('context').innerHTML = context;
}

function Table( itsTableName, itsColumnDefs, itsDataRequest, hasPaginator ) {
    this.name = itsTableName;
    this.columnDefs = itsColumnDefs;
    this.fieldsDefs = [];
    for(i=0; i < itsColumnDefs.length; i++)
        this.fieldsDefs.push( itsColumnDefs[i].key );
    this.dataSource = new YAHOO.util.DataSource( itsDataRequest );
    this.dataSource.responseType = YAHOO.util.DataSource.TYPE_JSON;
    this.dataSource.connXhrMode = "queueRequests";
    this.dataSource.responseSchema = {
        resultsList: "ResultSet.Result",
        fields:      this.fieldsDefs };
    this.paginator = null;
    if( hasPaginator )
        this.paginator = new YAHOO.widget.Paginator (
            {   containers : [this.name+"_table_paginator"],
                rowsPerPage: 20
            }
        );
    this.dataTable = new YAHOO.widget.DataTable(
        this.name+"_table_body",
        this.columnDefs,
        this.dataSource,
        { paginator: this.paginator/*new YAHOO.widget.Paginator( { rowsPerPage: 10 } )*/,
          initialRequest: "" } );

    this.refreshTable = function() {
        this.dataSource.sendRequest(
            "",
            { success: function() {
                  this.set( "sortedBy", null);
                  this.onDataReturnReplaceRows.apply( this, arguments );
              },
              failure: function() {
                  this.showTableMessage(
                      YAHOO.widget.DataTable.MSG_ERROR,
                      YAHOO.widget.DataTable.CLASS_ERROR );
                  this.onDataReturnAppendRows.apply( this, arguments );
              },
              scope: this.dataTable } ); };
}

function create_button( elementId, func2proceed ) {
    this.oButton = new YAHOO.widget.Button(
        elementId,
        {   type:  "submit",
            value: elementId+"_value" } );

    this.oButton.on(
        "click",
        function( p_oEvent ) {
            func2proceed();
        }
    );
    return this;
}

function list_experiments() {

    set_context( 'Home > Select Experiment >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "instrument",  sortable: true,  resizeable: true },
          { key: "experiment",  sortable: true,  resizeable: true },
          { key: "begin_time",  sortable: true,  resizeable: true },
          { key: "end_time",    sortable: true,  resizeable: true },
          { key: "description", sortable: false, resizeable: true } ],
        'RegDBRequestExperiments.php',
        true
    );
    table.refreshTable();
}

function create_params_table( source, paginator ) {
    this.paramsTableShown = false;
    this.oPushButton = new YAHOO.widget.Button( "params_button" );
    this.oPushButton.on (
        "click",
        function( p_oEvent ) {

            document.getElementById('params').innerHTML=
                '  <div id="params_table_paginator"></div>'+
                '  <div id="params_table_body"></div>';

            if( !this.paramsTableShown ) {
                var table = new Table (
                    "params",
                    [ { key: "name",        sortable: true,  resizeable: true },
                      { key: "value",       sortable: false, resizeable: true },
                      { key: "description", sortable: false, resizeable: true } ],
                    source,
                    paginator
                );
                table.refreshTable();
            }
            this.paramsTableShown = !this.paramsTableShown;
        }
    );
}

function view_experiment( id, name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_experiments()">Select Experiment</a> > '+
        '<i>'+name+'</i>' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="edit_button" title="bring in an experiment editor to modify the experiment records">Edit</button>'+
        '  <button id="delete_button" title="destroy the experiment from the database">Delete</button>'+
        '</div>'+
        '<br>'+
        '<div id="experiment_info"></div>'+
        '<br>'+
        '<div id="params_actions_container">'+
        '  <button id="params_button">Parameters &gt;</button>'+
        '</div>'+
        '<br>'+
        '<div id="params"></div>';

    load( 'DisplayExperiment_v1.php?id='+id, 'experiment_info' );

    var params = create_params_table(
        'RegDBRequestExperimentParams.php?id='+id,
        false );

    var action_edit = create_button (
        "edit_button",
        function() { edit_experiment( id, name ); } );

    var action_delete = create_button (
        "delete_button",
        function() { delete_experiment( id, name ); } );
}

function edit_experiment( id, name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_experiments()">Select Experiment</a> > Edit > '+
        '<i>'+name+'</i>' );

    document.getElementById('workarea').innerHTML=
        '<div id="experiment_info">'+
        '  <form name="edit_experiment_form" action="ProcessEditExperiment_v1.php" method="post">'+
        '    <div id="experiment_info_within_form"></div>'+
        '    <input type="hidden" name="id" value="'+id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="view_experiment" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="params_actions_container">'+
        '  <button id="params_button">Parameters &gt;</button>'+
        '</div>'+
        '<br>'+
        '<div id="params"></div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Submit</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'EditExperiment_v1.php?id='+id, 'experiment_info_within_form' );

    var params = create_params_table(
        'RegDBRequestExperimentParams.php?id='+id,
        false );

    var submit = create_button (
        "submit_button",
        function() { document.edit_experiment_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { view_experiment( id, name ); } );
}

function create_experiment( ) {

    set_context( 'Home > Create New Experiment > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="experiment_info">'+
        '  <form name="create_experiment_form" action="ProcessCreateExperiment_v1.php" method="post">'+
        '    <div id="experiment_info_within_form"></div>'+
        '    <input type="hidden" name="actionSuccess" value="view_experiment" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Submit</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'CreateExperiment_v1.php', 'experiment_info_within_form' );

    var submit = create_button (
        "submit_button",
        function() { document.create_experiment_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { list_experiments(); } );
}

function delete_experiment( id, name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_experiments()">Select Experiment</a> > Delete > '+
        '<i>'+name+'</i>' );

    document.getElementById('workarea').innerHTML=
        '<div id="experiment_info">'+
        '  <center><p style="color:red; font-size:24px;"><b>WARNING! WARNING! WARNING!</b></p></center>'+
        '  <br>'+
        '  <center>'+
        '    <p style="font-size:18px; width:640px;"><b>You are about to delete the selected experiment. '+
        '    This is irreversable operation.'+
        '    Think well before hitting the button!</b></p>'+
        '  </center>'+
        '  <br>'+
        '  <br>'+
        '  <form name="delete_experiment_form" action="ProcessDeleteExperiment_v1.php" method="post">'+
        '    <div id="experiment_info_within_form"></div>'+
        '    <input type="hidden" name="id" value="'+id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="list_experiments" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="params_actions_container">'+
        '  <button id="params_button">Parameters &gt;</button>'+
        '</div>'+
        '<br>'+
        '<div id="params"></div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Delete</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'DisplayExperiment_v1.php?id='+id, 'experiment_info_within_form' );

    var params = create_params_table (
        'RegDBRequestExperimentParams.php?id='+id,
        false );

    var submit = create_button (
        "submit_button",
        function() { document.delete_experiment_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { view_experiment( id, name ); } );
}

function list_instruments() {

    set_context( 'Home > Select Instrument >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "instrument",  sortable: true,  resizeable: true },
          { key: "description", sortable: false, resizeable: true } ],
        'RegDBRequestInstruments.php',
        false
    );
    table.refreshTable();
}

function view_instrument( id, name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_instruments()">Select Instrument</a> > '+
        '<i>'+name );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="edit_button" title="bring in an instrument editor to modify the instrument records">Edit</button>'+
        '  <button id="delete_button" title="destroy the instrument from the database">Delete</button>'+
        '</div>'+
        '<br>'+
        '<div id="instrument_info"></div>'+
        '<br>'+
        '<div id="params_actions_container">'+
        '  <button id="params_button">Parameters &gt;</button>'+
        '</div>'+
        '<br>'+
        '<div id="params">'+
        '  <div id="params_table_paginator"></div>'+
        '  <div id="params_table_body"></div>'+
        '</div>';

    load( 'DisplayInstrument_v1.php?id='+id, 'instrument_info' );

    var params = create_params_table (
        'RegDBRequestInstrumentParams.php?id='+id,
        false );

    var action_edit = create_button (
        "edit_button",
        function() { edit_instrument( id, name ); } );

    var action_delete = create_button (
        "delete_button",
        function() { delete_instrument( id, name ); } );
}

function create_instrument( ) {

    set_context( 'Home > Create New Instrument > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="instrument_info">'+
        '  <form name="create_instrument_form" action="ProcessCreateInstrument_v1.php" method="post">'+
        '    <div id="instrument_info_within_form"></div>'+
        '    <input type="hidden" name="actionSuccess" value="edit_instrument" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Submit</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'CreateInstrument_v1.php', 'instrument_info_within_form' );

    var submit = create_button (
        "submit_button",
        function() { document.create_instrument_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { list_instruments(); } );
}

function edit_instrument( id, name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_instruments()">Select Instrument</a> > Edit > '+
        '<i>'+name+'</i>' );

    document.getElementById('workarea').innerHTML=
        '<div id="instrument_info">'+
        '  <form name="edit_instrument_form" action="ProcessEditInstrument_v1.php" method="post">'+
        '    <div id="instrument_info_within_form"></div>'+
        '    <input type="hidden" name="id" value="'+id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="view_instrument" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="params_actions_container">'+
        '  <button id="params_button" >Parameters &gt;</button>'+
        '</div>'+
        '<br>'+
        '<div id="params">'+
        '  <div id="params_table_paginator"></div>'+
        '  <div id="params_table_body"></div>'+
        '</div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Submit</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'EditInstrument_v1.php?id='+id, 'instrument_info_within_form' );

    var params = create_params_table (
        'RegDBRequestInstrumentParams.php?id='+id,
        false );

    var submit = create_button (
        "submit_button",
        function() { document.edit_instrument_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { view_instrument( id, name ); } );

}

function create_instrument( ) {

    set_context( 'Home > Create New Instrument > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="instrument_info">'+
        '  <form name="create_instrument_form" action="ProcessCreateInstrument_v1.php" method="post">'+
        '    <div id="instrument_info_within_form"></div>'+
        '    <input type="hidden" name="actionSuccess" value="edit_instrument" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Submit</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'CreateInstrument_v1.php', 'instrument_info_within_form' );

    var submit = create_button (
        "submit_button",
        function() { document.create_instrument_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { list_instruments(); } );
}

function delete_instrument( id, name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_instruments()">Select Instrument</a> > Delete > '+
        '<i>'+name+'</i>' );

    document.getElementById('workarea').innerHTML=
        '<div id="instrument_info">'+
        '  <center><p style="color:red; font-size:24px;"><b>WARNING! WARNING! WARNING!</b></p></center>'+
        '  <br>'+
        '  <center>'+
        '    <p style="font-size:18px; width:640px;"><b>You are about to delete the selected instrument. '+
        '    This is irreversable operation. Also note that the relevant experiments will be deleted as well.'+
        '    Think well before hitting the button!</b></p>'+
        '  </center>'+
        '  <br>'+
        '  <br>'+
        '  <form name="delete_instrument_form" action="ProcessDeleteInstrument_v1.php" method="post">'+
        '    <div id="instrument_info_within_form"></div>'+
        '    <input type="hidden" name="id" value="'+id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="list_instruments" />'+
        '  </form>'+
        '</div>'+
        '<br>'+
        '<div id="params_actions_container">'+
        '  <button id="params_button">Parameters &gt;</button>'+
        '</div>'+
        '<br>'+
        '<div id="params"></div>'+
        '<br>'+
        '<div id="actions_container">'+
        '  <button id="submit_button">Delete</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>';

    load( 'DisplayInstrument_v1.php?id='+id, 'instrument_info_within_form' );

    var params = create_params_table (
        'RegDBRequestInstrumentParams.php?id='+id,
        false );

    var submit = create_button (
        "submit_button",
        function() { document.delete_instrument_form.submit(); } );

    var cancel = create_button (
        "cancel_button",
        function() { view_instrument( id, name ); } );
}

function list_groups() {

    set_context( 'Home > Select POSIX Group >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "group",   sortable: true, resizeable: true },
          { key: "members", sortable: true, resizeable: true } ],
        'RegDBRequestGroups.php',
        false
    );
    table.refreshTable();
}

function view_group( name ) {

    set_context(
        'Home > '+
        '<a href="javascript:list_groups()">Select POSIX Group</a> > '+
        '<i>'+name+'</i>' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "uid",   sortable: true, resizeable: true },
          { key: "name",  sortable: true, resizeable: true },
          { key: "email", sortable: true, resizeable: true } ],
        'RegDBRequestGroupMembers.php?name='+name,
        false
    );
    table.refreshTable();
}

function run_numbers( name ) {
    set_context(
        'Home > Run Numbers Generator > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "instrument",   sortable: true, resizeable: true },
          { key: "experiment",   sortable: true, resizeable: true },
          { key: "last_run",     sortable: true, resizeable: true },
          { key: "request_time", sortable: true, resizeable: true } ],
        'RegDBRequestRunNumbers.php?name='+name,
        false
    );
    table.refreshTable();}

</script>

    </head>
    <body class="yui-skin-sam" id="body" onload="init()">
        <div id="application_title">
            <p>Experiment Registry Database</p>
            <p style="font-size:32px;"><i>LCLS Detector Control</i></p>
        </div>
        <div id="menubar" class="yuimenubar yuimenubarnav">
            <div class="bd">
                <ul class="first-of-type">
                    <li class="yuimenubaritem first-of-type">
                        <a class="yuimenubaritemlabel" href="#applications" style="color:red; font-weight:bold;">Applications</a>
                    </li>
                    <li class="yuimenubaritem">
                        <a class="yuimenubaritemlabel" href="RegDB_v1.php">Home</a>
                    </li>
                    <li class="yuimenubaritem">
                        <a class="yuimenubaritemlabel" href="#experiments">Experiments</a>
                    </li>
                    <li class="yuimenubaritem">
                        <a class="yuimenubaritemlabel" href="#instruments">Instruments</a>
                    </li>
                    <li class="yuimenubaritem">
                        <a class="yuimenubaritemlabel" href="javascript:list_groups()">POSIX Groups</a>
                    </li>
                    <li class="yuimenubaritem">
                        <a class="yuimenubaritemlabel" href="javascript:run_numbers()">Run Numbers</a>
                    </li>
                    <li class="yuimenubaritem">
                        <a class="yuimenubaritemlabel" href="#">Help</a>
                    </li>
                </ul>
            </div>
        </div>
        <p id="context">Home > </p>
        <br>
        <div id="workarea"></div>
        <br>
        <br>
        <div id="application_footer">
            <center>2009 <a href="http://www.slac.stanford.edu">SLAC National Accelerator Laboratory</a></center>
        </div>
    </body>
</html>
