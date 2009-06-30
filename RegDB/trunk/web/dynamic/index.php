<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title>Experiment Registry Database</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">

<!--
Standard reset, fonts and grids
-->
<link rel="stylesheet" type="text/css" href="/yui/build/reset-fonts-grids/reset-fonts-grids.css">

<!--
CSS for YUI
-->
<link rel="stylesheet" type="text/css" href="/yui/build/fonts/fonts-min.css">
<link rel="stylesheet" type="text/css" href="/yui/build/menu/assets/skins/sam/menu.css">
<link rel="stylesheet" type="text/css" href="/yui/build/paginator/assets/skins/sam/paginator.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/datatable/assets/skins/sam/datatable.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/button/assets/skins/sam/button.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/container/assets/skins/sam/container.css" />

<!--
Page-specific styles
-->
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
    font-size:16px;
    border:solid 4px transparent;
    border-left-width:16px;
    text-align:left;
}
#workarea {
    margin-left:40px;
    margin-right:40px;
}
#experiment_info,
#instrument_info,
#runs_info {
    margin-top:0px;
    margin-left:4px;
}
#workarea_table_container table,
#params_table_container   table,
#runs_table_container     table {
}
#workarea_table_paginator,
#params_table_page,
#runs_table_page {
    margin-left:auto;
    margin-right:auto;
}
#workarea_table_container,
#workarea_table_container .yui-dt-loading,
#params_table_container,
#params_table_container .yui-dt-loading,
#runs_table_container,
#runs_table_container .yui-dt-loading {
    text-align:center;
    background-color:transparent;
}
#actions_container,
#params_actions_container,
#runs_actions_container {
    margin-top:24px;
    margin-left:0px;
    text-align:left;
}
</style>

<!--
Dependency source files
-->
<script type="text/javascript" src="/yui/build/yahoo-dom-event/yahoo-dom-event.js"></script>
<script type="text/javascript" src="/yui/build/animation/animation.js"></script>
<script type="text/javascript" src="/yui/build/container/container_core.js"></script>

<script type="text/javascript" src="/yui/build/dragdrop/dragdrop-min.js"></script>
<script type="text/javascript" src="/yui/build/container/container-min.js"></script>

<!--
Menu source file
-->
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

<!--
Custom JavaScript
-->
<script type="text/javascript" src="Loader.js"></script>
<script type="text/javascript" src="JSON.js"></script>

<!--
Page-specific script
-->
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
    } else if( $action == 'view_run_numbers' ) {
        $id   = $_GET['id'];
        $name = $_GET['name'];
        echo "  view_run_numbers( {$id}, '{$name}' );";
    } else {
        echo "  alert( 'unsupported action: {$action}' );";
    }
} else {
    echo <<<HERE
  //set_context( 'Home >' );
  //set_context( '' );
  load( 'Welcome.php', 'workarea' );
HERE;
}
echo <<<HERE
}
</script>
HERE;
?>

<script type="text/javascript">

function post_message( id, title, text ) {

    document.getElementById( id ).innerHTML =
        '<div class="hd">'+title+'</div>'+
        '<div class="bd">'+
        '  <center><p>'+text+'</p></center>'+
        '</div>';

    var handleOk = function() { this.submit(); };

    var dialog1 =
        new YAHOO.widget.Dialog (
            id,
			{   width : "480px",
                fixedcenter : true,
				visible : true,
                modal:true,
				constraintoviewport : true,
				buttons : [
                    { text:"Ok", handler: handleOk, isDefault:true }
                ]
			}
        );
    dialog1.render();
}

function post_warning( text ) {
    post_message (
        'popupdialogs',
        '<span style="color:red; font-size:16px;">Warning</span>',
        text );
}


function ask_yesno( title, text, onYes, onNo ) {

    var id = 'popupdialogs';

    document.getElementById( id ).innerHTML =
        '<div class="hd">'+title+'</div>'+
        '<div class="bd">'+
        '  <center><p>'+text+'</p></center>'+
        '</div>';

    var handleYes = function() {
        this.submit();
        onYes();
    };
    var handleNo = function() {
        this.cancel();
        onNo();
    };
    var dialog1 = new YAHOO.widget.Dialog (
        id,
        {   width : "480px",
            fixedcenter : true,
            visible : true,
            modal:true,
            constraintoviewport : true,
            buttons : [
                { text:"Yes", handler: handleYes },
                { text:"No",  handler: handleNo, isDefault:true }
            ]
        }
    );
    dialog1.render();
}

function ask_yesno_confirmation( text, onYes, onNo ) {
    ask_yesno (
        '<span style="color:red; font-size:16px;">Confirmation Request</span>',
        text,
        onYes, onNo
    );
}


function leave_current_app() {
    post_warning (
        "You're about to leave the current application. "+
        "All currently open connections will be closed, and "+
        "all unsaved data will be lost! Click <b>Yes</b> if you sure "+
        "you want to proceed. Click <b>Cancel</b> to abort the transition." );
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

function TableLocal( itsTableName, itsColumnDefs, itsDataArray, hasPaginator ) {
    this.name = itsTableName;
    this.columnDefs = itsColumnDefs;
    this.fieldsDefs = [];
    for(i=0; i < itsColumnDefs.length; i++)
        this.fieldsDefs.push( itsColumnDefs[i].key );
    this.dataSource = new YAHOO.util.DataSource( itsDataArray );
    this.dataSource.responseType = YAHOO.util.DataSource.TYPE_JSARRAY;
    this.dataSource.responseSchema = {
        resultsList: "ResultSet.Result",
        fields:      this.fieldsDefs };
    this.paginator = null;
    if( hasPaginator ) {
        this.paginator = new YAHOO.widget.Paginator (
            {   containers : [this.name+"_table_paginator"],
                rowsPerPage: 20
            }
        );
    }
    this.dataTable = new YAHOO.widget.DataTable(
        this.name+"_table_body",
        this.columnDefs,
        this.dataSource,
        { paginator: this.paginator/*new YAHOO.widget.Paginator( { rowsPerPage: 10 } )*/,
          initialRequest: "" } );

    this.highlightEditableCell = function(oArgs) {
        var elCell = oArgs.target;
        if(YAHOO.util.Dom.hasClass(elCell, "yui-dt-editable")) {
            this.highlightCell(elCell);
        }
    };
    this.dataTable.subscribe("cellMouseoverEvent", this.highlightEditableCell);
    this.dataTable.subscribe("cellMouseoutEvent", this.dataTable.onEventUnhighlightCell);
    this.dataTable.subscribe("cellClickEvent", this.dataTable.onEventShowCellEditor);
    this.dataTable.subscribe("checkboxClickEvent", function(oArgs){
        var elCheckbox = oArgs.target;
        var oRecord = this.getRecord( elCheckbox );
        oRecord.setData( "selected", elCheckbox.checked );
    });
    return this;
}

function Table1( itsTableName, itsColumnDefs, itsDataRequest, hasPaginator ) {
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
    if( hasPaginator ) {
        this.paginator = new YAHOO.widget.Paginator (
            {   containers : [this.name+"_table_paginator"],
                rowsPerPage: 20
            }
        );
    }
    this.dataTable = new YAHOO.widget.DataTable(
        this.name+"_table_body",
        this.columnDefs,
        this.dataSource,
        { paginator: this.paginator/*new YAHOO.widget.Paginator( { rowsPerPage: 10 } )*/,
          initialRequest: "" } );

    this.highlightEditableCell = function(oArgs) {
        var elCell = oArgs.target;
        if(YAHOO.util.Dom.hasClass(elCell, "yui-dt-editable")) {
            this.highlightCell(elCell);
        }
    };
    this.dataTable.subscribe("cellMouseoverEvent", this.highlightEditableCell);
    this.dataTable.subscribe("cellMouseoutEvent", this.dataTable.onEventUnhighlightCell);
    this.dataTable.subscribe("cellClickEvent", this.dataTable.onEventShowCellEditor);
    this.dataTable.subscribe("checkboxClickEvent", function(oArgs){
        var elCheckbox = oArgs.target;
        var oRecord = this.getRecord( elCheckbox );
        oRecord.setData( "selected", elCheckbox.checked );
    });
    return this;
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

function create_params_table( source, paginator ) {
    this.tableShown = false;
    this.oPushButton = new YAHOO.widget.Button( "params_button" );
    this.oPushButton.on (
        "click",
        function( p_oEvent ) {

            document.getElementById('params').innerHTML=
                '  <div id="params_table_paginator"></div>'+
                '  <div id="params_table_body"></div>';

            if( !this.tableShown ) {
                var table = new Table (
                    "params",
                    [ { key: "name",        sortable: true,  resizeable: true },
                      { key: "value",       sortable: false, resizeable: true },
                      { key: "description", sortable: false, resizeable: true } ],
                    source,
                    paginator
                );
                //table.refreshTable();
            }
            this.tableShown = !this.tableShown;
        }
    );
}

function create_params_table_editable( source, paginator ) {

    document.getElementById('params').innerHTML=
        '  <div id="params_table_paginator"></div>'+
        '  <div id="params_table_body"></div>';

    /* Decide on an initial source of the information to populate
     * the table from.
     */
    this.source = source;
    if( null == this.source ) this.storage = [];
    else                      this.storage = null;
    this.table = null;

    this.oPushButton = new YAHOO.widget.Button( "params_button" );
    this.oPushButtonAdd = new YAHOO.widget.Button( "add_button" );
    this.oPushButtonAdd.set( 'disabled', true );
    this.oPushButtonRemove = new YAHOO.widget.Button( "remove_button" );
    this.oPushButtonRemove.set( 'disabled', true );

    function synchronize_data( predicate ) {
        var rs = this.table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        this.storage = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( predicate( r ))
                this.storage.push ( {
                    'name': r.getData('name'),
                    'value': r.getData('value'),
                    'description': r.getData('description')} );
        }
    }

    function createTable() {
        if( null == this.storage ) {
            return new Table1(
                "params",
                [ { key: "selected", formatter: "checkbox" },
                  { key: "name", sortable: true, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true}) },
                  { key: "value", sortable: false, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true}) },
                  { key: "description", sortable: false, resizeable: true,
                    editor: new YAHOO.widget.TextareaCellEditor({disableBtns:true}) } ],
                this.source,
                paginator );
        } else {
            return new TableLocal(
                "params",
                [ { key: "selected", formatter: "checkbox" },
                  { key: "name", sortable: true, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true}) },
                  { key: "value", sortable: false, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true}) },
                  { key: "description", sortable: false, resizeable: true,
                    editor: new YAHOO.widget.TextareaCellEditor({disableBtns:true}) } ],
                this.storage,
                paginator );
        }
    }

    function toggleTable() {
        if( this.table == null ) {
            this.table = createTable();
        } else {
            synchronize_data( function() { return true; });
            this.table.dataTable.destroy();
            this.table = null;
        }
        this.oPushButtonAdd.set('disabled', this.table == null );
        this.oPushButtonRemove.set('disabled', this.table == null );
    }
    this.oPushButton.on (
        "click",
        function( p_oEvent ) { toggleTable(); }
    );

    function AddAndRefreshTable() {
        this.table.dataTable.addRow (
            {   'name': "name",
                'value': "value",
                'description': "description" }, 0 );
    }
    this.oPushButtonAdd.on (
        "click",
        function( p_oEvent ) { AddAndRefreshTable(); }
    );
    function deleteAndRefreshTable() {
        synchronize_data( function( r ) { return !r.getData('selected'); } );
        this.table.dataTable.destroy();
        this.table = createTable();
    }
    this.oPushButtonRemove.on (
        "click",
        function( p_oEvent ) { deleteAndRefreshTable(); }
    );
    this.toJSON = function() {
        var result = [];
        if( this.table != null ) {
            var rs = this.table.dataTable.getRecordSet();
            var rs_length = rs.getLength();
            for( var i = 0; i < rs_length; i++ ) {
                var r = rs.getRecord(i);
                result.push ( [
                    r.getData('name'),
                    r.getData('value'),
                    r.getData('description') ] );
            }
        } else if( this.storage != null) {
            for( var i = 0; i < this.storage.length; i++ ) {
                var r = this.storage[i];
                result.push ( [
                    r['name'],
                    r['value'],
                    r['description'] ] );
            }
        } else {
            // Pass this special string to indicate that we don't have any
            // information on parameters.
            //
            return 'null';
        }
        return JSON.stringify( result );
    };
    return this;
}

function create_runs_table( source, paginator ) {
    this.tableShown = false;
    this.oPushButton = new YAHOO.widget.Button( "runs_button" );
    this.oPushButton.on (
        "click",
        function( p_oEvent ) {
            document.getElementById('runs').innerHTML=
                '  <div id="runs_table_paginator"></div>'+
                '  <div id="runs_table_body"></div>';

            if( !this.tableShown ) {
                var table = new Table (
                    "runs",
                    [ { key: "run",          sortable: true,  resizeable: true },
                      { key: "request_time", sortable: false, resizeable: true } ],
                    source,
                    paginator
                );
                table.refreshTable();
            }
            this.tableShown = !this.tableShown;
        }
    );

}

function list_experiments() {

    set_context(
        //'Home > '+
        'Select Experiment >' );

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
        'RequestExperiments.php',
        true
    );
    table.refreshTable();
}

function view_experiment( id, name ) {

    set_context(
        '<a href="javascript:list_experiments()">Select Experiment</a> > '+
        '<i>'+name+'</i>'+
        '&nbsp;&nbsp;&nbsp;|&nbsp;<b>viewing</b>&nbsp;|' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="edit_button" title="bring in an experiment editor to modify the experiment records">Edit</button>'+
        '  <button id="delete_button" title="destroy the experiment from the database">Delete</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="experiment_info" style="height:250px;"></div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button">Parameters &gt;</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'DisplayExperiment.php?id='+id, 'experiment_info' );

    var params = create_params_table(
        'RequestExperimentParams.php?id='+id,
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
        '<a href="javascript:list_experiments()">Select Experiment</a> > '+
        '<i>'+name+'</i>'+
        '&nbsp;&nbsp;&nbsp;|&nbsp;<b><span style="color:red;">editing</span></b>&nbsp;|' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="experiment_info" style="height:250px;">'+
        '    <form name="edit_experiment_form" action="ProcessEditExperiment.php" method="post">'+
        '      <div id="experiment_info_within_form"></div>'+
        '      <input type="hidden" name="id" value="'+id+'" />'+
        '      <input type="hidden" name="actionSuccess" value="view_experiment" />'+
        '      <input type="hidden" name="params" value="" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button" >Parameters &gt;</button>'+
        '    <button id="add_button" >Add</button>'+
        '    <button id="remove_button" >Remove Selected</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'EditExperiment.php?id='+id, 'experiment_info_within_form' );

    var params = create_params_table_editable (
        'RequestExperimentParams.php?id='+id,
        false );

    var save = create_button (
        "save_button",
        function() {
            document.edit_experiment_form.params.value = params.toJSON();
            document.edit_experiment_form.submit();
        }
    );

    var cancel = create_button (
        "cancel_button",
        function() { view_experiment( id, name ); } );
}

function create_experiment( ) {

    set_context(
        'Create New Experiment > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="experiment_info" style="height:250px;">'+
        '    <form name="create_experiment_form" action="ProcessCreateExperiment.php" method="post">'+
        '      <div id="experiment_info_within_form"></div>'+
        '      <input type="hidden" name="actionSuccess" value="view_experiment" />'+
        '      <input type="hidden" name="params" value="" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button" >Parameters &gt;</button>'+
        '    <button id="add_button" >Add</button>'+
        '    <button id="remove_button" >Remove Selected</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'CreateExperiment.php', 'experiment_info_within_form' );

    var params = create_params_table_editable( null, false );

    var save = create_button (
        "save_button",
        function() {
            document.create_experiment_form.params.value = params.toJSON();
            document.create_experiment_form.submit();
        }
    );
    var cancel = create_button (
        "cancel_button",
        function() { list_experiments(); } );
}

function delete_experiment( id, name ) {

    set_context(
        '<a href="javascript:list_experiments()">Select Experiment</a> > '+
        '<i>'+name+'</i>'+
        '&nbsp;&nbsp;&nbsp;|&nbsp;<b><span style="color:red;">deleting</span></b>&nbsp;|' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="experiment_info" style="height:250px;">'+
        '    <form name="delete_experiment_form" action="ProcessDeleteExperiment.php" method="post">'+
        '      <div id="experiment_info_within_form"></div>'+
        '        <input type="hidden" name="id" value="'+id+'" />'+
        '      <input type="hidden" name="actionSuccess" value="list_experiments" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button">Parameters &gt;</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'DisplayExperiment.php?id='+id, 'experiment_info_within_form' );

    var params = create_params_table (
        'RequestExperimentParams.php?id='+id,
        false
    );
    var save = create_button (
        "save_button",
        function() {
            ask_yesno_confirmation (
                'Proceed with the operation and permanently delete this experiment '+
                'and all relevant data from the database? Enter <b>Yes</b> to do so. '+
                'Note, this is the last chance to abort destructive modifications '+
                'in the database!',
                function() { document.delete_experiment_form.submit(); },
                function() { view_experiment( id, name ); }
            );
        }
    );
    var cancel = create_button (
        "cancel_button",
        function() { view_experiment( id, name ); }
    );
    post_warning (
        'You are entering a dialog for deleting the selected experiment. '+
        'Please, be advised that the deletion is an irreversable operation resulting '+
        'in a permanent loss of data and potentially in a lost of referential integrity '+
        'within the ONLINE and OFFLINE data systems.'
    );
}

function list_instruments() {

    set_context(
        //'Home > '+
        'Select Instrument >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "instrument",  sortable: true,  resizeable: true },
          { key: "description", sortable: false, resizeable: true } ],
        'RequestInstruments.php',
        false
    );
    table.refreshTable();
}

function view_instrument( id, name ) {

    set_context(
        '<a href="javascript:list_instruments()">Select Instrument</a> > '+
        '<i>'+name+'</i>'+
        '&nbsp;&nbsp;&nbsp;|&nbsp;<b>viewing</b>&nbsp;|' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="edit_button" title="bring in an instrument editor to modify the instrument records">Edit</button>'+
        '  <button id="delete_button" title="destroy the instrument from the database">Delete</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="instrument_info" style="height:150px;"></div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button">Parameters &gt;</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'DisplayInstrument.php?id='+id, 'instrument_info' );

    var params = create_params_table (
        'RequestInstrumentParams.php?id='+id,
        false
    );
    var action_edit = create_button (
        "edit_button",
        function() { edit_instrument( id, name ); }
    );
    var action_delete = create_button (
        "delete_button",
        function() { delete_instrument( id, name ); }
    );
}

function edit_instrument( id, name ) {

    set_context(
        '<a href="javascript:list_instruments()">Select Instrument</a> > '+
        '<i>'+name+'</i>'+
        '&nbsp;&nbsp;&nbsp;|&nbsp;<b><span style="color:red;">editing</span></b>&nbsp;|' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="instrument_info" style="height:150px;">'+
        '    <form name="edit_instrument_form" action="ProcessEditInstrument.php" method="post">'+
        '      <div id="instrument_info_within_form"></div>'+
        '      <input type="hidden" name="id" value="'+id+'" />'+
        '      <input type="hidden" name="actionSuccess" value="view_instrument" />'+
        '      <input type="hidden" name="params" value="" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button" >Parameters &gt;</button>'+
        '    <button id="add_button" >Add</button>'+
        '    <button id="remove_button" >Remove Selected</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'EditInstrument.php?id='+id, 'instrument_info_within_form' );

    var params = create_params_table_editable (
        'RequestInstrumentParams.php?id='+id,
        false
    );
    var save = create_button (
        "save_button",
        function() {
            document.edit_instrument_form.params.value = params.toJSON();
            document.edit_instrument_form.submit();
        }
    );
    var cancel = create_button (
        "cancel_button",
        function() { view_instrument( id, name ); }
    );
}

var params = null;

function create_instrument( ) {

    set_context(
        'Create New Instrument > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="instrument_info" style="height:150px;">'+
        '    <form name="create_instrument_form" action="ProcessCreateInstrument.php" method="post">'+
        '      <div id="instrument_info_within_form"></div>'+
        '      <input type="hidden" name="actionSuccess" value="view_instrument" />'+
        '      <input type="hidden" name="params" value="" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button" >Parameters &gt;</button>'+
        '    <button id="add_button" >Add</button>'+
        '    <button id="remove_button" >Remove Selected</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'CreateInstrument.php', 'instrument_info_within_form' );

    var params = create_params_table_editable( null, false );

    var save = create_button (
        "save_button",
        function() {
            document.create_instrument_form.params.value = params.toJSON();
            document.create_instrument_form.submit();
        }
    );
    var cancel = create_button (
        "cancel_button",
        function() { list_instruments(); }
    );
}

function delete_instrument( id, name ) {

    set_context(
        '<a href="javascript:list_instruments()">Select Instrument</a> > '+
        '<i>'+name+'</i>'+
        '&nbsp;&nbsp;&nbsp;|&nbsp;<b><span style="color:red;">deleting</span></b>&nbsp;|' );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button" title="proceed with the deletion">Save</button>'+
        '  <button id="cancel_button" title="cancel the attempt">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="instrument_info" style="height:150px;">'+
        '    <form name="delete_instrument_form" action="ProcessDeleteInstrument.php" method="post">'+
        '      <div id="instrument_info_within_form"></div>'+
        '      <input type="hidden" name="id" value="'+id+'" />'+
        '      <input type="hidden" name="actionSuccess" value="list_instruments" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div id="params_actions_container">'+
        '    <button id="params_button">Parameters &gt;</button>'+
        '  </div>'+
        '  <div id="params" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="params_table_paginator"></div>'+
        '    <div id="params_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'DisplayInstrument.php?id='+id, 'instrument_info_within_form' );

    var params = create_params_table (
        'RequestInstrumentParams.php?id='+id,
        false
    );
    var save = create_button (
        "save_button",
        function() {
            ask_yesno_confirmation (
                'Proceed with the operation and permanently delete this instrument '+
                'and all relevant data from the database? Enter <b>Yes</b> to do so. '+
                'Note, this is the last chance to abort destructive modifications '+
                'in the database!',
                function() { document.delete_instrument_form.submit(); },
                function() { view_instrument( id, name ); }
            );
        }
    );
    var cancel = create_button (
        "cancel_button",
        function() { view_instrument( id, name ); }
    );
    post_warning (
        'You are entering a dialog for deleting the selected instrument. '+
        'Please, be advised that the deletion is an irreversable operation resulting '+
        'in a permanent loss of data and potentially in a lost of referential integrity '+
        'within the ONLINE and OFFLINE data systems.'
    );
}

function list_groups() {

    set_context(
        //'Home > '+
        'Select POSIX Group >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "group",   sortable: true, resizeable: true },
          { key: "members", sortable: true, resizeable: true } ],
        'RequestGroups.php',
        false
    );
    table.refreshTable();
}

function view_group( name ) {

    set_context(
        //'Home > '+
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
        'RequestGroupMembers.php?name='+name,
        false
    );
    table.refreshTable();
}

function run_numbers() {
    set_context(
        'Run Numbers Generator > ' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "instrument",   sortable: true, resizeable: true },
          { key: "experiment",   sortable: true, resizeable: true },
          { key: "last_run_num", sortable: true, resizeable: true },
          { key: "request_time", sortable: true, resizeable: true } ],
        'RequestRunNumbers.php',
        true
    );
    table.refreshTable();
}

function view_run_numbers( id, name ) {

    set_context(
        '<a href="javascript:run_numbers()">Run Numbers Generator</a> > '+
        '<i>'+name );

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        //'  <button disabled="disabled" id="generate_button" title="This will generate next run number. This operation is used for testing purposes only.">Generate Next Run</button>'+
        '  <button id="generate_button" title="This will generate next run number. This operation is used for testing purposes only.">Generate Next Run</button>'+
        '  <form name="generate_run_form" action="ProcessGenerateRun.php" method="post">'+
        '    <input type="hidden" name="id" value="'+id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="view_run_numbers" />'+
        '  </form>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="runs_info" style="height:125px;">'+
        '  </div>'+
        '  <br>'+
        '  <div id="runs_actions_container">'+
        '    <button id="runs_button">Runs &gt;</button>'+
        '  </div>'+
        '  <div id="runs" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="runs_table_paginator"></div>'+
        '    <div id="runs_table_body"></div>'+
        '  </div>'+
        '</div>';

    load( 'DisplayRunNumbers.php?exper_id='+id, 'runs_info' );

    var generate = create_button (
        "generate_button",
        function() { document.generate_run_form.submit(); }
    );
    var runs = create_runs_table (
        'RequestRunNumbers.php?exper_id='+id,
        false
    );
}

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
                    <a class="yuimenubaritemlabel" href="index.php">Home</a>
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
    <!--
    <p id="context">Home > </p>
    -->
    <p id="context"></p>
    <br>
    <div id="workarea"></div>
    <div id="popupdialogs"></div>
    <!--
    <br>
    <br>
    <div id="application_footer">
        <center>2009 <a href="http://www.slac.stanford.edu">SLAC National Accelerator Laboratory</a></center>
    </div>
    -->
</body>
</html>
