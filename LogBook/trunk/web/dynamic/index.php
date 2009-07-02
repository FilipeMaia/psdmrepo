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
#application_header {
    background-color:#d0d0d0;
    padding:12px;
    margin:0px;
}
#application_title {
    font-family: "Times", serif;
    font-size:42px;
}
#current_selection {
    color:#0071bc;
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
#runs_info,
#shifts_info {
    margin-top:0px;
    margin-left:4px;
}
#workarea_table_container table,
#params_table_container   table,
#runs_table_container     table,
#shifts_table_container   table,
#messages_table_container table {
}
#workarea_table_paginator,
#params_table_page,
#runs_table_paginator,
#shifts_table_paginator,
#messages_table_paginator {
    margin-left:auto;
    margin-right:auto;
}
#workarea_table_container,
#workarea_table_container .yui-dt-loading,
#params_table_container,
#params_table_container .yui-dt-loading,
#runs_table_container,
#runs_table_container .yui-dt-loading,
#shifts_table_container,
#shifts_table_container .yui-dt-loading,
#messages_table_container,
#messages_table_container .yui-dt-loading {
    text-align:center;
    background-color:transparent;
}
#actions_container,
#params_actions_container,
#runs_actions_container,
#shifts_actions_container,
#messages_actions_container {
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
<script type="text/javascript" src="Menubar.js"></script>
<script type="text/javascript" src="Dialogs.js"></script>
<script type="text/javascript" src="Loader.js"></script>
<script type="text/javascript" src="JSON.js"></script>

<!--
Page-specific script
-->
<script type="text/javascript">

/*
 * The current experiment selection (if any) is represented by
 * this dictionary which has the following keys:
 *
 *   { instrument: {
 *       id: <id>,
 *       name: <name>
 *     },
 *     experiment: {
 *       id: <id>,
 *       name: <name>
 *     },
 *     shift: {
 *       id: <id or null>
 *     },
 *     run: {
 *       id: <id or null>
 *     }
 *   }
 */
var current_selection = null;

function set_current_selection( instr_id, instr_name, exper_id, exper_name ) {
    current_selection = {
        instrument: {
            id:   instr_id,
            name: instr_name
        },
        experiment: {
            id:   exper_id,
            name: exper_name
        },
        shift: {
            id: null
        },
        run: {
            id: null
        }
    };
    document.getElementById( "current_selection" ).innerHTML =
        instr_name+' / '+exper_name;

    menubar_enable( menubar_group_shifts );
    menubar_enable( menubar_group_runs );
}

var dialog_element = "popupdialogs";

var menubar_element = "menubar";
var menubar_data = [];

var menubar_group_applications = menubar_data.length;
menubar_data.push ( {
    id: 'applications',
    href: '#applications',
    title: 'Applications',
    title_style: 'font-weight:bold;',
    itemdata: [
        { text: "Experiment Registry Database", url: "javascript:leave_current_app()" },
        { text: "Electronic Log Book", url: "javascript:leave_current_app()" } ],
    disabled: false }
);
var menubar_group_home = menubar_data.length;
menubar_data.push ( {
    id: null,
    href: 'index.php',
    title: 'Home',
    title_style: null,
    itemdata: null,
    disabled: false }
);
var menubar_group_experiments = menubar_data.length;
menubar_data.push ( {
    id:    'experiments',
    href:  '#experiments',
    title: 'Experiments',
    title_style: null,
    itemdata: [
        { text: "Select..", url: "javascript:list_experiments()" } ],
    disabled: false }
);
var menubar_group_shifts = menubar_data.length;
menubar_data.push ( {
    id:    'shifts',
    href:  '#shifts',
    title: 'Shifts',
    title_style: null,
    itemdata: [
        { text: "Select..", url: "javascript:list_shifts()" },
        { text: "Last shift", url: "javascript:select_last_shift()" } ],
    disabled: true }
);
var menubar_group_runs = menubar_data.length;
menubar_data.push ( {
    id:    'runs',
    href:  '#runs',
    title: 'Runs',
    title_style: null,
    itemdata: [
        { text: "Select..", url: "javascript:list_runs()" },
        { text: "Last run", url: "javascript:select_last_run()" } ],
    disabled: true }
);
var menubar_group_browse = menubar_data.length;
menubar_data.push ( {
    id:    null,
    href:  'javascript:browse_contents()',
    title: 'Browse',
    itemdata: null,
    disabled: false }
);
var menubar_group_search = menubar_data.length;
menubar_data.push ( {
    id:    null,
    href:  'javascript:search_contents()',
    title: 'Search',
    itemdata: null,
    disabled: false }
);
var menubar_group_help = menubar_data.length;
menubar_data.push ( {
    id:    'help',
    href:  '#help',
    title: 'Help',
    title_style: null,
    itemdata: [
        { text: "Help contents...", url: "#" },
        { text: "Help with the current page...", url: "#" },
        { text: "About the application", url: "#" } ],
    disabled: false }
);

YAHOO.util.Event.onContentReady (
    menubar_element,
    function () {
        menubar_create (
            menubar_element,
            menubar_data );
    }
);

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
    echo "  load( 'Welcome.php', 'workarea' );";
}
echo <<<HERE
}
</script>
HERE;
?>

<script type="text/javascript">

function leave_current_app() {
    post_warning (
        dialog_element,
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
            {   containers : [this.name+"_paginator"],
                rowsPerPage: 20
            }
        );
    this.dataTable = new YAHOO.widget.DataTable(
        this.name+"_body",
        this.columnDefs,
        this.dataSource,
        { paginator: this.paginator/*new YAHOO.widget.Paginator( { rowsPerPage: 10 } )*/,
          initialRequest: "" } );

    this.refreshTable = function() {
        this.dataSource.sendRequest(
            "",
            { success: function() {
                  this.set( "sortedBy", null );
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
/*
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
        { paginator: this.paginator,
                     // new YAHOO.widget.Paginator( { rowsPerPage: 10 } ),
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
        { paginator: this.paginator,
                     // new YAHOO.widget.Paginator( { rowsPerPage: 10 } ),
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
*/

function create_button( elementId, func2proceed ) {
    this.oButton = new YAHOO.widget.Button(
        elementId,
        {   /*type:  "submit",*/
            value: elementId+"_value" } );

    this.oButton.on(
        "click",
        function( p_oEvent ) {
            func2proceed();
        }
    );
    this.enable = function() {
        this.oButton.set( 'disabled', false );
    }
    this.disable = function() {
        this.oButton.set( 'disabled', true );
    }
    return this;
}

/*
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

    // Decide on an initial source of the information to populate
    // the table from.
    //
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
*/

function create_runs_table( source, paginator ) {

    document.getElementById('runs_table').innerHTML=
        '  <div id="runs_table_paginator"></div>'+
        '  <div id="runs_table_body"></div>';

    var table = new Table (
        "runs_table",
        [ { key: "num",              sortable: true, resizeable: false },
          { key: "begin_time",       sortable: true, resizeable: false },
          { key: "end_time",         sortable: true, resizeable: false },
          { key: "shift_begin_time", sortable: true, resizeable: false } ],
        source,
        paginator
    );
    //table.refreshTable();
}

function create_shifs_table( source, paginator ) {

    document.getElementById('shifts_table').innerHTML=
        '  <div id="shifts_table_paginator"></div>'+
        '  <div id="shifts_table_body"></div>';

    var table = new Table (
        "shifts_table",
        [ { key: "begin_time", sortable: true, resizeable: false },
          { key: "end_time",   sortable: true, resizeable: false },
          { key: "leader",     sortable: true, resizeable: false },
          { key: "num_runs",   sortable: true, resizeable: false } ],
        source,
        paginator
    );
    //table.refreshTable();
}

function list_experiments() {

    set_context(
        'Select Experiment >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea_table",
        [ { key: "instrument",        sortable: true,  resizeable: false },
          { key: "experiment",        sortable: true,  resizeable: false },
          { key: "status",            sortable: true,  resizeable: false },
          { key: "begin_time",        sortable: true,  resizeable: false },
          { key: "end_time",          sortable: true,  resizeable: false },
          { key: "registration_time", sortable: true,  resizeable: false },
          { key: "description",       sortable: false, resizeable: true } ],
        'RequestExperiments.php',
        false
    );
    table.refreshTable();
}

function select_experiment( instr_id, instr_name, exper_id, exper_name ) {
    set_current_selection( instr_id, instr_name, exper_id, exper_name );
    display_experiment();
}

var new_message=null;

function display_experiment() {

    set_context ( 'Experiment Summary >' );

    document.getElementById('workarea').innerHTML=
//        '<div id="actions_container">'+
//        '  <button id="detail_button" title="'+
//        'press the button to open a separate window with a detailed description '+
//        'of the experiment from the Experiments Registry Database '+
//        '">Get Registration Info</button>'+
//        '</div>'+
        '<div style="margin-top:0px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="experiment_info"></div>'+
        '  <br>'+
        '  <div id="messages_actions_container">'+
        '    <button id="new_message_button">New Message &gt;</button>'+
        '    <button id="message_extend_button">Extended &gt;</button>'+
        '    <button id="message_submit_button">Submit</button>'+
        '    <div id="new_message_dialog"></div>'+
        '  </div>'+
        '  <div id="messages_table" style="margin-left:25px; margin-top:25px;">'+
        '    <div id="messages_table_paginator"></div>'+
        '    <div id="messages_table_body"></div>'+
        '  </div>'+
//        '</div>'+
//        '<div style="margin-top:0px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
//        '  <div id="messages"></div>'+
//        '    <div style="position:relative;">'+
//        '      <div style="position:absolute; left:0px; top:0px; text-align:left; color:#0071bc; font-weight:bold;">'+
//        '        New Message: '+
//        '      </div>'+
//        '      <div style="position:absolute; left:100px; top:0px; text-align:left;">'+
//        '        <input id="message_text_id" type="text" name="message_text" style="padding:1px; width:400px;" />'+
//        '      </div>'+
//        '    </div>'+
//        '  </div>'+
        '</div>';

    load( 'DisplayExperiment.php?id='+current_selection.experiment.id, 'experiment_info' );

    YAHOO.util.Event.onContentReady (
        "detail_button",
        function () {
            var action_edit = create_button (
                "detail_button",
                function() {
                    window.open (
                        '/tests/RegDB/dynamic/index.php?action=view_experiment&id='+
                        current_selection.experiment.id+
                        '&name='+current_selection.experiment.name,
                        'mywindow'/*,'width=1280,height=1024'*/
                    );
                }
            );
        }
    );

    YAHOO.util.Event.onContentReady (
        "runs_table",
        function () {
            var runs = create_runs_table (
                'RequestRuns.php?id='+current_selection.experiment.id+'&last',
                false
            );
        }
    );
    YAHOO.util.Event.onContentReady (
        "shifts_table",
        function () {
            var shifts = create_shifs_table (
                'RequestShifts.php?id='+current_selection.experiment.id+'&last',
                false
            );
        }
    );
    new_message = create_new_message_dialog( 'experiment' );
    /*
    var messages = create_messages_table (
        'DisplayMessages.php?id='+current_selection.experiment.id+'&scope=experiment',
        false
    );
    */
}
function create_new_message_dialog( scope ) {

    document.getElementById('new_message_dialog').innerHTML=
        '<form name="new_message_form" action="NewFFEntry.php" method="post">'+
        '  <input type="hidden" name="id" value="'+current_selection.experiment.id+'" />'+
        '  <input type="hidden" name="scope" value="'+scope+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment" />'+
        '  <div id="new_message_body" style="margin-top:10px; padding:1px;"></div>'+
        '</form>';


    this.dialogShown = false;
    this.optionsShown = false;

    this.new_message_button     = new YAHOO.widget.Button( "new_message_button" );
    this.message_extend_button = new YAHOO.widget.Button( "message_extend_button" );
    this.message_submit_button  = new YAHOO.widget.Button( "message_submit_button" );

    this.message_extend_button.set( 'disabled', true );
    this.message_submit_button.set( 'disabled', true );

    function onNewMessageClick() {
        if( !this.dialogShown ) {
            this.dialogShown = true;
            this.message_extend_button.set( 'disabled', false );
            this.message_submit_button.set( 'disabled', false );
            document.getElementById('new_message_body').innerHTML=
                '<input id="message_text_id" type="text" name="message_text" style="padding:1px; width:400px;" />';
        } else {
            this.dialogShown = false;
            this.optionsShown = false;
            this.message_extend_button.set( 'disabled', true );
            this.message_submit_button.set( 'disabled', true );
            document.getElementById('new_message_body').innerHTML='';
        }
    }
    this.new_message_button.on (
        "click",
        function( p_oEvent ) {
            onNewMessageClick();
        }
    );
    this.message_extend_button.on ( "click", function( p_oEvent ) {
        post_info( "popupdialogs",
            "Sorry, this feature hasn't been implemented yet! "+
            "Come back later when a new version of the application will be available." );
    });
    this.message_submit_button .on ( "click", function( p_oEvent ) {
        post_info( "popupdialogs",
            "Sorry, this feature hasn't been implemented yet! "+
            "Come back later when a new version of the application will be available." );
    });
    return this;
}

function list_shifts() {

    set_context(
        '<a href="javascript:display_experiment()">Experiment Status</a> > '+
        'Select Shift >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea_table",
        [ { key: "begin_time", sortable: true, resizeable: false },
          { key: "end_time",   sortable: true, resizeable: false },
          { key: "leader",     sortable: true, resizeable: false },
          { key: "num_runs",   sortable: true, resizeable: false } ],
        'RequestShifts.php?id='+current_selection.experiment.id,
        false
    );
    table.refreshTable();
}

function select_shift( id ) {
    current_selection.shift.id = id;
    display_shift();
}

function display_shift() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment Status</a> > '+
        'Shift Summary >' );

    document.getElementById('workarea').innerHTML='';
}

function list_runs() {

    set_context(
        '<a href="javascript:display_experiment()">Experiment Status</a> > '+
        'Select Run >' );

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea_table",
        [ { key: "num",              sortable: true, resizeable: false },
          { key: "begin_time",       sortable: true, resizeable: false },
          { key: "end_time",         sortable: true, resizeable: false },
          { key: "shift_begin_time", sortable: true, resizeable: false } ],
        'RequestRuns.php?id='+current_selection.experiment.id,
        false
    );
    table.refreshTable();
}

function select_run( id ) {
    current_selection.run.id = id;
    display_run();
}

function display_run() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment Status</a> > '+
        'Run Summary >' );

    document.getElementById('workarea').innerHTML='';
}

function browse_contents() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment Status</a> > '+
        'Browse >' );

    document.getElementById('workarea').innerHTML='';

    post_info( "popupdialogs",
        "Sorry, this feature hasn't been implemented yet! "+
        "Come back later when a new version of the application will be available." );
}

function search_contents() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment Status</a> > '+
        'Search >' );

    document.getElementById('workarea').innerHTML='';

    post_info( "popupdialogs",
        "Sorry, this feature hasn't been implemented yet! "+
        "Come back later when a new version of the application will be available." );
}
</script>

</head>
<body class="yui-skin-sam" id="body" onload="init()">
    <div id="application_header">
        <p id="application_title">
        <em>Electronic LogBook</em>
        <br>
        <em id="current_selection" style="font-size:32px;">&nbsp;</em></p>
        <p style="text-align:right;">Logged as: <b><?php echo $_SERVER['WEBAUTH_USER']?></b><p>
    </div>
    <div id="menubar" class="yuimenubar yuimenubarnav"></div>
    <p id="context"></p>
    <br>
    <div id="workarea"></div>
    <div id="popupdialogs"></div>
</body>
</html>
