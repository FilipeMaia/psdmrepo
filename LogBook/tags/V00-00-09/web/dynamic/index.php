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
    <link rel="stylesheet" type="text/css" href="/yui/build/treeview/assets/skins/sam/treeview.css" />

    <!--bring in the folder-style CSS for the TreeView Control
    <link rel="stylesheet" type="text/css" href="/yui/examples/treeview/assets/css/folders/tree.css" />
    -->
    <!--
    Page-specific styles
    -->
    <style type="text/css">

    /*margin and padding on body element
      can introduce errors in determining
      element position and are not recommended;
      we turn them off as a foundation for YUI
      CSS treatments. */
    body {
        margin:0;
        padding:0;
    }
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
        margin-left:5px;
        font-family: "Times", serif;
        font-size:32px;
    }
    #application_subtitle {
        color:#0071bc;
    }
    #menubar {
        margin: 0 0 10px 0;
    }
    #context {
        margin-top:5px;
        margin-left:5px;
        margin-right:5px;
        margin-bottom:0px;
        /*font-family: "Times", serif;*/
        font-size:16px;
        /*font-weight:bold;*/
        border:solid 4px transparent;
        border-left-width:16px;
        text-align:left;
    }
    #nav-and-work-areas {
        margin-left:35px;
        margin-right:0px;
    }
    #navarea {
        overflow:auto;
    }
    #workarea {
        /*
        overflow:auto;
        */
    }
    #experiment_info_container,
    /*#runs_table,*/
    #run_parameters,
    #messages_actions_container {
        padding:10px;
    }
    #workarea_table_container          table,
    #entry_tags_table_container        table,
    #entry_attachments_table_container table {
    }
    #workarea_table_paginator,
    #params_table_page,
    #runs_table_paginator,
    #shifts_table_paginator,
    #tags_table_paginator,
    #files_table_paginator,
    #entry_tags_table_paginator,
    #entry_attachments_table_paginator {
        margin-left:auto;
        margin-right:auto;
    }
    #messages_table_paginator {
        /*margin-left:0px;
        margin-right:auto;*/
        text-align:center;
    }
    #workarea_table_container,
    #workarea_table_container .yui-dt-loading,
    #entry_tags_table_container,
    #entry_tags_table_container .yui-dt-loading,
    #entry_attachments_table_container,
    #entry_attachments_table_container .yui-dt-loading {
        text-align:center;
        background-color:transparent;
    }
    #actions_container,
    #params_actions_container,
    #runs_actions_container,
    #shifts_actions_container {
        margin-top:18px;
        margin-left:0px;
        text-align:left;
    }
    .lb_label {
        text-align:left;
        /*color:#0071bc;*/
        font-weight:bold;
    }
    .lb_link {
        text-decoration:none;
        color:#0071bc;
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
    <script type="text/javascript" src="/yui/build/treeview/treeview-min.js"></script>

    <!--
    Custom JavaScript
    -->
    <script type="text/javascript" src="Menubar.js"></script>
    <script type="text/javascript" src="Dialogs.js"></script>
    <script type="text/javascript" src="Loader.js"></script>
    <script type="text/javascript" src="JSON.js"></script>
    <script type="text/javascript" src="Utilities.js"></script>

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
    document.getElementById( "application_subtitle" ).innerHTML =
        instr_name+' / '+exper_name;

    menubar_enable( menubar_group_shifts );
    menubar_enable( menubar_group_runs );
    menubar_enable( menubar_group_browse );
    menubar_enable( menubar_group_search );
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
        { text: "Experiment Registry Database", url: "../../RegDB/dynamic/" },
        { text: "Electronic Log Book", url: "../../LogBook/dynamic/" } ],
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
    id:    'browse',
    href:  'javascript:browse_contents()',
    title: 'Browse',
    itemdata: null,
    disabled: true }
);
var menubar_group_search = menubar_data.length;
menubar_data.push ( {
    id:    'search',
    href:  '#find',
    title: 'Find',
    itemdata:  [
        { text: "Find a text in all messages..", url: "javascript:search_contents_simple()" },
        { text: "Advanced find..", url: "javascript:search_contents()" } ],
    disabled: true }
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

    if( $action == 'select_experiment' ) {
        $instr_id   = $_GET['instr_id'];
        $instr_name = $_GET['instr_name'];
        $exper_id   = $_GET['exper_id'];
        $exper_name = $_GET['exper_name'];
        echo "  select_experiment({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}');";

    } else if( $action == 'select_experiment_and_shift' ) {
        $instr_id   = $_GET['instr_id'];
        $instr_name = $_GET['instr_name'];
        $exper_id   = $_GET['exper_id'];
        $exper_name = $_GET['exper_name'];
        $shift_id   = $_GET['shift_id'];
        echo "  select_experiment_and_shift({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}',{$shift_id});";

    } else if( $action == 'select_experiment_and_run' ) {
        $instr_id   = $_GET['instr_id'];
        $instr_name = $_GET['instr_name'];
        $exper_id   = $_GET['exper_id'];
        $exper_name = $_GET['exper_name'];
        $shift_id   = $_GET['shift_id'];
        $run_id     = $_GET['run_id'];
        echo "  select_experiment_and_run({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}',{$shift_id},{$run_id});";

    } else {
        echo "  alert( 'unsupported action: {$action}' );";
    }
} else {
    echo "  load( 'help/Welcome.html', 'workarea' );";
}
echo <<<HERE
}
</script>
HERE;
?>

<script type="text/javascript">

function set_context( context ) {
    document.getElementById('context').innerHTML = context;
}

function Table( itsTableName, itsColumnDefs, itsDataRequest, hasPaginator, rowsPerPage ) {
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
                rowsPerPage: rowsPerPage
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
            {   containers : [this.name+"_paginator"],
                rowsPerPage: 20
            }
        );
    }
    this.dataTable = new YAHOO.widget.DataTable(
        this.name+"_body",
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
        oRecord.setData( "delete", elCheckbox.checked );
    });
    return this;
}

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

function create_runs_table( source, paginator, rows_per_page ) {

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
        paginator,
        rows_per_page
    );
    //table.refreshTable();
}

function create_shifts_table( source, paginator, rows_per_page ) {

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
        paginator,
        rows_per_page
    );
    //table.refreshTable();
}

function create_messages_table( source, paginator, rows_per_page ) {

    document.getElementById('messages_table').innerHTML=
        '  <div id="messages_table_paginator"></div>'+
        '  <div id="messages_table_body"></div>';

    var table = new Table (
        "messages_table",
        [ { key: "posted",      sortable: true,  resizeable: false },
          { key: "author",      sortable: true,  resizeable: false },
          { key: "run",         sortable: true,  resizeable: false },
          { key: "shift",       sortable: true,  resizeable: false },
          { key: "message",     sortable: false, resizeable: true  },
          { key: "tags",        sortable: false, resizeable: true  },
          { key: "attachments", sortable: false, resizeable: true  } ],
        source,
        paginator,
        rows_per_page
    );
    //table.refreshTable();
}

function reset_navarea() {
    var navarea = document.getElementById('navarea');
    navarea.style.padding = '0px';
    navarea.style.minWidth = '0px';
    //navarea.style.width = '0px';
    navarea.innerHTML='';
}

function reset_workarea() {
    var workarea = document.getElementById('workarea');
    workarea.style.borderLeft='0px';
    workarea.style.padding = '0px';
    //workarea.style.minWidth = '0px';
    //workarea.style.width = '0px';
    workarea.innerHTML='';
}

function list_experiments() {

    set_context(
        'Select Experiment >' );

    reset_navarea();
    reset_workarea();

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
        false,
        10
    );
    table.refreshTable();
}

function select_experiment( instr_id, instr_name, exper_id, exper_name ) {
    set_current_selection( instr_id, instr_name, exper_id, exper_name );
    display_experiment();
}

function display_experiment() {

    set_context ( 'Experiment >' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Summary</b></center>'+
        '</div>'+
        '<div id="experiment_info_container" style="height:160px;">Loading...</div>'+
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Operator and Control System Messages</b></center>'+
        '</div>'+
        '<div id="messages_actions_container"></div>';

    load( 'DisplayExperiment.php?id='+current_selection.experiment.id, 'experiment_info_container' );

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
    var messages_dialog = create_messages_dialog( 'experiment' );
}

function create_messages_dialog( scope ) {

    var html =
        '<form enctype="multipart/form-data" name="new_message_form" action="NewFFEntry.php" method="post">'+
        '  <input type="hidden" name="author_account" value="<?php  echo $_SERVER['WEBAUTH_USER'] ?>" style="padding:2px; width:200px;" />'+
        '  <input type="hidden" name="id" value="'+current_selection.experiment.id+'" />'+
        '  <input type="hidden" name="scope" value="'+scope+'" />';
    if( scope == "experiment") {
         html +=
        '  <input type="hidden" name="actionSuccess" value="select_experiment" />';
    } else if( scope == "shift") {
         html +=
        '  <input type="hidden" name="shift_id" value="'+current_selection.shift.id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment_and_shift" />';
    } else if( scope == "run") {
         html +=
        '  <input type="hidden" name="run_id" value="'+current_selection.run.id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment_and_run" />';
    }
    html +=
        '  <input type="hidden" name="MAX_FILE_SIZE" value="1000000">'+
        '  <table></tbody>'+
        '    <tr>'+
        '      <td valign="top">'+
        '        <div id="new_message_body" style="margin-right:10px; background-color:#e0e0e0; padding:4px;">'+
        '          <input id="new_message_text" type="text" name="message_text" size="71" value="" />'+
        '        </div>'+
        '      </td>'+
        '      <td valign="top">'+
        '        <div id="new_message_dialog_container">'+
        '          <button id="message_submit_button">Post</button>'+
        '          <button id="message_extend_button">Options &gt;</button>'+
        '        </div>'+
        '      </td>'+
        '    </tr>'+
        '  </tbody></table>'+
        '</form>'+
        '<div id="messages_table" style="margin-top:10px;"></div>';

    document.getElementById('messages_actions_container').innerHTML=html;

    var url='RequestFFEntries.php?id='+current_selection.experiment.id+'&scope='+scope;
    if( scope == "experiment") {
        ;
    } else if( scope == "shift") {
         url += '&shift_id='+current_selection.shift.id;
    } else if( scope == "run") {
         url += '&run_id='+current_selection.run.id;
    }
    create_messages_table( url, true, 10 );

    this.extendedShown = false;

    this.message_submit_button = new YAHOO.widget.Button( "message_submit_button" );
    this.message_extend_button = new YAHOO.widget.Button( "message_extend_button" );

    this.tags = [];
    this.tags_table = null;

    this.oPushButtonAddTag = null;
    this.oPushButtonRemoveTag = null;

    function synchronize_tags_data() {
        if( this.files_table == null ) return;
        var rs = this.tags_table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        this.tags = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( !r.getData('delete')) {
                this.tags.push ( {
                    tag: r.getData('tag'),
                    value: r.getData('value')});
            } else {
                this.tags_table.dataTable.deleteRow(i);
            }
        }
        // Also refresh the markup with inputs for tags
        //
        var tags = ' <'+'input type="hidden" name="num_tags" value="'+this.tags.length+'" />';;
        for( var i = 0; i < this.tags.length; i++ ) {
            tags += ' <'+'input type="hidden" name="tag_name_'+i+'" value="'+this.tags[i].tag+'" />';
            tags += ' <'+'input type="hidden" name="tag_value_'+i+'" value="'+this.tags[i].value+'" />';
        }
        document.getElementById('message_tags').innerHTML=tags;
    }

    function AddAndRefreshTagsTable() {
        this.tags_table.dataTable.addRow (
            { tag: "", value: "" }, 0 );
    }

    this.files = [];
    this.files_table = null;

    this.oPushButtonAddFile = null;
    this.oPushButtonRemovFile = null;

    function synchronize_files_data() {
        if( this.files_table == null ) return;
        var rs = this.files_table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        this.files = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( !r.getData('delete')) {
                this.files.push ( {
                    file: r.getData('file'),
                    filename: document.getElementById(r.getData('id')).value,
                    description: r.getData('description'),
                    id: r.getData('id')});
            } else {
                this.files_table.dataTable.deleteRow(i);
            }
        }

        // Also refresh the markup with inputs for file descriptions
        //
        var descriptions ='';
        for( var i = 0; i < this.files.length; i++ ) {
            var id = this.files[i].id;
            descriptions += ' <'+'input type="hidden" name="'+id+'" value="'+this.files[i].description+'" />';
        }
        document.getElementById('file_descriptions').innerHTML=descriptions;
    }
    var file2attach_sequence=0;
    function AddAndRefreshFilesTable() {
        file2attach_sequence++;
        id = 'file2attach_'+file2attach_sequence;
        this.files_table.dataTable.addRow (
            { file: '<input type="file" name="'+id+'" id="'+id+'" />', description: "", id: id }, 0 );
    }

    function onExtendedClick() {
        var new_message_body = document.getElementById('new_message_body');
        if( !this.extendedShown ) {
            document.getElementById('new_message_body').innerHTML=
                '<textarea id="new_message_text" type="text" name="message_text"'+
                ' rows="12" cols="68" style="padding:1px;"'+
                ' title="This is multi-line text area in which return will add a new line of text.'+
                ' Use Submit button to post the message.">'+
                document.getElementById('new_message_text').value+'</textarea>'+
                '<div style="margin-left:6px; margin-top:12px;">'+
                '  <em class="lb_label">Author:</em>'+
                '  <input id="author_id" type="text" name="author_name" value="<?php echo $_SERVER['WEBAUTH_USER'] ?>" style="padding:2px; width:200px;" />'+
                '</div>'+
                '<div style="margin-left:6px; margin-right:6px;" align="left">'+
                '  <table style="margin-top:20px;"><tbody>'+
                '    <tr>'+
                '      <td align="left"><em class="lb_label">Tags</em></td>'+
                '      <td><div style="width:8px;"></div></td>'+
                '      <td align="left"><em class="lb_label">Attachments</em></td>'+
                '    </tr>'+
                '    <tr>'+
                '      <td valign="top">'+
                '        <div style="margin-top:8px; padding:8px; background-color:#f0f0f0;" >'+
                '          <div id="tags_table">'+
                '            <div id="tags_table_paginator"></div>'+
                '            <div id="tags_table_body"></div>'+
                '          </div>'+
                '          <div style="margin-top:8px;" align="right" >'+
                '            <button id="add_tag_button">Expand</button>'+
                '            <button id="remove_tag_button">Update</button>'+
                '          </div>'+
                '        </div>'+
                '        <div id="message_tags"></div>'+
                '      </td>'+
                '      <td><div style="width:8px;"></div></td>'+
                '      <td valign="top">'+
                '        <div style="margin-top:8px; padding:8px; background-color:#f0f0f0;" >'+
                '          <div id="files_table">'+
                '            <div id="files_table_paginator"></div>'+
                '            <div id="files_table_body"></div>'+
                '          </div>'+
                '          <div style="margin-top:8px;" align="right" >'+
                '            <button id="add_file_button">Expand</button>'+
                '            <button id="remove_file_button">Update</button>'+
                '          </div>'+
                '        </div>'+
                '        <div id="file_descriptions"></div>'+
                '      </td>'+
                '    </tr>'+
                '  </tbody></table>'+
                '</div>';
            new_message_body.style.paddingBottom="10px";
            this.oPushButtonAddTag = new YAHOO.widget.Button( "add_tag_button" );
            this.oPushButtonAddTag.on (
                "click",
                function( p_oEvent ) { AddAndRefreshTagsTable(); }
            );
            this.oPushButtonRemoveTag = new YAHOO.widget.Button( "remove_tag_button" );
            this.oPushButtonRemoveTag.on (
                "click",
                function( p_oEvent ) { synchronize_tags_data(); }
            );
            this.tags_table = new TableLocal (
                "tags_table",
                [ { key: "delete", formatter: "checkbox" },
                  { key: "tag",   sortable: true, resizeable: false,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true})},
                  { key: "value", sortable: true, resizeable: false,
                    editor: new YAHOO.widget.TextareaCellEditor({disableBtns:true})} ],
                this.tags,
                null
            );

            this.oPushButtonAddFile = new YAHOO.widget.Button( "add_file_button" );
            this.oPushButtonAddFile.on (
                "click",
                function( p_oEvent ) { AddAndRefreshFilesTable(); }
            );
            this.oPushButtonRemoveFile = new YAHOO.widget.Button( "remove_file_button" );
            this.oPushButtonRemoveFile.on (
                "click",
                function( p_oEvent ) { synchronize_files_data(); }
            );
            this.files_table = new TableLocal (
                "files_table",
                [ { key: "delete", formatter: "checkbox" },
                  { key: "file",   sortable: true, resizeable: false },
                  { key: "description", sortable: true, resizeable: false,
                    editor: new YAHOO.widget.TextareaCellEditor({disableBtns:true})},
                  { key: "id", sortable: false, hidden:true }],
                this.tags,
                null
            );

        } else {
            document.getElementById('new_message_body').innerHTML=
                '<input id="new_message_text" type="text" name="message_text" size="71" value="'+
                document.getElementById('new_message_text').value+'" />';
            new_message_body.style.paddingBottom="4px";
            this.tags_table = null;
        }
        this.extendedShown = !this.extendedShown;
        this.tags = [];
        this.files = [];
    }
    this.message_extend_button.on (
        "click",
        function( p_oEvent ) {
            onExtendedClick();
        }
    );
    function SubmitRequest() {
        synchronize_tags_data();
        synchronize_files_data();
        document.forms.new_message_form.submit();
    }
    this.message_submit_button .on (
        "click",
        function( p_oEvent ) {
            SubmitRequest();
        }
    );
    return this;
}

var ffentrypanel=null;

function select_entry( id ) {
    if( ffentrypanel == null) {
        document.getElementById('ffentryheader').innerHTML=
            '<p>Operator Message</p>';
        document.getElementById('ffentryfooter').innerHTML=
            '<p>if the window is closed it will be automatically open with the new message</p>';
        ffentrypanel = new YAHOO.widget.Panel(
            "ffentrydialog", {
                //height:"500px",
                height:"auto",
                width:"auto",
                fixedcenter:false,
                visible:true,
                constrainviewport:false,
                close:true,
                draggable:true
            }
        );
        ffentrypanel.render();
    }
    ffentrypanel.show();
    load( "DisplayFFEntry.php?id="+id, 'ffentrybody');

    create_tags_table (
        'RequestTags.php?id='+id,
        false
    );
    create_attachments_table (
        'RequestAttachments.php?id='+id,
        false
    );
}

function preview_atatchment( id ) {

    var viewarea = document.getElementById('viewarea');
    //viewarea.style.width = '250px';
    //viewarea.style.height = '250px';
    viewarea.innerHTML='<img src="ShowAttachment.php?id='+id+'" width="250" height="250"/>';

/*
    var attachmentWindowRef =
        window.open(
            'ShowAttachment.php?id='+id,
            'attachmentWindow',
            'toolbar=0,location=1,menubar=0,status=0,directories=0');
*/
}

var entry_tags_table=null;

function create_tags_table( source, paginator ) {

    entry_tags_table = new Table (
        "entry_tags_table",
        [ { key: "tag",   sortable: true, resizeable: false },
          { key: "value", sortable: false, resizeable: false } ],
        source,
        paginator,
        10
    );
}

var entry_attachments_table=null;

function create_attachments_table( source, paginator ) {

    entry_attachments_table = new Table (
        "entry_attachments_table",
        [ { key: "attachment", sortable: true, resizeable: false },
          { key: "size",       sortable: true, resizeable: false } ],
        source,
        paginator,
        10
    );
}

function list_shifts() {

    set_context(
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        'Select Shift >' );

    reset_navarea();
    reset_workarea();

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
    //table.refreshTable();
}

function select_shift( shift_id ) {
    current_selection.shift.id = shift_id;
    display_shift();
}

function select_experiment_and_shift( instr_id, instr_name, exper_id, exper_name, shift_id ) {
    set_current_selection( instr_id, instr_name, exper_id, exper_name );
    select_shift( shift_id );
}

function display_shift() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        'Shift >' );

    document.getElementById('workarea').innerHTML=
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Summary</b></center>'+
        '</div>'+
        '<div id="experiment_info_container" style="height:100px;">Loading...</div>'+
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Runs</b></center>'+
        '</div>'+
        '<div id="runs_table" style="padding:10px;"></div>'+
        '<br>'+
        '<br>'+
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Operator and Control System Messages</b></center>'+
        '</div>'+
        '<div id="messages_actions_container"></div>'+
        '<form  name="close_shift_form" action="CloseShift.php" method="post">'+
        '  <input type="hidden" name="id" value="'+current_selection.shift.id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment_and_shift" />'+
        '</form>';

    load( 'DisplayShift.php?id='+current_selection.shift.id, 'experiment_info_container' );

    var runs = create_runs_table (
        'RequestRuns.php?shift_id='+current_selection.shift.id,
        false
    );
    YAHOO.util.Event.onContentReady (
        "close_shift_button",
        function () {
            var close_shift_button = new YAHOO.widget.Button( "close_shift_button" );
            close_shift_button.on (
                "click",
                function( p_oEvent ) {
                    ask_yesno_confirmation (
                        "popupdialogs",
                        'Proceed with the operation and permanently close this shift '+
                        'as of this time? Enter <b>Yes</b> to do so. '+
                        'Note, this is the last chance to abort making modifications '+
                        'in the database!',
                        function() { document.close_shift_form.submit(); },
                        function() { display_shift(); }
                    );
                }
            );
        }
    );
    var messages_dialog = create_messages_dialog( 'shift' );
}

function list_runs() {

    set_context(
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        'Select Run >' );


    reset_navarea();
    reset_workarea();

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

function select_run( shift_id, run_id ) {
    current_selection.shift.id = shift_id;
    current_selection.run.id = run_id;
    display_run();
}

function select_experiment_and_run( instr_id, instr_name, exper_id, exper_name, shift_id, run_id ) {
    set_current_selection( instr_id, instr_name, exper_id, exper_name );
    select_run( shift_id, run_id );
}

function display_run() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        '<a href="javascript:display_shift()">Shift</a> > '+
        'Run >' );

    document.getElementById('workarea').innerHTML=
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Summary</b></center>'+
        '</div>'+
        '<div id="experiment_info_container" style="height:60px;">Loading...</div>'+
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Run Parameters</b></center>'+
        '</div>'+
        '<div id="run_parameters" style="height:370px;">Loading...</div>'+
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Operator and Control System Messages</b></center>'+
        '</div>'+
        '<div id="messages_actions_container"></div>';

    load( 'DisplayRun.php?id='+current_selection.run.id, 'experiment_info_container' );
    load( 'DisplayRunParams.php?id='+current_selection.run.id, 'run_parameters' );

    var messages_dialog = create_messages_dialog( 'run' );
}


/* History browser.
 *
 * NOTE: Ideally these would need to be declared as 'const'. Unfortunatelly,
 * ECMAScript won't support this (the code won't work on MS Explorer). Only
 * Mozilla family of browsers will.
 */
var TYPE_HISTORY_P     = 110,
    TYPE_HISTORY_D     = 120,
    TYPE_HISTORY_D_DAY = 121,
    TYPE_HISTORY_F     = 130,
    TYPE_SHIFTS  = 200,
    TYPE_SHIFT   = 201,
    TYPE_RUNS    = 300,
    TYPE_RUN     = 301;

function container_highlight( which ) {
    which.style.backgroundColor = '#efefef';    // #cfecec';    // 'Light Cyan 2''
}

function container_unhighlight ( which, color ) {
    which.style.backgroundColor = color;
}

function display_history( type, data ) {

    var request_messages_url =
        'Search.php?id='+current_selection.experiment.id+
        '&format=compact'+
        '&search_in_messages=1&search_in_tags=1&search_in_values=1'+
        '&posted_at_experiment=1&posted_at_shifts=1&posted_at_runs=1';

    var request_shifts_url =
        'RequestShifts.php?id='+current_selection.experiment.id;

    var request_runs_url =
        'RequestRuns.php?id='+current_selection.experiment.id;

    var context='';

    switch( type ) {
        case TYPE_HISTORY_P:
            request_messages_url += '&end=b';
            request_shifts_url += '&end=b';
            request_runs_url += '&end=b';
            context += 'Preparation >';
            break;
        case TYPE_HISTORY_D_DAY:
            request_messages_url += '&begin='+encodeURIComponent(data.begin)+'&end='+encodeURIComponent(data.end);
            request_shifts_url += '&begin='+encodeURIComponent(data.begin)+'&end='+encodeURIComponent(data.end);
            request_runs_url += '&begin='+encodeURIComponent(data.begin)+'&end='+encodeURIComponent(data.end);
            context += 'Data Taking > '+data.day;
            break;
        case TYPE_HISTORY_F:
            request_messages_url += '&begin=e';
            request_shifts_url += '&begin=e';
            request_runs_url += '&begin=e';
            context += 'Follow Up >';
            break;
        default:
            alert( "unsupported history element" );
            return;
    }
    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        '<a href="javascript:browse_contents()">Browse</a> > '+
        'History > '+context );

    var subheader_style = 'padding:2px; background-color:#e0e0e0;';

    document.getElementById('workarea').innerHTML =
        '<div style="margin-bottom:20px; '+subheader_style+'">'+
        '  <b>Shifts</b>'+
        '</div>'+
        '<div id="shifts_table" style="margin-left:10px; margin-bottom:20px;"></div>'+
        '<div style="margin-bottom:20px; '+subheader_style+'">'+
        '  <b>Runs</b>'+
        '</div>'+
        '<div id="runs_table" style="margin-left:10px; margin-bottom:20px;"></div>'+
        '<div style="margin-bottom:20px; '+subheader_style+'">'+
        '  <b>Operator and Control System Messages</b>'+
        '</div>'+
        '<div id="messages_table" style="margin-left:10px;"></div>';

    // Build  YUI tables and use their loading mechanism
    //
    create_shifts_table( request_shifts_url, false );
    create_runs_table( request_runs_url, false );
    create_messages_table( request_messages_url, false );
}

var browse_tree = null;

function browse_contents() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        'Browse By Categories >' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    workarea.style.borderLeft="solid 1px";
    workarea.style.padding = "10px";
    workarea.innerHTML='No context selected yet.';

    var navarea = document.getElementById('navarea');
    navarea.style.minWidth = "200px";
    navarea.style.padding = "10px";
    navarea.innerHTML=
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Select Context</b></center>'+
        '</div>'+
        '<div id="browse_tree">Loading...</div>';

    browse_tree = new YAHOO.widget.TreeView( "browse_tree" );

    // Start build a tree from the current context
    //
    var node_i = new YAHOO.widget.TextNode(
        {   label: '<b>'+current_selection.instrument.name+'</b>',
            expanded: true,
            title: 'This is currently selected instrument' },
        browse_tree.getRoot());

    var node_e = new YAHOO.widget.TextNode(
        {   label: '<b>'+current_selection.experiment.name+'</b>',
            expanded: true,
            title: 'This is currently selected experiment' },
        node_i );

    var node_h = new YAHOO.widget.TextNode(
        {   label: 'History',
            expanded: true,
            title: 'Explore the history of various events happened in the experiment.' },
        node_e );

    var node_h_p = new YAHOO.widget.TextNode(
        {   label: 'Preparation',
            expanded: false,
            title: 'The preparation phase of the experiment',
            type: TYPE_HISTORY_P },
        node_h );

    var node_h_d = new YAHOO.widget.TextNode(
        {   label: 'Data Taking',
            expanded: false,
            title: 'The data taking phase of the experiment',
            type: TYPE_HISTORY_D },
        node_h );

    var node_h_f = new YAHOO.widget.TextNode(
        {   label: 'Follow up',
            expanded: false,
            title: 'After the data taking phase was over',
            type: TYPE_HISTORY_F },
        node_h );

    var node_s = new YAHOO.widget.TextNode(
        {   label: 'Shifts',
            expanded: false,
            title: 'Explore shifts of the experiment',
            type: TYPE_SHIFTS },
        node_e );

    var node_r = new YAHOO.widget.TextNode(
        {   label: 'Runs',
            expanded: false,
            title: 'Explore data taking runs',
            type: TYPE_RUNS },
        node_e );

    // turn dynamic loading on for the last children:
    //
    var currentIconMode = 0;

    node_h_d.setDynamicLoad( loadNodeData, currentIconMode );
    node_s.setDynamicLoad  ( loadNodeData, currentIconMode );
    node_r.setDynamicLoad  ( loadNodeData, currentIconMode );

    // Dispatch clicks on selected nodes to the corresponding
    // functions.
    //
    function onNodeSelection( node ) {
        switch( node.data.type ) {
            case TYPE_HISTORY_P:     display_history( TYPE_HISTORY_P,     null); break;
            case TYPE_HISTORY_D_DAY: display_history( TYPE_HISTORY_D_DAY, node.data ); break;
            case TYPE_HISTORY_F:     display_history( TYPE_HISTORY_F,     null); break;

            case TYPE_SHIFTS:
                set_context (
                    '<a href="javascript:display_experiment()">Experiment</a> > '+
                    '<a href="javascript:browse_contents()">Browse</a> > '+
                    'Shifts > ' );
                break;
            case TYPE_RUNS:
                set_context (
                    '<a href="javascript:display_experiment()">Experiment</a> > '+
                    '<a href="javascript:browse_contents()">Browse</a> > '+
                    'Runs > ' );
                break;

            case TYPE_SHIFT: select_shift( node.data.shift_id ); break;
            case TYPE_RUN:   select_run  ( node.data.shift_id, node.data.run_id ); break;
        }
    }
    browse_tree.subscribe( "labelClick", onNodeSelection );
    browse_tree.subscribe( "enterKeyPressed", onNodeSelection );
    browse_tree.draw();

    function loadNodeData( node, fnLoadComplete ) {

        //We'll create child nodes based on what we get back when we
        //use Connection Manager to pass the text label of the
        //expanding node to the Yahoo!
        //Search "related suggestions" API.  Here, we're at the
        //first part of the request -- we'll make the request to the
        //server.  In our Connection Manager success handler, we'll build our new children
        //and then return fnLoadComplete back to the tree.

        //Get the node's label and urlencode it; this is the word/s
        //on which we'll search for related words:
        //
        // alert( "node: "+node.label+", type: "+node.data.type );
        // var nodeLabel = encodeURI( node.data.label );

        //prepare URL for XHR request:
        //
        var sUrl = "RequestInfo.php?type="+node.data.type;
        switch( node.data.type ) {
            case TYPE_HISTORY_P:
            case TYPE_HISTORY_D:
            case TYPE_HISTORY_F:
            case TYPE_SHIFTS:
            case TYPE_RUNS:
                sUrl += '&exper_id='+current_selection.experiment.id;
                break;
        }
        // alert(sUrl);

        //prepare our callback object
        //
        var callback = {

            //if our XHR call is successful, we want to make use
            //of the returned data and create child nodes.
            //
            success: function(oResponse) {
                var oResults = eval( "(" + oResponse.responseText + ")" );
                if(( oResults.ResultSet.Result ) && ( oResults.ResultSet.Result.length )) {

                    // Result is an array if more than one result, string otherwise
                    //
                    if( YAHOO.lang.isArray( oResults.ResultSet.Result )) {
                        for( var i = 0, j = oResults.ResultSet.Result.length; i < j; i++ ) {
                            var tempNode = new YAHOO.widget.TextNode( oResults.ResultSet.Result[i], node, false );
                        }
                    } else {
                        // there is only one result; comes as string:
                        //
                        var tempNode = new YAHOO.widget.TextNode( oResults.ResultSet.Result, node, false )
                    }
                }

                //When we're done creating child nodes, we execute the node's
                //loadComplete callback method which comes in via the argument
                //in the response object (we could also access it at node.loadComplete,
                //if necessary):
                //
                oResponse.argument.fnLoadComplete();
            },

            //if our XHR call is not successful, we want to
            //fire the TreeView callback and let the Tree
            //proceed with its business.
            //
            failure: function(oResponse) {
                alert( "failed to get the information from server for node: "+node.label+", type: "+node.data.type );
                oResponse.argument.fnLoadComplete();
            },

            //our handlers for the XHR response will need the same
            //argument information we got to loadNodeData, so
            //we'll pass those along:
            //
            argument: {
                "node": node,
                "fnLoadComplete": fnLoadComplete
            },

            //timeout -- if more than 7 seconds go by, we'll abort
            //the transaction and assume there are no children:
            //
            timeout: 7000
        };

        //With our callback object ready, it's now time to
        //make our XHR call using Connection Manager's
        //asyncRequest method:
        //
        YAHOO.util.Connect.asyncRequest( 'GET', sUrl, callback );
    }
}

function expander_highlight( which ) {
    which.style.backgroundColor = '#cfecec';    // #cfecec';    // 'Light Cyan 2''
}
function expander_unhighlight( which, color ) {
    which.style.backgroundColor = color;
}

function expander_toggle( hid, bid ) {
    var h = document.getElementById(hid);
    var b = document.getElementById(bid);
    if( b.style.display == 'none') {
        h.innerHTML='<b>V</b>';
        b.style.display = 'block';
    } else {
        h.innerHTML='<b>&gt;</b>';
        b.style.display = 'none';
    }
    h.style.width='14px';
}

var message_ids=[];
var expand_messages = false;
function expand_collapse_all_messages( expand ) {
    expand_messages = expand;
    for( var i=0; i<message_ids.length; i++ ) {

        var h = document.getElementById( message_ids[i].hid);
        var b = document.getElementById( message_ids[i].bid);
        if(  expand_messages ) {
            h.innerHTML='<b>V</b>';
            b.style.display = 'block';
        } else {
            h.innerHTML='<b>&gt;</b>';
            b.style.display = 'none';
        }
        h.style.width='14px';
    }
}

var attachment_ids=[];
var show_attachments = false;
function show_hide_all_attachments( show ) {
    show_attachments = show;
    for( var i=0; i<attachment_ids.length; i++ ) {

        var h = document.getElementById( attachment_ids[i].ahid);
        var a = document.getElementById( attachment_ids[i].aid);
        if(  show_attachments ) {
            h.innerHTML='<b>V</b>';
            a.style.display = 'block';
        } else {
            h.innerHTML='<b>&gt;</b>';
            a.style.display = 'none';
        }
        h.style.width='14px';
    }
}

function event2html_1( re ) {
    // Initialize and register identifiers for group operations like hiding/expanding
    // all messages.
    //
    var hid = 'message_header_'+re.id;
    var bid = 'message_body_'+re.id;
    message_ids.push( { 'hid' : hid, 'bid' : bid } );

    var run = re.run == '' ? '' : 'run: <em style="padding:2px;">'+re.run+'</em>';
    var shift = re.shift == '' ? '' : 'shift: <em style="padding:2px;">'+re.shift+'</em>';
    var run_shift = re.run == '' && re.shift == '' ? '' : ' - '+run+' '+shift;

    var show_attachments_str = show_attachments ? '' : 'display:none;';

    var attachments='';
    for( var i=0; i < re.attachments.length; i++ ) {
        var a = re.attachments[i];

        var ahid = 'attachment_header_id_'+a.id;
        var aid = 'attachment_id_'+a.id;
        attachment_ids.push( { 'ahid' : ahid, 'aid' : aid } );

        attachments +=
            '<div style="margin-top:10px; ">'+
            '  <span id="'+ahid+'" style="border:solid 1px #c0c0c0; width:14px; height:14px; padding-left:2px; margin-right:12px; font-size:14px; text-align:center; cursor:pointer;"'+
            '     onclick="javascript:expander_toggle('+"'"+ahid+"','"+aid+"')"+'"'+
            '     onmouseover="javascript:expander_highlight(this)" '+
            '     onmouseout="javascript:expander_unhighlight(this,document.bgColor)">'+
            '     <b>&gt;</b>'+'</span>'+
            '  <span><b>Attached:</b> '+a.description+'</span>'+
            '</div>'+
            '<div id="'+aid+'" style="margin-left:8px; padding:17px; border-left:solid 2px #efefef; '+show_attachments_str+'">';
        if( a.type == 'image' ) {
            attachments +=
            '    <img max-width="800" src="ShowAttachment.php?id='+a.id+'" />';
        } else if( a.type == 'text' ) {
            var aid4text = 'attachment_id_'+a.id+'_txt';
            attachments +=
            '    <div id="'+aid4text+'" style="max-width:800px;"></div>';
            load( 'ShowAttachment.php?id='+a.id, aid4text );
        } else if( a.type == 'pdf' ) {
            attachments +=
            '    <object data="ShowAttachment.php?id='+a.id+'" type="application/pdf" width="800" height="600"></object>';
        }
        attachments +=
            '</div>';
    }

    var expand_messages_str = expand_messages ? '' : 'display:none;';

    var attachments_sign = '';
    for( var i=0; i < re.attachments.length; i++ )
        attachments_sign += '<img src="images/attachment.png" />';

    var result =
        '<div style="position:relative; left:0px; margin-top:10px; margin-left:10px; padding:2px;">'+
        '  <span id="'+hid+'" style="border:solid 1px #c0c0c0; width:14px; height:14px; padding-left:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '     onclick="javascript:expander_toggle('+"'"+hid+"','"+bid+"')"+'"'+
        '     onmouseover="javascript:expander_highlight(this)" '+
        '     onmouseout="javascript:expander_unhighlight(this,document.bgColor)">'+
        '     <b>&gt;</b>'+'</span>'+
        '  <span>'+
        '    <b><em style="padding:2px;">'+re.event_time+'</em></b>'+
        '    by: <b><em style="padding:2px;">'+re.author+'</em></b>'+
        '    - <em style="padding:2px; color:blue;"><i>'+re.subject+'</i></em>'+run_shift+
        '    '+attachments_sign+'</span>'+
        '</div>'+
        '<div id="'+bid+'" style="'+expand_messages_str+' margin-left:17px; margin-bottom:20px;">'+
        '  <div style="padding:10px; padding-left:20px; border-left:solid 1px #c0c0c0;">'+
        '    <div style="margin-top:8px; margin-right:0px; margin-bottom:20px; background-color:#efefef;">'+re.html+'</div>'+
        attachments+
        '  </div>'+
        '</div>';
    return result;
}

var last_search_result = null;
var limit_per_page = null;
var first_message_idx = null;
var last_message_idx = null;

function display_messages_page() {

    message_ids = [];
    attachment_ids = [];

    document.getElementById('messages_showing_id').innerHTML=(1+first_message_idx)+' through '+last_message_idx;

    prev_message_page_button.set( 'disabled', first_message_idx == 0 );
    next_message_page_button.set( 'disabled', last_message_idx >= last_search_result.length );

    var html = '';
    for( var i = first_message_idx; i < last_message_idx; i++ ) {
        var re = last_search_result[i];
        var t = event2html_1( re );
        html += t;
    }
    document.getElementById('messages_area').innerHTML = html;
}

var prev_message_page_button = null;
function prev_message_page() {
    last_message_idx  = first_message_idx;
    first_message_idx = Math.max( 0, first_message_idx - limit_per_page );
    display_messages_page();
}

var next_message_page_button = null;
function next_message_page() {
    first_message_idx = last_message_idx;
    last_message_idx  = Math.min( last_search_result.length, last_message_idx + limit_per_page );
    display_messages_page();
}

function display_messages() {
    if( last_search_result == null ) {
        alert( 'display_messages(): application logic error - no messages downloaded' );
        return;
    }
    var limit_per_page_str = document.search_display_form.limit_per_page.options[document.search_display_form.limit_per_page.selectedIndex].value;
    limit_per_page = limit_per_page_str == 'all' ? last_search_result.length : Number( limit_per_page_str );

    first_message_idx = 0;
    last_message_idx  = Math.min( last_search_result.length, limit_per_page );

    display_messages_page();
}

function search_and_display() {

    var tag = document.search_form.tag.options[document.search_form.tag.selectedIndex].value;
    var author = document.search_form.author.options[document.search_form.author.selectedIndex].value;

    var url='Search.php?id='+current_selection.experiment.id+
        '&format=detailed'+
        '&text2search='+encodeURIComponent(document.search_form.text2search.value)+
        '&search_in_messages='+(document.search_form.search_in_messages.checked ? '1' : '0')+
        '&search_in_tags='+(document.search_form.search_in_tags.checked ? '1' : '0')+
        '&search_in_values='+(document.search_form.search_in_values.checked ? '1' : '0')+
        '&posted_at_experiment='+(document.search_form.posted_at_experiment.checked ? '1' : '0')+
        '&posted_at_shifts='+(document.search_form.posted_at_shifts.checked ? '1' : '0')+
        '&posted_at_runs='+(document.search_form.posted_at_runs.checked ? '1' : '0')+
        '&begin='+encodeURIComponent(document.search_form.begin.value)+
        '&end='+encodeURIComponent(document.search_form.end.value)+
        '&tag='+encodeURIComponent(tag)+
        '&author='+encodeURIComponent(author);

    document.getElementById('messagesarea').innerHTML='<img src="images/ajaxloader.gif" />&nbsp;searching...';

    function callback_on_load( result ) {
        if( result.ResultSet.Status != "success" ) {
            document.getElementById('messagesarea').innerHTML = result.ResultSet.Message;
        } else {
            last_search_result = result.ResultSet.Result;
            expand_messages = false;
            show_attachments = false;
            document.getElementById('messagesarea').innerHTML =
                '<div style="margin-bottom:20px; padding:2px; padding-left:4px; background-color:#e0e0e0;">'+
                '  <b>'+last_search_result.length+' message(s) found, showing: <em id="messages_showing_id"></em></b>'+
                '</div>'+
                '<div style="margin-left:10px; margin-top:20px; margin-bottom:10px; padding-bottom:10px; border-bottom:solid 2px #e0e0e0;">'+
                '  <form name="search_display_form">'+
                '    <table><tbody>'+
                '      <tr>'+
                '        <td>'+
                '          <button id="expand_button">Expand All</button>'+
                '          <button id="collapse_button">Collapse All</button>'+
                '        </td>'+
                '        <td>'+
                '          <div style="padding-left:40px;">'+
                '            <button id="show_attachments_button">View Attachments</button>'+
                '            <button id="hide_attachments_button">Hide Attachments</button>'+
                '          </div>'+
                '        </td>'+
                '        <td>'+
                '          <div style="padding-left:40px;">'+
                '            <select align="center" type="text" name="limit_per_page" style="padding:1px;" onchange="display_messages()">'+
                '              <option value="5">5</option>'+
                '              <option value="10">10</option>'+
                '              <option value="20">20</option>'+
                '              <option value="50">50</option>'+
                '              <option value="100">100</option>'+
                '              <option value="all">no limit</option>'+
                '            </select> messages on page'+
                '          </div>'+
                '        </td>'+
                '        <td>'+
                '          <div style="padding-left:20px;">'+
                '            <button id="prev_message_page_button">&lt; Prev</button>'+
                '            <button id="next_message_page_button">Next &gt;</button>'+
                '          </div>'+
                '        </td>'+
                '      </tr>'+
                '    </tbody></table></center>'+
                '  </form>'+
                '</div>'+
                '<div id="messages_area"></div>';

            var expand_button = new YAHOO.widget.Button( "expand_button" );
            expand_button.on (
                "click",
                function( p_oEvent ) {
                    expand_collapse_all_messages( true );
                }
            );
            var collapse_button = new YAHOO.widget.Button( "collapse_button" );
            collapse_button.on (
                "click",
                function( p_oEvent ) {
                    expand_collapse_all_messages( false );
                }
            );
            var show_attachments_button = new YAHOO.widget.Button( "show_attachments_button" );
            show_attachments_button.on (
                "click",
                function( p_oEvent ) {
                    show_hide_all_attachments( true );
                }
            );
            var hide_attachments_button = new YAHOO.widget.Button( "hide_attachments_button" );
            hide_attachments_button.on (
                "click",
                function( p_oEvent ) {
                    show_hide_all_attachments( false );
                }
            );
            prev_message_page_button = new YAHOO.widget.Button( "prev_message_page_button" );
            prev_message_page_button.on (
                "click",
                function( p_oEvent ) {
                    prev_message_page();
                }
            );
            next_message_page_button = new YAHOO.widget.Button( "next_message_page_button" );
            next_message_page_button.on (
                "click",
                function( p_oEvent ) {
                    next_message_page();
                }
            );
            display_messages();
        }
    }
    function callback_on_failure( http_status ) {
        document.getElementById('messagesarea').innerHTML=
            '<b><em style="color:red;" >Error</em></b>&nbsp;Request failed. HTTP status: '+http_status;
    }
    load_then_call( url, callback_on_load, callback_on_failure );
}

function search_contents() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        'Find Messages >' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    //workarea.style.borderLeft="solid 1px";
    workarea.style.borderLeft="solid 6px #e0e0e0";
    workarea.style.padding = "10px";
    workarea.style.minHeight="620px";
    workarea.innerHTML=
        '<div id="messagesarea">'+
        '  <div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '    <center><b>How to use the tool</b></center>'+
        '  </div>'+
        '  <div id="search_instructions" style="margin:10px;"></div>'+
        '</div>';

    var navarea = document.getElementById('navarea');
    navarea.style.minWidth = "200px";
    navarea.style.padding = "10px";
    //navarea.style.backgroundColor = "#c0c0c0";
    navarea.innerHTML=
        '<div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '  <center><b>Filter</b></center>'+
        '</div>'+
        '<form name="search_form" action="javascript:search_and_display()">'+
        '  <div id="search_form_params" style="margin:10px;"></div>'+
        '</form>';

    load( 'SearchFormParams.php?id='+current_selection.experiment.id, 'search_form_params' );
    load( 'help/SearchForm.html', 'search_instructions' );

    YAHOO.util.Event.onContentReady (
        "reset_form_button",
        function () {
            var reset_from_button = new YAHOO.widget.Button( "reset_form_button" );
            reset_from_button.on (
                "click",
                function( p_oEvent ) {
                    document.search_form.text2search.value='';
                    document.search_form.search_in_messages.checked=true;
                    document.search_form.search_in_tags.checked=true;
                    document.search_form.search_in_values.checked=true;
                    document.search_form.posted_at_experiment.checked=true;
                    document.search_form.posted_at_shifts.checked=true;
                    document.search_form.posted_at_runs.checked=true;
                    document.search_form.begin.value='';
                    document.search_form.end.value='';
                    document.search_form.tag.value='';
                    document.getElementById('tag_default').selected="selected";
                    document.search_form.author.value='';
                    document.getElementById('author_default').selected="selected";
                }
            );
        }
    );
    YAHOO.util.Event.onContentReady (
        "submit_search_button",
        function () {
            var submit_search_button = new YAHOO.widget.Button( "submit_search_button" );
            submit_search_button.on (
                "click",
                search_and_display
            );
        }
    );
}
    </script>
  </head>
  <body class="yui-skin-sam" id="body" onload="init()">
    <div id="application_header">
      <p id="application_title" style="text-align:left;">
        <em>Electronic LogBook of Experiment: </em>
        <em id="application_subtitle" style="font-size:24px;"><a href="javascript:list_experiments()">select &gt;</a></em>
      </p>
      <p style="text-align:right;">Logged as: <b><?php echo $_SERVER['WEBAUTH_USER']?></b><p>
    </div>
    <div id="menubar" class="yuimenubar yuimenubarnav"></div>
    <p id="context"></p>
    <br>
    <div id="nav-and-work-areas" align="left">
      <table>
        <tbody>
          <tr>
            <td valign="top">
              <!--
              Optional navigation tools (menus, trees, etc.) can be placed
              below.
              -->
              <div id="navarea"></div>
            </td>
            <td valign="top">
              <!--
              Invisible placeholder for free-form entries viewing dialog
              -->
              <div id="ffentrydialog" style="height:1px;">
                <div class="hd" id="ffentryheader"></div>
                <div class="bd">
                  <div id="ffentrybody"></div>
                  <table>
                    <tbody>
                      <tr>
                        <td valign="top">
                          <div id="entry_tags_table_container" style="width:250px;">
                            <div id="entry_tags_table_paginator"></div>
                            <div id="entry_tags_table_body"></div>
                          </div>
                        </td>
                        <td valign="top">
                          <div id="entry_attachments_table_container">
                            <div id="entry_attachments_table_paginator"></div>
                            <div id="entry_attachments_table_body"></div>
                          </div>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div class="ft" id="ffentryfooter"></div>
              </div>
              <!--
              Here comes the main viwing area of the application.
              -->
              <div id="workarea"></div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div id="popupdialogs"></div>
  </body>
</html>
