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
        padding:15px;
        background-color:#f0f0f0;
        */
        overflow:auto;
    }
    #experiment_info_container,
    #runs_table,
    #run_parameters,
    #messages_actions_container {
        padding:10px;
    }
    #workarea_table_container          table,
    #params_table_container            table,
    /*
    #runs_table_container              table,
    */
    #shifts_table_container            table,
    #messages_table_container          table,
    #tags_table_container              table,
    #files_table_container             table,
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
    #params_table_container,
    #params_table_container .yui-dt-loading,
    /*
    #runs_table_container,
    #runs_table_container .yui-dt-loading,
    */
    #shifts_table_container,
    #shifts_table_container .yui-dt-loading,
    #messages_table_container,
    #messages_table_container .yui-dt-loading,
    #tags_table_container,
    #tags_table_container .yui-dt-loading,
    #files_table_container,
    #files_table_container .yui-dt-loading,
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
        paginator,
        10
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
        paginator,
        10
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
        '<div id="messages_table" style="margin-top:10px;">'+
        '  <div id="messages_table_body"></div>'+
        '  <div id="messages_table_paginator"></div>'+
//        '  <div id="messages_table_paginator" style="background-color:#e0e0e0;"></div>'+
        '</div>';
    document.getElementById('messages_actions_container').innerHTML=html;

    var url='RequestFFEntries.php?id='+current_selection.experiment.id+'&scope='+scope;
    if( scope == "experiment") {
        ;
    } else if( scope == "shift") {
         url += '&shift_id='+current_selection.shift.id;
    } else if( scope == "run") {
         url += '&run_id='+current_selection.run.id;
    }

    this.table = new Table (
        "messages_table",
        [ { key: "posted",      sortable: true, resizeable: false },
          { key: "author",      sortable: true, resizeable: false },
          { key: "message",     sortable: true, resizeable: true  },
          { key: "tags",        sortable: true, resizeable: false },
          { key: "attachments", sortable: true, resizeable: false } ],
        url,
        true,
        10
    );

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
        new_message_body = document.getElementById('new_message_body');
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
    );}

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
        '<div id="runs_table"></div>'+
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

function browse_contents() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
        'Browse >' );

    document.getElementById('workarea').innerHTML='';

    post_info( "popupdialogs",
        "Sorry, this feature hasn't been implemented yet! "+
        "Come back later when a new version of the application will be available." );
}

function search_contents() {

    set_context (
        '<a href="javascript:display_experiment()">Experiment</a> > '+
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
