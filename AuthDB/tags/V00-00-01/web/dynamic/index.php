<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title>Authorization Database Manager</title>
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
    a {
        text-decoration:none;
        font-weight:bold;
        color:#0071bc;
        /*
        */
    }
    a:hover {
        color:red;
    }
    p.definitions {
        font-family: Times, serif;
        font-size:18px;
        text-align:left;
    }
    #application_header {
        background-color:#d0d0d0;
        padding:8px;
        padding-bottom:12px;
        margin:0px;
    }
    #application_title {
        margin-left:5px;
        font-family: "Times", serif;
        font-size:32px;
    }
    #application_subtitle {
        color:#0071bc;
        font-size:24px;
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
    #nav-and-work-areas {
        margin-left:20px;
    }
    #navarea {
        overflow:auto;
        padding:20px;
        border-right:solid 4px #f0f0f0;
        display:none;
    }
    #workarea {
        overflow:auto;
        padding:20px;
        /*margin-left:40px;*/
    }
    #experiment_info,
    #instrument_info,
    #runs_info {
        margin-top:0px;
        margin-left:4px;
    }
    #workarea_table_container table,
    #params_table_container   table,
    #runs_table_container     table,
    #role_players_container   table {
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
    #runs_table_container .yui-dt-loading,
    #role_players_container,
    #role_players_container .yui-dt-loading {
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
PHP Generated JavaScript with initialization parameters
-->
<?php

echo <<<HERE
    <script type="text/javascript">

HERE;

echo <<<HERE

/* Authentication and authorization context
 */
var auth_type="{$_SERVER['AUTH_TYPE']}";
var auth_remote_user="{$_SERVER['REMOTE_USER']}";

var auth_webauth_user="{$_SERVER['WEBAUTH_USER']}";
var auth_webauth_token_creation="{$_SERVER['WEBAUTH_TOKEN_CREATION']}";
var auth_webauth_token_expiration="{$_SERVER['WEBAUTH_TOKEN_EXPIRATION']}";

var auth_granted = {
  manage_roles : auth_remote_user != '',
  manage_users : auth_remote_user != '',
  view_ldap    : auth_remote_user != '' };

function refresh_page() {
    window.location = "{$_SERVER['REQUEST_URI']}";
}

HERE;

// Initial action dispatcher's generator
//
echo <<<HERE

function init() {

HERE;
if( isset( $_GET['action'] )) {
    $action = trim( $_GET['action'] );
    if( $action == 'list_roles' ) {
        echo "  list_roles();";
    } else if( $action == 'list_role_players' ) {
        echo "  list_role_players();";
    } else if( $action == 'list_groups' ) {
        echo "  list_groups();";
    } else if( $action == 'list_accounts' ) {
        echo "  list_accounts();";
    } else {
        echo "  alert( 'unsupported action: {$action}' );";
    }
} else {
    echo "  load( 'Welcome.php', 'workarea' );";
}
echo <<<HERE

  auth_timer_restart();

}
</script>

HERE;
?>

<!--
Page-specific script
-->
<script type="text/javascript">

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
        { text: "Authorization Database Manager", url: "../authdb/" },
        { text: "Experiment Registry Database", url: "../regdb/" },
        { text: "Electronic Log Book", url: "../logbook/" } ],
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
var menubar_roles_home = menubar_data.length;
menubar_data.push ( {
    id: "roles",
    href: "#roles",
    title: 'Roles',
    title_style: 'font-weight:bold;',
    itemdata: [
        { text: "Select..", url: "javascript:list_roles()" },
        { text: "Create New..", url: "javascript:create_role()", disabled: !auth_granted.manage_roles } ],
    disabled: false }
);
var menubar_users_home = menubar_data.length;
menubar_data.push ( {
    id: "players",
    href: "#players",
    title: 'Role Players',
    title_style: 'font-weight:bold;',
    itemdata: [
        { text: "Select..", url: "javascript:list_role_players()" },
        { text: "Add New..", url: "javascript:add_role_player()", disabled: !auth_granted.manage_users } ],
    disabled: false }
);
var menubar_groups_home = menubar_data.length;
menubar_data.push ( {
    id: "ldap",
    href: "#ldap",
    title: 'LDAP',
    title_style: 'font-weight:bold;',
    itemdata: [
        { text: "POSIX Groups..", url: "javascript:list_groups()" },
        { text: "User accounts..", url: "javascript:list_accounts()" } ],
    disabled: !auth_granted.view_ldap }
);
var menubar_group_help = menubar_data.length;
menubar_data.push ( {
    id:    'help',
    href:  '#help',
    title: 'Help',
    title_style: null,
    itemdata: [
        { text: "Contents",              url: "#", disabled: true },
        { text: "With the current page", url: "#", disabled: true },
        { text: "About the application", url: "#", disabled: true } ],
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

/*
 * Session expiration timer for WebAuth authentication.
 */
var auth_timer = null;
function auth_timer_restart() {
    if( auth_type == 'WebAuth' )
        auth_timer = window.setTimeout( 'auth_timer_event()', 1000 );
}
var auth_last_secs = null;
function auth_timer_event() {

    var auth_expiration_info = document.getElementById( "auth_expiration_info" );
    var now = mktime();
    var seconds = auth_webauth_token_expiration - now;
    if( seconds <= 0 ) {
        auth_expiration_info.innerHTML=
            '<b><em style="color:red;">expired</em></b>';
        ask_action_confirmation(
            'popupdialogs',
            '<span style="color:red; font-size:16px;">Session Expiration Warning</span>',
            '<p style="text-align:left;">Your WebAuth session has expired. '+
            'Press <b>Ok</b> or use <b>Refresh</b> button of the browser to renew your credentials.</p>',
            refresh_page );
        return;
    }
    var hours_left   = Math.floor(seconds / 3600);
    var minutes_left = Math.floor((seconds % 3600) / 60);
    var seconds_left = Math.floor((seconds % 3600) % 60);

    var hours_left_str = hours_left;
    if( hours_left < 10 ) hours_left_str = '0'+hours_left_str;
    var minutes_left_str = minutes_left;
    if( minutes_left < 10 ) minutes_left_str = '0'+minutes_left_str;
    var seconds_left_str = seconds_left;
    if( seconds_left < 10 ) seconds_left_str = '0'+seconds_left_str;

    auth_expiration_info.innerHTML=
        '<b>'+hours_left_str+':'+minutes_left_str+'.'+seconds_left_str+'</b>';

    auth_timer_restart();
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

function create_button( elementId, func2proceed, disabled ) {
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
    this.oButton.set( 'disabled', disabled );
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

function create_privileges_table_editable( source, paginator ) {

    document.getElementById('privileges').innerHTML=
        '  <div id="privileges_table_paginator"></div>'+
        '  <div id="privileges_table_body"></div>';

    /* Decide on an initial source of the information to populate
     * the table from.
     */
    this.source = source;
    if( null == this.source ) this.storage = [];
    else                      this.storage = null;

    function synchronize_data( predicate ) {
        var rs = this.table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        this.storage = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( predicate( r ))
                this.storage.push ( {
                    'privilege': r.getData('privilege')} );
        }
    }

    function createTable() {
        if( null == this.storage ) {
            return new Table1(
                "privileges",
                [ { key: "selected", formatter: "checkbox" },
                  { key: "privilege", sortable: true, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true}) } ],
                this.source,
                paginator );
        } else {
            return new TableLocal(
                "privileges",
                [ { key: "selected", formatter: "checkbox" },
                  { key: "privilege", sortable: true, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true}) } ],
                this.storage,
                paginator );
        }
    }
    this.table = createTable();

    function AddAndRefreshTable() {
        this.table.dataTable.addRow (
            { 'privilege': "" }, 0 );
    }

    this.oPushButtonAdd = new YAHOO.widget.Button( "add_button" );
    this.oPushButtonAdd.on (
        "click",
        function( p_oEvent ) { AddAndRefreshTable(); }
    );

    function deleteAndRefreshTable() {
        synchronize_data( function( r ) { return !r.getData('selected'); } );
        this.table.dataTable.destroy();
        this.table = createTable();
    }
    this.oPushButtonRemove = new YAHOO.widget.Button( "remove_button" );
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
                result.push ( r.getData('privilege') );
            }
        } else if( this.storage != null) {
            for( var i = 0; i < this.storage.length; i++ ) {
                var r = this.storage[i];
                result.push ( r['privilege'] );
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

function reset_navarea() {
    var navarea = document.getElementById('navarea');
    navarea.style.display = 'none';
    //navarea.innerHTML='';
}

function reset_workarea() {
    var workarea = document.getElementById('workarea');
    //workarea.style.borderLeft='0px';
    //workarea.style.padding = '0px';
    workarea.innerHTML='';
}

/* Roles browser and relevant operations.
 *
 * NOTE: Ideally these would need to be declared as 'const'. Unfortunatelly,
 * ECMAScript won't support this (the code won't work on MS Explorer). Only
 * Mozilla family of browsers will.
 */
var ROLES_APPS   = 1,   // applications
    ROLES        = 2,   // roles
    ROLES_PRIVS  = 3;   // privileges

var roles_tree = null;

var default_application_msg = '&lt;select application&gt;';
var default_role_msg        = '&lt;select role&gt;';

var selected_application = default_application_msg;
var selected_role_name   = default_role_msg;
var selected_role_id     = null;

function list_roles() {

    selected_application = default_application_msg;
    selected_role_name   = default_role_msg;
    selected_role_id     = null;

    set_context(
        'Applications, Roles and Privileges >' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    workarea.innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Definitions.png" />'+
        '</div>'+
        '<div style="margin-bottom:40px; padding-left:20px;">'+
        '  <p class="definitions">'+
        '  The main purpose of <b>Roles</b> in the <b>Authorization Database</b>'+
        '  is to simplify a process of managing <b>Privileges</b> individual users and/or'+
        '  groups of users may have in a context of <b>Applications</b>.'+
        '  Each application has its own set of unique roles and privileges. Each role'+
        '  may have zero or more privileges.'+
        '  A semantic of specifics roles and privileges is beyond a scope of the Authorization Database.'+
        '  It is up to a developer of a particular application to define the semantics and'+
        '  create roles and privileges which would best suit a problem domain of the application.'+
        '  </p>'+
        '  <br>'+
        '  <p class="definitions">'+
        '  The application tree shown on the left of the page allows to explore a hierarchy of roles'+
        '  and privileges for applications which are registed in the database.'+
        '  By clicking on a role one can see a table with a list of <b>Role Players</b> associated with'+
        '  that role accross all known instruments/experiments. Th etable will be created in a separate section below.'+
        '  A role player can be either a single user or a group of users.'+
        '  </p>'+
        '</div>'+
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Operations.png" />'+
        '</div>'+
        '<div style="margin-bottom:40px; padding-left:20px;">'+
        '  <table><thead>'+
        '    <tr>'+
        '      <td>'+
        '        <table style="width:400px;"><thead>'+
        '          <tr>'+
        '            <td>'+
        '              <div id="selected_application" style="margin-left:10px; min-width:75px; margin-right:10px; font-weight:bold;">'+default_application_msg+'</div></td>'+
        '            <td>'+
        '              <button id="delete_application_button" >Delete Appl</button>'+
        '            </td>'+
        '          </tr>'+
        '          <tr>'+
        '            <td><div style="height:10px;"></div></td>'+
        '          </tr>'+
        '          <tr>'+
        '            <td>'+
        '              <div id="selected_role" style="margin-left:10px; min-width:75px; margin-right:10px; font-weight:bold;">'+default_role_msg+'</div></td>'+
        '            <td>'+
        '              <button id="delete_role_button" >Delete Role</button>'+
        '              <button id="edit_role_button" >Edit</button>'+
        '            </td>'+
        '          </tr>'+
        '        </thead></table>'+
        '      </td>'+
        '      <td valign="top">'+
        '        <p class="definitions">'+
        '        This section defined operations on a whole application or a role. Note that deleting'+
        '        is an irreversable operating resulting in a loss of user privileges in the corresponding'+
        '        context!'+
        '        </p>'+
        '      </td>'+
        '    </tr>'+
        '  </thead></table>'+
        '</div>'+
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/RolePlayers.png" />'+
        '</div>'+
        '<div id="role_players" style="padding-left:20px;"></div>';

    var DeleteApplicationButton = new YAHOO.widget.Button( "delete_application_button" );
    DeleteApplicationButton.set( 'disabled', true );
    DeleteApplicationButton.on (
        "click",
        function( p_oEvent ) { delete_application( selected_application ); }
    );
    var DeleteRoleButton = new YAHOO.widget.Button( "delete_role_button" );
    DeleteRoleButton.set( 'disabled', true );
    DeleteRoleButton.on (
        "click",
        function( p_oEvent ) { delete_role( selected_role_id ); }
    );
    var EditRoleButton = new YAHOO.widget.Button( "edit_role_button" );
    EditRoleButton.set( 'disabled', true );
    EditRoleButton.on (
        "click",
        function( p_oEvent ) { alert("not implemented yet"); }
    );

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    navarea.innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Roles.png" />'+
        '</div>'+
        '<div id="roles_tree"></div>';

    roles_tree = new YAHOO.widget.TreeView( "roles_tree" );

    // The whole tree will be built dynamically
    //
    var root_node = new YAHOO.widget.TextNode(
        {   label:    'Applications',
            expanded: false,
            title:    'Expand and select an application to see roles available in its context',
            type:     ROLES_APPS },
        roles_tree.getRoot());

    var currentIconMode = 0;
    root_node.setDynamicLoad( loadNodeData, currentIconMode );
    root_node.toggle();     // Force the node to be instantly open. this will also
                            // trigger the dynamic loading of its children (if any).

    roles_tree.subscribe( "labelClick", onNodeSelection );
    roles_tree.subscribe( "enterKeyPressed", onNodeSelection );
    roles_tree.draw();

    // Dispatch clicks on selected nodes to the corresponding
    // functions.
    //
    function onNodeSelection( node ) {
        var delete_application_disabled = true;
        var delete_role_disabled        = true;
        var edit_role_disabled          = true;
        if( node.data.type == ROLES_APPS) {
            selected_application = default_application_msg;
            selected_role_name   = default_role_msg;
            selected_role_id     = null;
        } else if( node.data.type == ROLES) {
            delete_application_disabled = false;
            selected_application = node.data.application;
            selected_role_name   = default_role_msg;
            selected_role_id     = null;
        } else if( node.data.type == ROLES_PRIVS) {
            delete_application_disabled = false;
            delete_role_disabled        = false;
            edit_role_disabled          = false;
            selected_application = node.data.application;
            selected_role_name   = node.data.name;
            selected_role_id     = node.data.role_id;
            display_role_players( selected_application, selected_role_name, selected_role_id );
        }
        document.getElementById('selected_application').innerHTML = selected_application;
        document.getElementById('selected_role').innerHTML = selected_role_name;
        DeleteApplicationButton.set( 'disabled', delete_application_disabled );
        DeleteRoleButton.set( 'disabled', delete_role_disabled );
        EditRoleButton.set( 'disabled', edit_role_disabled );
    }

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
        var sUrl = "RequestRoles.php?type="+node.data.type;
        switch( node.data.type ) {
            case ROLES_PRIVS:
                sUrl += '&role_id='+node.data.role_id;
            case ROLES:
                sUrl += '&application='+node.data.application;
            case ROLES_APPS:
                break;
        }

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
                            if( tempNode.data.type != undefined )
                                tempNode.setDynamicLoad( loadNodeData, currentIconMode );
                        }
                    } else {
                        // there is only one result; comes as string:
                        //
                        var tempNode = new YAHOO.widget.TextNode( oResults.ResultSet.Result, node, false );
                        if( tempNode.data.type != undefined )
                            tempNode.setDynamicLoad( loadNodeData, currentIconMode );
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

function delete_application( application_name ) {

    var dialog_id = "popupdialogs";
    var dialog_title = '<em style="color:red; font-weight:bold; font-size:18px;">Delete Selected Application?</em>';
    var dialog_body =
        '<div style="text-align:left;">'+
        '  <form  name="delete_application_form" action="ProcessDeleteApplication.php" method="post">'+
        '    <b>PLEASE, READ THIS:</b> the selected application, its roles and connected role players are about to be destroyed!'+
        '    The information may be permanently lost as a result of the operation.'+
        '    Also note that your identity will be recorded.'+
        '    <br><br>'+
        '    Press <b>Yes</b> to delete the application, or press <b>No</b> to abort the operation.'+
        '    <input type="hidden" name="name" value="'+application_name+'" />'+
        '    <input type="hidden" name="actionSuccess" value="list_roles" />'+
        '  </form>'+
        '</div>';
    ask_yesno(
        dialog_id,
        dialog_title,
        dialog_body,
        function() { document.delete_application_form.submit(); },
        function() { }
    );
}

function delete_role( role_id ) {

    var dialog_id = "popupdialogs";
    var dialog_title = '<em style="color:red; font-weight:bold; font-size:18px;">Delete Selected Role?</em>';
    var dialog_body =
        '<div style="text-align:left;">'+
        '  <form  name="delete_role_form" action="ProcessDeleteRole.php" method="post">'+
        '    <b>PLEASE, READ THIS:</b> the selected role and connected role players are about to be destroyed!'+
        '    The information may be permanently lost as a result of the operation.'+
        '    Also note that your identity will be recorded.'+
        '    <br><br>'+
        '    Press <b>Yes</b> to delete the role, or press <b>No</b> to abort the operation.'+
        '    <input type="hidden" name="id" value="'+role_id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="list_roles" />'+
        '  </form>'+
        '</div>';
    ask_yesno(
        dialog_id,
        dialog_title,
        dialog_body,
        function() { document.delete_role_form.submit(); },
        function() { }
    );
}

function display_role_players( application, role_name, role_id ) {

    set_context(
        '<a href="javascript:list_roles()">Applications, Roles and Privileges</a> > '+
        '<i>'+application+'</i> > '+
        '<i>'+role_name+'</i> >' );

    document.getElementById('role_players').innerHTML=
        '<div id="role_players_container">'+
        '  <div id="role_players_table_paginator"></div>'+
        '  <div id="role_players_table_body"></div>'+
        '</div>';

    var table = new Table (
        "role_players",
        [ { key: "instrument", sortable: true, resizeable: true },
          { key: "experiment", sortable: true, resizeable: true },
          { key: "group",      sortable: true, resizeable: true },
          { key: "user",       sortable: true, resizeable: true },
          { key: "comment",    sortable: true, resizeable: true } ],
        'RequestRolePlayers.php?role_id='+role_id,
        false
    );
    //table.refreshTable();
}

function create_role( ) {

    set_context(
        'Create New Role > ' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:10px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="experiment_info" style="height:125px;">'+
        '    <form name="create_role_form" action="ProcessCreateRole.php" method="post">'+
        '      <div id="role_info_within_form"></div>'+
        '      <input type="hidden" name="actionSuccess" value="list_roles" />'+
        '      <input type="hidden" name="privileges" value="" />'+
        '    </form>'+
        '  </div>'+
        '  <br>'+
        '  <div style="padding-left:5px;">'+
        '    <table>'+
        '      <tbody>'+
        '        <tr>'+
        '          <td valign="top">'+
        '            <div style="font-weight:bold; width:100px;">Privileges:</div>'+
        '          </td>'+
        '          <td>'+
        '            <div id="privileges"></div>'+
        '          </td>'+
        '          <td valign="top">'+
        '            <div style="padding-left:10px;">'+
        '              <button id="add_button" >Add</button>'+
        '              <button id="remove_button" >Remove Selected</button>'+
        '            </div>'+
        '          </td>'+
        '        </tr>'+
        '      </tbody>'+
        '    </table>'+
        '  </div>'+
        '</div>';

    load( 'CreateRole.php', 'role_info_within_form' );

    var privileges = create_privileges_table_editable( null, false );

    var save = create_button (
        "save_button",
        function() {
            if( document.create_role_form.application_name.value == '' )
                document.create_role_form.application_name.value = document.create_role_form.application_name_select.value;
            document.create_role_form.privileges.value = privileges.toJSON();
            document.create_role_form.submit();
        },
        !auth_granted.manage_roles
    );
    var cancel = create_button (
        "cancel_button",
        function() { list_roles(); } );
}


/* Role playerss browser.
 */
var PLAYERS_INSTR = 1,  // instruments
    PLAYERS_EXPER = 2,  // experiments
    PLAYERS       = 3;  // role players in a context of an experiment

var players_tree = null;

function list_role_players() {

    set_context(
        'Role Players >' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    workarea.innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Definitions.png" />'+
        '</div>'+
        '<div style="margin-bottom:40px; padding-left:20px;">'+
        '  <p class="definitions">'+
        '  The tree shown on the left of the page allows to select an experiment'+
        '  and privileges for applications which are registed in the database.'+
        '  By clicking on a role one can see a table with a list of <b>Role Players</b> associated with'+
        '  that role accross all known instruments/experiments. Th etable will be created in a separate section below.'+
        '  A role player can be either a single user or a group of users.'+
        '  </p>'+
        '</div>'+
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/RolePlayers.png" />'+
        '</div>'+
        '<div id="role_players_exper" style="padding-left:20px;"></div>';

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    navarea.innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Experiment.png" />'+
        '</div>'+
        '<div id="players_tree"></div>';

    players_tree = new YAHOO.widget.TreeView( "players_tree" );

    // The whole tree will be built dynamically
    //
    var root_node = new YAHOO.widget.TextNode(
        {   label:    'Instruments/Experiments',
            expanded: false,
            title:    'Expand and select an instrument',
            type:     PLAYERS_INSTR },
        players_tree.getRoot());

    var currentIconMode = 0;
    root_node.setDynamicLoad( loadNodeData, currentIconMode );
    root_node.toggle();     // Force the node to be instantly open. this will also
                            // trigger the dynamic loading of its children (if any).

    players_tree.subscribe( "labelClick", onNodeSelection );
    players_tree.subscribe( "enterKeyPressed", onNodeSelection );
    players_tree.draw();

    // Dispatch clicks on selected nodes to the corresponding
    // functions.
    //
    function onNodeSelection( node ) {
        if( node.data.type == PLAYERS ) {
            display_role_players_exper( node.data.instrument, node.data.experiment, node.data.exper_id );
        }
    }

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
        var sUrl = "RequestRolePlayersExper.php?type="+node.data.type;
        switch( node.data.type ) {
            case PLAYERS_EXPER:
                sUrl += '&instr_id='+node.data.instr_id;
            case PLAYERS_INSTR:
                break;
        }

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
                            if( tempNode.data.type != undefined )
                                tempNode.setDynamicLoad( loadNodeData, currentIconMode );
                        }
                    } else {
                        // there is only one result; comes as string:
                        //
                        var tempNode = new YAHOO.widget.TextNode( oResults.ResultSet.Result, node, false );
                        if( tempNode.data.type != undefined )
                            tempNode.setDynamicLoad( loadNodeData, currentIconMode );
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

function display_role_players_exper( instrument, experiment, exper_id ) {

    set_context(
        '<a href="javascript:list_role_players()">Role Players</a> > '+
        '<i>'+instrument+'</i> > '+
        '<i>'+experiment+'</i> >' );

    document.getElementById('role_players_exper').innerHTML=
        '<div id="role_players_exper_container">'+
        '  <div id="role_players_exper_table_paginator"></div>'+
        '  <div id="role_players_exper_table_body"></div>'+
        '</div>';

    var table = new Table (
        "role_players_exper",
        [ { key: "application", sortable: true, resizeable: true },
          { key: "role",        sortable: true, resizeable: true },
          { key: "group",       sortable: true, resizeable: true },
          { key: "user",        sortable: true, resizeable: true },
          { key: "comment",     sortable: true, resizeable: true } ],
        'RequestRolePlayersExper.php?type='+PLAYERS+'&exper_id='+exper_id,
        false
    );
}

function add_role_player( ) {

    set_context(
        'Add Role Player > ' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div id="actions_container">'+
        '  <button id="save_button">Save</button>'+
        '  <button id="cancel_button">Cancel</button>'+
        '</div>'+
        '<div style="margin-top:25px; margin-right:0px; background-color:#f0f0f0; padding-left:25px; padding-right:25px; padding-top:25px; padding-bottom:25px; overflow:auto;">'+
        '  <div id="instrument_info" style="height:150px;">'+
        '    <form name="create_player_form" action="ProcessAddRolePlayer.php" method="post">'+
        '      <div id="player_info_within_form"></div>'+
        '      <input type="hidden" name="actionSuccess" value="list_role_players" />'+
        '    </form>'+
        '  </div>'+
        '</div>';

    load( 'AddRolePlayer.php', 'player_info_within_form' );

    var save = create_button (
        "save_button",
        function() {
            // check if all fileds are non-empty
            //
            if( document.create_player_form.application_name.value == '' ||
                document.create_player_form.role_name.value == '' ||
                ( document.create_player_form.instrument_name.value == '' && document.create_player_form.experiment_name.value != '' ) ||
                ( document.create_player_form.instrument_name.value != '' && document.create_player_form.experiment_name.value == '' ) ||
                ( document.create_player_form.user.value == '' && document.create_player_form.group.value == '' )
            ) {
                alert( "Please,fill in the form as required!" );
                return;
            }
            document.create_player_form.submit();
        },
        !auth_granted.manage_users
    );
    var cancel = create_button (
        "cancel_button",
        function() { list_role_players(); }
    );
}






function list_groups() {

    set_context(
        'Select POSIX Group >' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "group",   sortable: true, resizeable: true } ],
        '../regdb/RequestGroups.php',
        true
    );
    table.refreshTable();
}

function view_group( name ) {

    set_context(
        //'Home > '+
        '<a href="javascript:list_groups()">Select POSIX Group</a> > '+
        '<i>'+name+'</i>' );

    reset_navarea();
    reset_workarea();

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
        '../regdb/RequestGroupMembers.php?name='+name,
        true
    );
    table.refreshTable();
}

function list_accounts() {

    set_context(
        'Select User Account >' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var table = new Table (
        "workarea",
        [ { key: "uid",    sortable: true,  resizeable: true },
          { key: "name",   sortable: true,  resizeable: true },
          { key: "email",  sortable: true,  resizeable: true },
          { key: "groups", sortable: false, resizeable: true } ],
        '../regdb/RequestUserAccounts.php',
        true
    );
    table.refreshTable();
}

</script>

</head>
<body class="yui-skin-sam" id="body" onload="init()">
    <div id="application_header">
      <div>
        <div style="float:left;">
          <p id="application_title" style="text-align:left;">
            <em>Authorization Database Manager: </em>
            <em id="application_subtitle">LCLS Detector Control</em>
          </p>
        </div>
        <div style="float:right; height:50px;">
<?php
$remote_user = $_SERVER['REMOTE_USER'];
if( $remote_user == '' ) echo <<<HERE
          <br>
          <br>
          <a href="../../apps/regdb"><p title="login here to proceed to the full version of the application">login</p></a>
HERE;
else echo <<<HERE
          <table><tbody>
            <tr>
              <td>&nbsp;</td>
              <td></td>
            </tr>
            <tr>
              <td>Welcome,&nbsp;</td>
              <td><p><b>{$remote_user}</b></p></td>
            </tr>
            <tr>
              <td>Session expires in:&nbsp;</td>
              <td><p id="auth_expiration_info"><b>00:00.00</b></p></td>
            </tr>
          </tbody></table>
HERE;
?>
        </div>
        <div style="height:40px;">&nbsp;</div>
      </div>
    </div>
    <div id="menubar" class="yuimenubar yuimenubarnav"></div>
    <p id="context"></p>
    <br>
    <div id="nav-and-work-areas" align="left">
      <table>
        <tbody>
          <tr>
            <td valign="top">
              <div id="navarea"></div>
            </td>
            <td valign="top">
              <div id="workarea"></div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div id="popupdialogs"></div>
<!--
    <div id="workarea"></div>
    <div id="popupdialogs"></div>
-->
</body>
</html>
