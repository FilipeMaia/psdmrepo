<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title>Electronic LogBook:</title>
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

    /* margin and padding on body element can introduce errors in determining
       element position and are not recommended;
       we turn them off as a foundation for YUI CSS treatments.
     */
    body {
        margin: 0;
        padding: 0;
        /*
         background-color:#e0e0e0;
         background:url('images/paperbg.gif');
         */
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
    
    .lb_label {
        text-align: left;
        /*color:#0071bc;*/
        font-weight: bold;
    }
    
    a,.lb_link {
        text-decoration: none;
        font-weight: bold;
        color: #0071bc;
    }
    
    a:hover,a.lb_link:hover {
        color: red;
    }
    
    .lb_msg_subject {
        color: maroon;
        /*font-style:italic;*/
    }
    
    input.lb_sparse {
        padding: 2px;
        color: blue;
    }
    
    div.help_area {
        width: 800px;
    }
    
    p.help {
        font-family: Times, serif;
        font-size: 18px;
        padding-bottom: 1em;
    }
    
    #application_header {
        background-color: #d0d0d0;
        padding: 8px;
        padding-bottom: 12px;
        margin: 0px;
    }
    
    #application_title {
        margin-left: 5px;
        font-family: "Times", serif;
        font-size: 32px;
    }
    
    #application_subtitle {
        color: #0071bc;
        font-size: 24px;
    }
    
    #menubar {
        margin: 0 0 10px 0;
    }
    
    #context {
        margin: 20px;
        margin-bottom: 0px;
        font-size: 16px;
        text-align: left;
    }
    
    #nav-and-work-areas {
        margin-left: 20px;
    }
    
    #navarea {
        overflow: auto;
        padding: 20px;
        border-right: solid 4px #f0f0f0;
        display: none;
    }
    
    #workarea {
        overflow: auto;
        padding: 20px;
        /*background-color:#E3F6CE;*/ /*background-color:#f0f0f0;*/
    }
    
    #experiment_info_container,
    #run_parameters {
        padding: 10px;
        margin-left: 10px;
    }
    
    #workarea_table_container table {
        
    }
    
    #workarea_table_paginator,
    #params_table_page,
    #runs_table_paginator,
    #shifts_table_paginator,
    #tags_table_paginator,
    #files_table_paginator
    {
        margin-left: auto;
        margin-right: auto;
    }
    
    #workarea_table_container,
    #workarea_table_container .yui-dt-loading {
        text-align: center;
        background-color: transparent;
    }
    
    #actions_container,
    #params_actions_container,
    #runs_actions_container,
    #shifts_actions_container
        {
        margin-top: 18px;
        margin-left: 0px;
        text-align: left;
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
<script type="text/javascript">

<?php

require_once('LogBook/LogBook.inc.php');

try {
    $auth_svc = LogBookAuth::instance();

    $logbook = new LogBook();
    $logbook->begin();

    $instruments = $logbook->instruments();
    $experiments = $logbook->experiments();

    echo <<<HERE

var instruments = [
HERE;
    $first = true;
    foreach( $instruments as $i ) {
        $is_location_str = $i->is_location() ? 'true' : 'false';
        if( $first ) {
            $first = false;
            echo <<<HERE

   {name: '{$i->name()}', id: {$i->id()}, is_location: {$is_location_str}}
HERE;
        } else {
            echo <<<HERE

  ,{name: '{$i->name()}', id: {$i->id()}, is_location: {$is_location_str}}
HERE;
        }
    }
    echo <<<HERE

];

var experiments = [
HERE;
    $first = true;
    foreach( $experiments as $e ) {
        $is_facility_str = $e->is_facility() ? 'true' : 'false';
        if( $first ) {
            $first = false;
            echo <<<HERE

   {name: '{$e->name()}', id: {$e->id()}, is_facility: {$is_facility_str}}
HERE;
        } else {
            echo <<<HERE

  ,{name: '{$e->name()}', id: {$e->id()}, is_facility: {$is_facility_str}}
HERE;
        }
    }
    echo <<<HERE

];

/* Authentication and authorization context
 */
var auth_is_authenticated="{$auth_svc->isAuthenticated()}";
var auth_type="{$auth_svc->authType()}";
var auth_remote_user="{$auth_svc->authName()}";

var auth_webauth_token_creation="{$_SERVER['WEBAUTH_TOKEN_CREATION']}";
var auth_webauth_token_expiration="{$_SERVER['WEBAUTH_TOKEN_EXPIRATION']}";

var auth_granted = {
HERE;
    $experiments = $logbook->experiments();
    $first = true;
    foreach( $experiments as $e ) {
        if( $first ) $first = false;
        else echo ',';
        $canRead            = $auth_svc->canRead($e->id()) ? 'true' : 'false';
        $canPostNewMessages = $auth_svc->canPostNewMessages($e->id()) ? 'true' : 'false';
        $canEditMessages    = $auth_svc->canEditMessages($e->id()) ? 'true' : 'false';
        $canDeleteMessages  = $auth_svc->canDeleteMessages($e->id()) ? 'true' : 'false';
        $canManageShifts    = $auth_svc->canManageShifts($e->id()) ? 'true' : 'false';
        echo <<<HERE

   {$e->id()} :
     { 'canRead'            : {$canRead},
       'canPostNewMessages' : {$canPostNewMessages},
       'canEditMessages'    : {$canEditMessages},
       'canDeleteMessages'  : {$canDeleteMessages},
       'canManageShifts'    : {$canManageShifts}
     }
   
HERE;
    }
    echo <<<HERE

};

function isAuthorizedFor( priv ) {
    return auth_granted[current_selection.experiment.id][priv];
}

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

        if( $action == 'select_experiment' ) {
            $instr_id   = $_GET['instr_id'];
            $instr_name = $_GET['instr_name'];
            $exper_id   = $_GET['exper_id'];
            $exper_name = $_GET['exper_name'];
            echo "  select_experiment({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}');";

        } else if( $action == 'select_experiment_by_id' ) {
            $id = $_GET['id'];
            $experiment = $logbook->find_experiment_by_id( $id );
            if( is_null( $experiment )) {
                echo "  alert( 'no experiment found for id: {$id}' );";
            } else {
                $instrument = $experiment->instrument();
                $instr_id   = $instrument->id();
                $instr_name = $instrument->name();
                $exper_id   = $experiment->id();
                $exper_name = $experiment->name();
            }
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

        } else if( $action == 'select_run' ) {
            $instr_name = $_GET['instr_name'];
            $exper_name = $_GET['exper_name'];
            $experiment = $logbook->find_experiment( $instr_name, $exper_name );
            if( is_null( $experiment )) {
                echo "  alert( 'no experiment found for instrument: {$instr_name} and experiment: {$exper_name}' );";
            } else {
                $run_num = $_GET['num'];
                $run = $experiment->find_run_by_num( $run_num );
                if( is_null( $run )) {
                    echo "  alert( 'no run found for number: {$run_num}' );";
                } else {
                    $instrument = $experiment->instrument();
                    $instr_id   = $instrument->id();
                    $exper_id   = $experiment->id();
                    $shift_id   = $run->shift()->id();
                    $run_id     = $run->id();
                    echo "  select_experiment_and_run({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}',{$shift_id},{$run_id});";
                }
            }

        } else if( $action == 'select_run_by_id' ) {
            $run_id  = $_GET['id'];
            $run = $logbook->find_run_by_id( $run_id );
            if( is_null( $run )) {
                echo "  alert( 'no run found for id: {$run_id}' );";
            } else {
                $experiment = $run->parent();
                $instrument = $experiment->instrument();
                $instr_id   = $instrument->id();
                $instr_name = $instrument->name();
                $exper_id   = $experiment->id();
                $exper_name = $experiment->name();
                $shift_id   = $run->shift()->id();
                echo "  select_experiment_and_run({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}',{$shift_id},{$run_id});";
            }

        } else if( $action == 'select_message' ) {
            $id    = $_GET['id'];
            $entry = $logbook->find_entry_by_id( $id );
            if( is_null( $entry )) {
                echo "  alert( 'no message found for id: {$id}' );";
            } else {
                $experiment = $entry->parent();
                $instrument = $experiment->instrument();
                $instr_id   = $instrument->id();
                $instr_name = $instrument->name();
                $exper_id   = $experiment->id();
                $exper_name = $experiment->name();
                echo "  select_experiment_and_message({$instr_id},'{$instr_name}',{$exper_id},'{$exper_name}',{$id});";
            }
        } else {
            echo "  alert( 'unsupported action: {$action}' );";
        }
    } else {
        echo "  load( 'help/Welcome.html', 'workarea' );";
    }
    echo <<<HERE

  auth_timer_restart();
}

HERE;

?>

</script>
<!--
This is the end of the PHP generated JavaScript code
-->

<!--
Page-specific script
-->
<script type="text/javascript">

/*
 * Browser information.
 */
var browser_name=navigator.appName;
var browser_version=parseFloat(navigator.appVersion);

function browser_is_MSIE() {
    return browser_name == 'Microsoft Internet Explorer';
}

/*
 * Session expiration timer for WebAuth authentication.
 */
var auth_timer = null;
function auth_timer_restart() {
    if( auth_is_authenticated && ( auth_type == 'WebAuth' ))
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
    document.title = 'Electronic LogBook: '+instr_name+' / '+exper_name;
    document.getElementById( "application_subtitle" ).innerHTML =
        instr_name+' / '+exper_name;

    menubar_enable( menubar_group_shifts );
    if( is_facility( exper_id ))
        menubar_disable( menubar_group_runs );
    else
        menubar_enable( menubar_group_runs );
    menubar_enable( menubar_group_browse );
    menubar_enable( menubar_group_search );
    menubar_enable( menubar_group_subscribe );
}

/* Check if the specified id corresponds to a facility or an experiment.
 * Return 'true' if it's a facility.
 * Alert if the specified id doesn't correspond to anything.
 */
function is_facility( id ) {
    for( i = 0; i < experiments.length; i++ )
        if( experiments[i].id == id ) return experiments[i].is_facility;
    alert("no experiment/facility for id: "+id);
}
function experiment_or_facility( id ) {
    if( is_facility( id )) return 'Facility';
    return 'Experiment';
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
var locations_list   = [];
var instruments_list = [];
for( i = 0; i < instruments.length; i++ ) {
    if( instruments[i].is_location ) {
        locations_list.push (
            {
                text: instruments[i].name,
                url:  "javascript:list_experiments('"+instruments[i].name+"', true)"
            }
        );
    } else {
        instruments_list.push (
            {
                text: instruments[i].name,
                url:  "javascript:list_experiments('"+instruments[i].name+"', false)"
            }
        );
    }
}
var menubar_group_facilities = menubar_data.length;
menubar_data.push ( {
    id:    'facilities',
    href:  '#facilities',
    title: 'Facilities',
    title_style: null,
    itemdata: [
        locations_list,
        [
            { text: "List all", url: "javascript:list_experiments(null, true)" }
        ]
    ],
    disabled: false }
);
var menubar_group_experiments = menubar_data.length;
menubar_data.push ( {
    id:    'experiments',
    href:  '#experiments',
    title: 'Experiments',
    title_style: null,
    itemdata: [
        instruments_list,
        [
            { text: "List all", url: "javascript:list_experiments(null, false)" }
        ]
    ],
    disabled: false }
);
var menubar_group_shifts = menubar_data.length;
menubar_data.push ( {
    id:    'shifts',
    href:  '#shifts',
    title: 'Shifts',
    title_style: null,
    itemdata: [
        { text: "List all",     url: "javascript:list_shifts()" },
        { text: "Display last", url: "javascript:select_last_shift()" },
        { text: "Begin new",    url: "javascript:begin_new_shift()" } ],
    disabled: true }
);
var menubar_group_runs = menubar_data.length;
menubar_data.push ( {
    id:    'runs',
    href:  '#runs',
    title: 'Runs',
    title_style: null,
    itemdata: [
        { text: "List all",     url: "javascript:list_runs()" },
        { text: "Display last", url: "javascript:select_last_run()" } ],
    disabled: true }
);
var menubar_group_browse = menubar_data.length;
menubar_data.push ( {
    id:    'browse',
    href:  '#browse',
    title: 'Browse',
    itemdata: [
        { text: "History...",    url: "javascript:browse_contents()" } ],
    disabled: true }
);
var menubar_group_search = menubar_data.length;
menubar_data.push ( {
    id:    'search',
    href:  '#find',
    title: 'Find',
    itemdata:  [
        { text: "Text in all messages", url: "javascript:search_contents_simple()", disabled: true },
        { text: "Advanced dialog",      url: "javascript:search_contents()" } ],
    disabled: true }
);
var menubar_group_subscribe = menubar_data.length;
menubar_data.push ( {
    id:    'subscribe',
    href:  '#subscribe',
    title: 'Subscribe',
    itemdata:  [
        { text: "Automatic notifications by e-mail", url: "javascript:subscribe()" } ],
    disabled: true }
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

function logout() {
    ask_yesno(
        'popupdialogs',
        '<span style="color:red; font-size:16px;">Session Logout Warning</span>',
        '<p style="text-align:left;">You are about to logout from the current WebAuth session. '+
        'Press <b>Yes</b> to proceed with the logout. Press <b>No</b> to stay in the session.</p>',
        function() {
            document.cookie = 'webauth_wpt_krb5=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
            document.cookie = 'webauth_at=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
            refresh_page();
        },
        function() {}
    );
    return;
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
              scope: this.dataTable } );
    };
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

// Display a list of experiments (or facilities) in a given scope.
//
// If null is passed as 'instr' then all known instanceswill be displayed.
// Otherwise only those instances registered within 'instr' will beshown.
//
// The last parameter 'is_location' indicates whether 'instr' is an instrument
// or a location name.
//
function list_experiments( instr, is_location ) {

    set_context(
        ( is_location ? 'Select Facility >' : 'Select Experiment >' ));

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div id="workarea_table_container">'+
        '  <div id="workarea_table_paginator"></div>'+
        '  <div id="workarea_table_body"></div>'+
        '</div>';

    var params = ( instr == null ? '' : '?instr='+instr );
    if( is_location ) {
        if( params == '' ) params = '?is_location=';
        else               params = params + '&is_location=';
    }
    var url = 'RequestExperiments.php'+params;
    var table = null;
    if( is_location )
        table = new Table (
            "workarea_table",
            [ { key: "location",          sortable: true,  resizeable: false },
              { key: "facility",          sortable: true,  resizeable: false },
              { key: "registration_time", sortable: true,  resizeable: false },
              { key: "description",       sortable: false, resizeable: true } ],
            url,
            false,
            10
        );
    else
        table = new Table (
            "workarea_table",
            [ { key: "instrument",        sortable: true,  resizeable: false },
              { key: "experiment",        sortable: true,  resizeable: false },
              { key: "status",            sortable: true,  resizeable: false },
              { key: "begin_time",        sortable: true,  resizeable: false },
              { key: "end_time",          sortable: true,  resizeable: false },
              { key: "registration_time", sortable: true,  resizeable: false },
              { key: "description",       sortable: false, resizeable: true } ],
            url,
            false,
            10
        );
    //table.refreshTable();
}

function select_experiment( instr_id, instr_name, exper_id, exper_name ) {
    set_current_selection( instr_id, instr_name, exper_id, exper_name );
    display_experiment();
}

function display_experiment() {

    var exper_id = current_selection.experiment.id;

    set_context ( experiment_or_facility( exper_id )+' >' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/'+( is_facility( exper_id ) ? 'Fac' : 'Exp' )+'Summary.png" />'+
        '</div>'+
        '<div id="experiment_info_container" style="height:250px;">Loading...</div>'+
        '<div style="margin-top:40px; padding-right:15px;">'+
        '  <div id="messages_actions_container"></div>'+
        '</div>';

    load( 'Display'+( is_facility( exper_id ) ? 'Facility' : 'Experiment' )+'.php?id='+exper_id, 'experiment_info_container' );

    var messages_dialog = create_messages_dialog( 'experiment' );
}

function close_messages_dialog() {
    document.getElementById('new_message_dialog').style.display='none';
    new_message_button.set( 'disabled', false );
}

var current_messages_table = null;

function create_messages_dialog( scope ) {

    var html_new_message =
        '<div id="new_message_dialog" style="padding:10px; margin-left:10px; margin-top:30px; background-color:#e0e0e0; border:solid 1px #c0c0c0; display:none;">'+
        '<form enctype="multipart/form-data" name="new_message_form" action="NewFFEntry.php" method="post">'+
        '  <input type="hidden" name="author_account" value="'+auth_remote_user+'" style="padding:2px; width:200px;" />'+
        '  <input type="hidden" name="id" value="'+current_selection.experiment.id+'" />'+
        '  <input type="hidden" name="scope" value="'+scope+'" />'+
        '  <div id="file_descriptions"></div>'+
        '  <div id="message_tags"></div>';
    if( scope == "experiment") {
         html_new_message +=
        '  <input type="hidden" name="actionSuccess" value="select_experiment" />';
    } else if( scope == "shift") {
         html_new_message +=
        '  <input type="hidden" name="shift_id" value="'+current_selection.shift.id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment_and_shift" />';
    } else if( scope == "run") {
         html_new_message +=
        '  <input type="hidden" name="run_id" value="'+current_selection.run.id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment_and_run" />';
    }
    html_new_message +=
        '  <input type="hidden" name="MAX_FILE_SIZE" value="10000000">'+
        '  <div>'+
        '    <em class="lb_label">New message:</em>'+
        '  </div>'+
        '  <table><tbody>'+
        '    <tr>'+
        '      <td valign="top">'+
        '        <div id="new_message_body" style="margin-right:0px; padding:4px;">'+
        '          <input id="new_message_text" type="text" name="message_text" size="128" value="" />'+
        '        </div>'+
        '      </td>'+
        '      <td valign="top">'+
        '        <div id="new_message_dialog_container">'+
        '          <button id="message_extend_button">Options &gt;</button>'+
        '          <button id="message_submit_button">Submit</button>'+
        '          <button id="message_cancel_button">Cancel</button>'+
        '        </div>'+
        '      </td>'+
        '    </tr>'+
        '  </tbody></table>'+
        '</form>'+
        '</div>';

    document.getElementById('messages_actions_container').innerHTML=
        '<div id="messagesarea"></div>';

    var limit_per_view = null;
    var scope_str = '';
    if(      scope == "experiment" ) { scope_str = ''; limit_per_view = '25'; }
    else if( scope == "shift"      ) { scope_str = 'shift_id='+current_selection.shift.id; }
    else if( scope == "run"        ) { scope_str = 'run_id='+current_selection.run.id; }

    var text2search='',
        search_in_messages=true, search_in_tags=true, search_in_values=true,
        posted_at_experiment=true, posted_at_shifts=true, posted_at_runs=true,
        begin='', end='',
        tag='',
        author='',
        auto_refresh=true;
    current_messages_table = display_messages_table(
        scope_str,
        text2search,
        search_in_messages, search_in_tags, search_in_values,
        posted_at_experiment, posted_at_shifts, posted_at_runs,
        begin, end,
        tag,
        author,
        isAuthorizedFor( 'canPostNewMessages' ) ? html_new_message : '',
        auto_refresh,
        limit_per_view );

    this.extendedShown = false;

    this.message_extend_button = new YAHOO.widget.Button( "message_extend_button" );
    this.message_submit_button = new YAHOO.widget.Button( "message_submit_button" );
    this.message_cancel_button = new YAHOO.widget.Button( "message_cancel_button" );
    this.message_cancel_button.on (
        "click",
        close_messages_dialog
    );

    this.tags = [ { 'delete': false, 'tag': '', 'value' : ''}  ];
    this.tags_table = null;

    this.oPushButtonAddTag = null;
    this.oPushButtonRemoveTag = null;

    function synchronize_tags_data() {
        if( this.tags_table == null ) return;
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
        var tags_html = ' <'+'input type="hidden" name="num_tags" value="'+this.tags.length+'" />';;
        for( var i = 0; i < this.tags.length; i++ ) {
            tags_html += ' <'+'input type="hidden" name="tag_name_'+i+'" value="'+this.tags[i].tag+'" />';
            tags_html += ' <'+'input type="hidden" name="tag_value_'+i+'" value="'+this.tags[i].value+'" />';
        }
        document.getElementById('message_tags').innerHTML=tags_html;
    }

    function AddAndRefreshTagsTable() {
        this.tags_table.dataTable.addRow (
            { tag: "", value: "" }, 0 );
    }

    this.files = [];
    var file2attach_sequence=1;
    var id = 'file2attach_'+file2attach_sequence;
    this.files.push({
        'delete': false,
        'file': '<input type="file" name="'+id+'" id="'+id+'" />',
        'description': '',
        'id': id
    });
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
    function AddAndRefreshFilesTable() {
        file2attach_sequence++;
        var id = 'file2attach_'+file2attach_sequence;
        this.files_table.dataTable.addRow (
            { file: '<input type="file" name="'+id+'" id="'+id+'" />', description: "", id: id }, 0 );
    }

    function onExtendedClick() {
        var new_message_body = document.getElementById('new_message_body');
        if( !this.extendedShown ) {
            document.getElementById('new_message_body').innerHTML=
                '<textarea id="new_message_text" type="text" name="message_text"'+
                ' rows="12" cols="128" style="padding:1px;"'+
                ' title="This is multi-line text area in which return will add a new line of text.'+
                ' Use Submit button to post the message.">'+
                document.getElementById('new_message_text').value+'</textarea>'+
                '<div style="margin-top:12px;">'+
                '  <em class="lb_label">Author:</em>'+
                '  <input type="text" name="author_name" value="'+auth_remote_user+'" style="padding:2px; width:200px;" />'+
                '</div>'+
                '<div style="margin-right:6px;" align="left">'+
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
                '            <button id="add_tag_button">Add</button>'+
                '            <button id="remove_tag_button">Update</button>'+
                '          </div>'+
                '        </div>'+
                '      </td>'+
                '      <td><div style="width:8px;"></div></td>'+
                '      <td valign="top">'+
                '        <div style="margin-top:8px; padding:8px; background-color:#f0f0f0;" >'+
                '          <div id="files_table">'+
                '            <div id="files_table_paginator"></div>'+
                '            <div id="files_table_body"></div>'+
                '          </div>'+
                '          <div style="margin-top:8px;" align="right" >'+
                '            <button id="add_file_button">Add</button>'+
                '            <button id="remove_file_button">Update</button>'+
                '          </div>'+
                '        </div>'+
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
                this.files,
                null
            );

        } else {
            document.getElementById('new_message_body').innerHTML=
                '<input id="new_message_text" type="text" name="message_text" size="128" value="'+
                document.getElementById('new_message_text').value+'" />';
            new_message_body.style.paddingBottom="4px";
            this.tags_table = null;
            this.files_table = null;
        }
        this.extendedShown = !this.extendedShown;
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

/* This is a separate dialog for displaying persistent links for messages
 */
 var last_lid = null;    // The last element used to display the message link dialog
                         // This element can be reused.

function create_message_link_dialog( lid, message_id ) {

    var idx = window.location.href.indexOf( '?' );
    var link = ( idx < 0 ? window.location.href : window.location.href.substr( 0, idx ))+'?action=select_message&id='+message_id;

    // Check if we can reuse the last element. We can only do ths if
    // it was associated with the same message. Otherwise we'll recreate
    // the whole dialog from scratch at the specified location.
    //
    if( last_lid != null ) {
        if( last_lid != lid ) {
            last_lid.style.display = 'none';
            last_lid.innerHTML='';
        } else {
            if( lid.style.display == 'none') lid.style.display = 'block';
            else                             lid.style.display = 'none';
            return;
        }
    }
    last_lid = lid;
    lid.style.display = 'block';

    lid.innerHTML =
        '<div>'+
        '  <em class="lb_label">URL: </em><a href="'+link+
        '" target="_blank" class="lb_link" title="Click to open in a separate window/tab, or cut and paste to incorporate into another document as a link.">'+
        link+'</a>'+
        '</div>';
}

/* This is a separate dialog for posting replies to messages
 */
var last_rid = null;    // The last element used to display the message reply dialog
                        // This element can be reused.

function create_message_reply_dialog( rid, message_id ) {

    // Close the editor if any is open
    //
    if( last_eid != null ) last_eid.style.display = 'none';

    // Check if we can reuse the last element. We can only do ths if
    // it was associated with the same message. Otherwise we'll recreate
    // the whole dialog from scratch at the specified location.
    //
    if( last_rid != null ) {
        if( last_rid != rid ) {
            last_rid.style.display = 'none';
            last_rid.innerHTML='';
        } else {
            if( rid.style.display == 'none') rid.style.display = 'block';
            else                             rid.style.display = 'none';
            return;
        }
    }
    last_rid = rid;
    rid.style.display = 'block';

    var scope='message';

    rid.innerHTML =
        '<form enctype="multipart/form-data" name="message_reply_form" action="NewFFEntry.php" method="post">'+
        '  <input type="hidden" name="author_account" value="'+auth_remote_user+'" style="padding:2px; width:200px;" />'+
        '  <input type="hidden" name="id" value="'+current_selection.experiment.id+'" />'+
        '  <input type="hidden" name="scope" value="'+scope+'" />'+
        '  <input type="hidden" name="message_id" value="'+message_id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment" />'+
        '  <input type="hidden" name="MAX_FILE_SIZE" value="10000000">'+
        '  <div id="message_reply_file_descriptions"></div>'+
        '  <div id="message_tags"></div>'+
        '  <div>'+
        '    <em class="lb_label">Reply with:</em>'+
        '  </div>'+
        '  <table><tbody>'+
        '    <tr>'+
        '      <td valign="top">'+
        '        <div id="message_reply_body" style="margin-right:10px; padding:4px;">'+
        '          <input id="message_reply_text" type="text" name="message_text" size="128" value="" />'+
        '        </div>'+
        '      </td>'+
        '      <td valign="top">'+
        '        <div id="message_reply_dialog_container">'+
        '          <button id="message_reply_extend_button">Options &gt;</button>'+
        '          <button id="message_reply_submit_button">Submit</button>'+
        '          <button id="message_reply_cancel_button">Cancel</button>'+
        '        </div>'+
        '      </td>'+
        '    </tr>'+
        '  </tbody></table>'+
        '</form>';

    var extendedShown = false;

    this.message_reply_extend_button = new YAHOO.widget.Button( "message_reply_extend_button" );
    this.message_reply_submit_button = new YAHOO.widget.Button( "message_reply_submit_button" );
    this.message_reply_cancel_button = new YAHOO.widget.Button( "message_reply_cancel_button" );
    this.message_reply_cancel_button.on (
        "click",
        function() { rid.style.display = 'none'; }
    );

    var tags = [ { 'delete': false, 'tag': '' , 'value': '' } ];
    var tags_table = null;

    this.oPushButtonAddTag = null;
    this.oPushButtonRemoveTag = null;

    function synchronize_tags_data() {
        if( tags_table == null ) return;
        var rs = tags_table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        tags = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( !r.getData('delete')) {
                tags.push ( {
                    tag: r.getData('tag'),
                    value: r.getData('value')});
            } else {
                tags_table.dataTable.deleteRow(i);
            }
        }
        // Also refresh the markup with inputs for tags
        //
        var tags_html = ' <'+'input type="hidden" name="num_tags" value="'+tags.length+'" />';;
        for( var i = 0; i < tags.length; i++ ) {
            tags_html += ' <'+'input type="hidden" name="tag_name_'+i+'" value="'+tags[i].tag+'" />';
            tags_html += ' <'+'input type="hidden" name="tag_value_'+i+'" value="'+tags[i].value+'" />';
        }
        document.getElementById('message_tags').innerHTML=tags_html;
    }

    function AddAndRefreshTagsTable() {
        tags_table.dataTable.addRow (
            { tag: "", value: "" }, 0 );
    }

    var files = [];
    var file2attach_sequence=1;
    var id = 'file2attach_'+file2attach_sequence;
    files.push({
        'delete': false,
        'file': '<input type="file" name="'+id+'" id="'+id+'" />',
        'description': '',
        'id': id
    });
    var files_table = null;

    this.oPushButtonAddFile = null;
    this.oPushButtonRemovFile = null;

    function synchronize_files_data() {
        if( files_table == null ) return;
        var rs = files_table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        files = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( !r.getData('delete')) {
                files.push ( {
                    file: r.getData('file'),
                    filename: document.getElementById(r.getData('id')).value,
                    description: r.getData('description'),
                    id: r.getData('id')});
            } else {
                files_table.dataTable.deleteRow(i);
            }
        }

        // Also refresh the markup with inputs for file descriptions
        //
        var descriptions ='';
        for( var i = 0; i < files.length; i++ ) {
            var id = files[i].id;
            descriptions += ' <'+'input type="hidden" name="'+id+'" value="'+files[i].description+'" />';
        }
        document.getElementById('message_reply_file_descriptions').innerHTML=descriptions;
    }
    var file2attach_sequence=1;
    function AddAndRefreshFilesTable() {
        file2attach_sequence++;
        id = 'file2attach_'+file2attach_sequence;
        files_table.dataTable.addRow (
            { file: '<input type="file" name="'+id+'" id="'+id+'" />', description: "", id: id }, 0 );
    }

    function onExtendedClick() {
        var message_reply_body = document.getElementById('message_reply_body');
        if( !extendedShown ) {
            document.getElementById('message_reply_body').innerHTML=
                '<textarea id="message_reply_text" type="text" name="message_text"'+
                ' rows="12" cols="128" style="padding:1px;"'+
                ' title="This is multi-line text area in which return will add a new line of text.'+
                ' Use Submit button to post the message.">'+
                document.getElementById('message_reply_text').value+'</textarea>'+
                '<div style="margin-top:12px;">'+
                '  <em class="lb_label">Author:</em>'+
                '  <input type="text" name="author_name" value="'+auth_remote_user+'" style="padding:2px; width:200px;" />'+
                '</div>'+
                '<div style="margin-right:6px;" align="left">'+
                '  <table style="margin-top:20px;"><tbody>'+
                '    <tr>'+
                '      <td align="left"><em class="lb_label">Tags</em></td>'+
                '      <td><div style="width:8px;"></div></td>'+
                '      <td align="left"><em class="lb_label">Attachments</em></td>'+
                '    </tr>'+
                '    <tr>'+
                '      <td valign="top">'+
                '        <div style="margin-top:8px; padding:8px; background-color:#f0f0f0;" >'+
                '          <div id="message_reply_tags_table">'+
                '            <div id="message_reply_tags_table_paginator"></div>'+
                '            <div id="message_reply_tags_table_body"></div>'+
                '          </div>'+
                '          <div style="margin-top:8px;" align="right" >'+
                '            <button id="message_reply_add_tag_button">Add</button>'+
                '            <button id="message_reply_remove_tag_button">Update</button>'+
                '          </div>'+
                '        </div>'+
                '      </td>'+
                '      <td><div style="width:8px;"></div></td>'+
                '      <td valign="top">'+
                '        <div style="margin-top:8px; padding:8px; background-color:#f0f0f0;" >'+
                '          <div id="message_reply_files_table">'+
                '            <div id="message_reply_files_table_paginator"></div>'+
                '            <div id="message_reply_files_table_body"></div>'+
                '          </div>'+
                '          <div style="margin-top:8px;" align="right" >'+
                '            <button id="message_reply_add_file_button">Add</button>'+
                '            <button id="message_reply_remove_file_button">Update</button>'+
                '          </div>'+
                '        </div>'+
                '      </td>'+
                '    </tr>'+
                '  </tbody></table>'+
                '</div>';
            message_reply_body.style.paddingBottom="10px";
            this.oPushButtonAddTag = new YAHOO.widget.Button( "message_reply_add_tag_button" );
            this.oPushButtonAddTag.on (
                "click",
                function( p_oEvent ) { AddAndRefreshTagsTable(); }
            );
            this.oPushButtonRemoveTag = new YAHOO.widget.Button( "message_reply_remove_tag_button" );
            this.oPushButtonRemoveTag.on (
                "click",
                function( p_oEvent ) { synchronize_tags_data(); }
            );
            tags_table = new TableLocal (
                "message_reply_tags_table",
                [ { key: "delete", formatter: "checkbox" },
                  { key: "tag",   sortable: true, resizeable: false,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true})},
                  { key: "value", sortable: true, resizeable: false,
                    editor: new YAHOO.widget.TextareaCellEditor({disableBtns:true})} ],
                tags,
                null
            );

            this.oPushButtonAddFile = new YAHOO.widget.Button( "message_reply_add_file_button" );
            this.oPushButtonAddFile.on (
                "click",
                function( p_oEvent ) { AddAndRefreshFilesTable(); }
            );
            this.oPushButtonRemoveFile = new YAHOO.widget.Button( "message_reply_remove_file_button" );
            this.oPushButtonRemoveFile.on (
                "click",
                function( p_oEvent ) { synchronize_files_data(); }
            );
            files_table = new TableLocal (
                "message_reply_files_table",
                [ { key: "delete", formatter: "checkbox" },
                  { key: "file",   sortable: true, resizeable: false },
                  { key: "description", sortable: true, resizeable: false,
                    editor: new YAHOO.widget.TextareaCellEditor({disableBtns:true})},
                  { key: "id", sortable: false, hidden:true }],
                files,
                null
            );

        } else {
            document.getElementById('message_reply_body').innerHTML=
                '<input id="message_reply_text" type="text" name="message_text" size="128" value="'+
                document.getElementById('message_reply_text').value+'" />';
            message_reply_body.style.paddingBottom="4px";
            tags_table = null;
        }
        extendedShown = !extendedShown;
    }
    this.message_reply_extend_button.on (
        "click",
        function( p_oEvent ) {
            onExtendedClick();
        }
    );
    function SubmitRequest() {
        synchronize_tags_data();
        synchronize_files_data();
        document.forms.message_reply_form.submit();
    }
    this.message_reply_submit_button.on (
        "click",
        function( p_oEvent ) {
            SubmitRequest();
        }
    );
}

function find_message_by_id_impl( re, id ) {
    if( re.id == id ) return Object( re );
    for( var i = 0; i < re.children.length; i++ ) {
        var child = find_message_by_id_impl( Object( eval( '('+re.children[i]+')' )), id );
        if( child != null ) return child;
    }
    return null;
}

function find_message_by_id( id ) {

    // Pay attention to the little optimization which is based on two facts:
    //
    // - newest messages are stored in the very end of the cache
    // - this lookup operation is most frequently used on newer rather than
    //   older messages
    //
    for( var i = last_search_result.length-1; i >= 0; i-- ) {
        var re = find_message_by_id_impl( Object( last_search_result[i] ), id );
        if( re != null ) return re;
    }
}

/* This is a separate dialog for editing messages
 */
var last_eid = null;    // The last element used to display the message editing dialog
                        // This element can be reused.

function create_message_edit_dialog( eid, message_id ) {

    // Close the reply dialog if any is open
    //
    if( last_rid != null ) last_rid.style.display = 'none';

    // Check if we can reuse the last element. We can only do ths if
    // it was associated with the same message. Otherwise we'll recreate
    // the whole dialog from scratch at the specified location.
    //
    if( last_eid != null ) {
        if( last_eid != eid ) {
            last_eid.style.display = 'none';
            last_eid.innerHTML='';
        } else {
            if( eid.style.display == 'none') eid.style.display = 'block';
            else                             eid.style.display = 'none';
            return;
        }
    }
    last_eid = eid;
    eid.style.display = 'block';

   /* TODO; Replace this with a recursive search in the message tree,
    * then get rid of the 'message_idx' in the function's signature and
    * its client code.
    *
    *   var re = find_message_by_if( id );
    *   if( re == null ) alert("internal bug");
    */
    //var re = last_search_result[message_idx];

    var re = find_message_by_id( message_id );
    if( re == null ) {
        alert( 'Internal error #12: inconsistent data cache. Try reloading the application.' );
        return;
    }

    eid.innerHTML =
        '<form name="message_edit_form" action="UpdateFFEntry.php" method="post">'+
        '  <input type="hidden" name="id" value="'+message_id+'" />'+
        '  <input type="hidden" name="content_type" value="TEXT" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment" />'+
        '  <div>'+
        '    <em class="lb_label">Edited message:</em>'+
        '  </div>'+
        '  <table><tbody>'+
        '    <tr>'+
        '      <td valign="top">'+
        '        <div id="message_edit_body" style="margin-right:10px; padding:4px;">'+
        '          <textarea rows="12" cols="128" name="content" size="128" style="padding:1px;">'+re.content+'</textarea>'+
        '        </div>'+
        '      </td>'+
        '      <td valign="top">'+
        '        <div id="message_edit_dialog_container">'+
        '          <button id="message_edit_submit_button">Submit</button>'+
        '          <button id="message_edit_cancel_button">Cancel</button>'+
        '        </div>'+
        '      </td>'+
        '    </tr>'+
        '  </tbody></table>'+
        '</form>';

    var message_edit_submit_button = new YAHOO.widget.Button( "message_edit_submit_button" );
    message_edit_submit_button.on (
        "click",
        function( p_oEvent ) { document.message_edit_form.submit(); }
    );
    var message_edit_cancel_button = new YAHOO.widget.Button( "message_edit_cancel_button" );
    message_edit_cancel_button.on (
        "click",
        function() { eid.style.display = 'none'; }
    );
}

function create_message_delete_dialog( did, id ) {
    /*
     * TODO: Dialogs do not work consistently on different browsers if
     * a dynamically allocated HTML element is used. Investigate the problem
     * later!
     *
     * var dialog_id = browser_is_MSIE() ? "popupdialogs" : did.id;
     */
    var dialog_id = "popupdialogs";
    var dialog_title = '<em style="color:red; font-weight:bold; font-size:18px;">Delete Selected Message?</em>';
    var dialog_body =
        '<div style="text-align:left;">'+
        '  <form  name="delete_message_form" action="DeleteFFEntry.php" method="post">'+
        '    <b>PLEASE, READ THIS:</b> the selected message and all its children are about to be destroyed!'+
        '    The information may be permanently lost as a result of the operation. In most cases a better'+
        '    alternative would be to <b>Reply</b> to the message or to <b>Edit</b> its contents.'+
        '    Also note that your identity will be recorded.'+
        '    <br><br>'+
        '    Press <b>Yes</b> to delete the message, or press <b>No</b> to abort the operation.'+
        '    <input type="hidden" name="id" value="'+id+'" />'+
        '    <input type="hidden" name="actionSuccess" value="select_experiment" />'+
        '  </form>'+
        '</div>';
    ask_yesno(
        dialog_id,
        dialog_title,
        dialog_body,
        function() { document.delete_message_form.submit(); },
        function() {
            /*
             * NOTE: See 'TODO' above for an explanation why this is prohibited.
             *
             * did.innerHTML='';
             */
        }
    );
}

function preview_atatchment( id ) {

    var viewarea = document.getElementById('viewarea');
    viewarea.innerHTML='<img src="ShowAttachment.php?id='+id+'" width="250" height="250"/>';
}

function list_shifts() {

    set_context(
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
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

function select_last_shift() {
    load_then_call(
        'RequestShifts.php?id='+current_selection.experiment.id+'&last',
        function( result ) {
            select_shift( result.ResultSet.Result[0].id );
        },
        function( status ) {
            alert( status );
        }
    );
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
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
        'Shift >' );
    var html =
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/ShiftSummary.png" />'+
        '</div>'+
        '<div id="experiment_info_container" style="height:200px;">Loading...</div>';
    if( !is_facility( current_selection.experiment.id )) {
        html +=
        '<div style="margin-top:40px; margin-bottom:20px;">'+
        '  <img src="images/Runs.png" />'+
        '</div>'+
        '<div id="runs_table" style="margin-left:10px; padding:10px;"></div>';
    }
    html +=
        '<br>'+
        '<br>'+
        '<div id="messages_actions_container" style="margin-top:20px;" ></div>';

    document.getElementById('workarea').innerHTML = html;

    load( 'DisplayShift.php?id='+current_selection.shift.id, 'experiment_info_container' );

    if( !is_facility( current_selection.experiment.id )) {
        var runs = create_runs_table (
            'RequestRuns.php?shift_id='+current_selection.shift.id,
            false
        );
    }
    var messages_dialog = create_messages_dialog( 'shift' );
}

function close_shift( shift_id ) {
    if( !isAuthorizedFor( 'canManageShifts' )) {
        post_auth_error( "popupdialogs" );
        return;
    }
    ask_complex_input(
        "popupdialogs",
        "Close Last Shift",
        '<form  name="close_shift_form" action="CloseShift.php" method="post">'+
        '  <b>ATTENTION:</b> the selected shift is about to be closed. Press <b>Submit</b>'+
        '   to proceed, or press <b>Cancel</b> to abort the operation.'+
        '  <input type="hidden" name="id" value="'+shift_id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment_and_shift" />'+
        '</form>',
        function() {
            document.close_shift_form.submit();
        }
    );
}

function begin_new_shift() {
    if( !isAuthorizedFor( 'canManageShifts' )) {
        post_auth_error( "popupdialogs" );
        return;
    }
    ask_complex_input(
        "popupdialogs",
        "Begin New Shift",
        '<p>This operation will start a new shift. '+
        'The shift will begin instantly after submitting the form. '+
        'The previously open shift (if any) will be atomatically closed. '+
        'Please, provide a leader name or account and a list of (up to 10 extra) crew members.</p><br>'+
        '<form  name="begin_new_shift_form" action="CreateShift.php" method="post">'+
        '  <input type="hidden" name="max_crew_size" value="10" />'+
        '  <input type="hidden" name="author" value="'+auth_remote_user+'" />'+
        '  <table><tbody>'+
        '    <tr>'+
        '      <td><b>Shift Leader:&nbsp;</b></td>'+
        '      <td><input class="lb_sparse" type="text" name="leader" value="'+auth_remote_user+'" /></td>'+
        '      <td></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td><div style="height:10px;"></div></td>'+
        '      <td></td>'+
        '      <td></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td><b>Crew Members:&nbsp;</b></td>'+
        '      <td><input class="lb_sparse" type="text" name="member0" value="" /></td>'+
        '      <td><input class="lb_sparse" type="text" name="member1" value="" /></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td></td>'+
        '      <td><input class="lb_sparse" type="text" name="member2" value="" /></td>'+
        '      <td><input class="lb_sparse" type="text" name="member3" value="" /></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td></td>'+
        '      <td><input class="lb_sparse" type="text" name="member4" value="" /></td>'+
        '      <td><input class="lb_sparse" type="text" name="member5" value="" /></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td></td>'+
        '      <td><input class="lb_sparse" type="text" name="member6" value="" /></td>'+
        '      <td><input class="lb_sparse" type="text" name="member7" value="" /></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td></td>'+
        '      <td><input class="lb_sparse" type="text" name="member8" value="" /></td>'+
        '      <td><input class="lb_sparse" type="text" name="member9" value="" /></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td><div style="height:10px;"></div></td>'+
        '      <td></td>'+
        '      <td></td>'+
        '    </tr>'+
        '    <tr>'+
        '      <td valign="top"><b>Shift Goals:&nbsp;</b></td>'+
        '      <td colspan=2><textarea type="text" rows="12" cols="50" style="padding:2px;" name="goals"></textarea></td>'+
        '      <td></td>'+
        '    </tr>'+
        '  </tbody></table><br>'+
        '  <input class="lb_sparse" type="hidden" name="id" value="'+current_selection.experiment.id+'" />'+
        '  <input class="lb_sparse" type="hidden" name="actionSuccess" value="select_experiment_and_shift" />'+
        '</form>',
        function() {
            document.begin_new_shift_form.submit();
        }
    );
}

function list_runs() {

    set_context(
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
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

function select_last_run() {
    load_then_call(
        'RequestRuns.php?id='+current_selection.experiment.id+'&last',
        function( result ) {
            select_run( result.ResultSet.Result[0].shift_id, result.ResultSet.Result[0].id );
        },
        function( status ) {
            alert( status );
        }
    );
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
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
        '<a href="javascript:display_shift()">Shift</a> > '+
        'Run >' );

    document.getElementById('workarea').innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/RunSummary.png" />'+
        '</div>'+
        '<div id="experiment_info_container" style="height:60px;">Loading...</div>'+
        '<div style="margin-top:40px; margin-bottom:40px;">'+
        '  <img src="images/Parameters.png" />'+
        '</div>'+
        //'<div id="run_parameters" style="height:1500px;">Loading...</div>'+
        '<div id="run_parameters" style="width:800px; height:440px; overflow:auto; border:solid 2px #e0e0e0; margin-left:20px; padding:10px;">Loading...</div>'+
        '<div id="messages_actions_container" style="margin-top:50px;"></div>';

    load( 'DisplayRun.php?id='+current_selection.run.id, 'experiment_info_container' );
    load( 'DisplayRunParams.php?id='+current_selection.run.id, 'run_parameters' );

    var messages_dialog = create_messages_dialog( 'run' );
}

function select_experiment_and_message( instr_id, instr_name, exper_id, exper_name, id ) {
    set_current_selection( instr_id, instr_name, exper_id, exper_name );
    display_message( id );
}

function display_message( id ) {
    set_context(
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
        'Message Viewer > ID='+id );

    stopRefreshOfMessagesTable();

    reset_navarea();
    reset_workarea();

    this.url='SearchOne.php?id='+id;

    var html=
        '<div id="messages_area" style="min-width:800px;">'+
        '  <img src="images/ajaxloader.gif" />&nbsp;searching messages...'+
        '</div>';

    document.getElementById('workarea').innerHTML=html;

    function callback_on_load( result ) {
        if( result.ResultSet.Status != "success" ) {
            document.getElementById('messages_area').innerHTML = result.ResultSet.Message;
        } else {
            last_search_result = result.ResultSet.Result;

            // Put the list of all message headers onto the current page
            // and expand them all.
            //
            var html1 = '';
            for( var i=last_search_result.length-1; i >= 0 ; i-- )
                html1 += event2html(i);
            document.getElementById('messages_area').innerHTML = html1;
            for( var i = 0; i < last_search_result.length; i++ ) {
                var re = last_search_result[i];
                var mid = 'message_'+re.id;
                var m = document.getElementById(mid);
                m.style.display = 'block';
                message_expander_toggle( i );
            }
        }
    }
    function callback_on_failure( http_status ) {
        document.getElementById('messages_area').innerHTML=
            '<b><em style="color:red;" >Error</em></b>&nbsp;Request failed. HTTP status: '+http_status;
    }
    load_then_call( this.url, callback_on_load, callback_on_failure );
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

    var request_shifts_url =
        'RequestShifts.php?id='+current_selection.experiment.id;

    var request_runs_url =
        'RequestRuns.php?id='+current_selection.experiment.id;

    var context='';
    var begin='', end='';

    switch( type ) {
        case TYPE_HISTORY_P:
            end = 'b';
            request_shifts_url += '&end=b';
            request_runs_url += '&end=b';
            context += 'Preparation >';
            break;
        case TYPE_HISTORY_D_DAY:
            begin = data.begin;
            end = data.end;
            request_shifts_url += '&begin='+encodeURIComponent(data.begin)+'&end='+encodeURIComponent(data.end);
            request_runs_url += '&begin='+encodeURIComponent(data.begin)+'&end='+encodeURIComponent(data.end);
            context += ( is_facility( current_selection.experiment.id ) ? ' Day by Day > ' : 'Data Taking > ' )+data.day;
            break;
        case TYPE_HISTORY_F:
            begin = 'e';
            //request_messages_url += '&begin=e';
            request_shifts_url += '&begin=e';
            request_runs_url += '&begin=e';
            context += 'Follow Up >';
            break;
        default:
            alert( "unsupported history element" );
            return;
    }
    set_context (
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
        '<a href="javascript:browse_contents()">Browse</a> > '+
        'History > '+context );

    var subheader_style = 'padding:2px; background-color:#e0e0e0;';

    var html =
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Shifts.png" />'+
        '</div>'+
        '<div id="shifts_table" style="margin-left:10px; padding:10px; margin-bottom:40px;"></div>';
    if( !is_facility( current_selection.experiment.id )) {
        html +=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Runs.png" />'+
        '</div>'+
        '<div id="runs_table" style="margin-left:10px; padding:10px; margin-bottom:40px;"></div>';
    }
    html +=
        '<div id="messagesarea"></div>';

    document.getElementById('workarea').innerHTML = html;

    // Build  YUI tables and use their loading mechanism
    //
    create_shifts_table( request_shifts_url, false );
    if( !is_facility( current_selection.experiment.id )) {
        create_runs_table( request_runs_url, false );
    }
    var scope='',
        text2search='',
        search_in_messages=true, search_in_tags=true, search_in_values=true,
        posted_at_experiment=true, posted_at_shifts=true, posted_at_runs=true,
        tag='',
        author='';
    display_messages_table(
        scope,
        text2search,
        search_in_messages, search_in_tags, search_in_values,
        posted_at_experiment, posted_at_shifts, posted_at_runs,
        begin, end,
        tag,
        author,
        '',
        false, /* auto_refresh */
        null   /* limit_per_view */ );
}

var browse_tree = null;

function browse_contents() {

    set_context (
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
        'Browse By Categories >' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    //workarea.style.borderLeft="solid 6px #e0e0e0";
    //workarea.style.padding = "10px";
    //workarea.style.minHeight="620px";
    workarea.innerHTML='';

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    //navarea.style.minWidth = "220px";
    //navarea.style.padding = "10px";
    navarea.innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/History.png" />'+
        '</div>'+
        '<div id="browse_tree"></div>';

    browse_tree = new YAHOO.widget.TreeView( "browse_tree" );

    // Start build a tree from the current context
    //
    var node_i = new YAHOO.widget.TextNode(
        {   label: '<em style="color:#0071bc;"><b>'+current_selection.instrument.name+'</b></em>',
            expanded: true,
            title: 'This is currently selected instrument/location' },
        browse_tree.getRoot());

    var node_e = new YAHOO.widget.TextNode(
        {   label: '<em style="color:#0071bc;"><b>'+current_selection.experiment.name+'</b></em>',
            expanded: true,
            title: 'This is currently selected experiment/facility' },
        node_i );

    var node_h = new YAHOO.widget.TextNode(
        {   label: 'Timeline',
            expanded: true,
            title: 'Explore the history of various events happened in the experiment/facility.' },
        node_e );

    if( is_facility( current_selection.experiment.id )) {

        var node_h_d = new YAHOO.widget.TextNode(
              {   label: 'Day by day',
                expanded: false,
                title: 'The day-bay-day history of the facility',
                type: TYPE_HISTORY_D },
            node_h );

        var node_s = new YAHOO.widget.TextNode(
            {   label: 'Shifts',
                expanded: false,
                title: 'Explore shifts of the facility',
                type: TYPE_SHIFTS },
            node_e );

        var currentIconMode = 0;

        node_h_d.setDynamicLoad( loadNodeData, currentIconMode );
        node_s.setDynamicLoad  ( loadNodeData, currentIconMode );

    } else {

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

            var currentIconMode = 0;

            node_h_d.setDynamicLoad( loadNodeData, currentIconMode );
            node_s.setDynamicLoad  ( loadNodeData, currentIconMode );
            node_r.setDynamicLoad  ( loadNodeData, currentIconMode );
    }

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
                    '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
                    '<a href="javascript:browse_contents()">Browse</a> > '+
                    'Shifts > ' );
                break;
            case TYPE_RUNS:
                set_context (
                    '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
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

function message_expander_toggle_impl( re ) {

    // Handle the current object first
    //
    var hid = 'message_header_'+re.id;
    var bid = 'message_body_'+re.id;
    var h = document.getElementById(hid);
    var b = document.getElementById(bid);
    if( b.style.display == 'none') {
        h.innerHTML='<b>V</b>';
        b.style.display = 'block';
        show_hide_attachments_impl( true, Object(re));

        // Proceed to children (if any)
        //
        for( var i = 0; i < re.children.length; i++ ) {
            var child = eval( '('+re.children[i]+')' );
            expand_collapse_message_impl( true, Object( child ));
        }
    } else {
        h.innerHTML='<b>&gt;</b>';
        b.style.display = 'none';
    }
    h.style.width='14px';
}

function message_expander_toggle( message_idx ) {
    var re = last_search_result[message_idx];
    message_expander_toggle_impl( Object(re));
}

function expand_collapse_message_impl( expand, re ) {

    // Handle the current object first
    //
    var hid = 'message_header_'+re.id;
    var bid = 'message_body_'+re.id;
    var h = document.getElementById(hid);
    var b = document.getElementById(bid);
    if( expand ) {
        h.innerHTML='<b>V</b>';
        b.style.display = 'block';
        show_hide_attachments_impl( expand, Object(re));
    } else {
        h.innerHTML='<b>&gt;</b>';
        b.style.display = 'none';
    }
    h.style.width='14px';

    // Proceed to children (if any)
    //
    for( var i = 0; i < re.children.length; i++ ) {
        var child = eval( '('+re.children[i]+')' );
        expand_collapse_message_impl( expand, Object( child ));
    }
}

function expand_collapse_all_messages_impl( expand, message_idx ) {
    var re = last_search_result[message_idx];
    expand_collapse_message_impl( expand, Object( re ));
}

function expand_collapse_all_messages( expand ) {
    for( var message_idx = last_message_idx-1; message_idx >= first_message_idx; message_idx-- ) {
        expand_collapse_all_messages_impl( expand, message_idx );
    }
}

function show_hide_attachments_impl( show, re ) {

    for( var i = 0; i < re.attachments.length; i++ ) {

        var a = re.attachments[i];
        var descr = attachment_descriptors[String(a.id)];
        var h = document.getElementById( descr.ahid );
        var a = document.getElementById( descr.aid );
        if( show ) {
            descr.loader.load();
            h.innerHTML='<b>V</b>';
            a.style.display = 'block';
        } else {
            h.innerHTML='<b>&gt;</b>';
            a.style.display = 'none';
        }
        h.style.width='14px';
    }
    // Proceed to children (if any)
    //
    for( var i = 0; i < re.children.length; i++ ) {
        var child = eval( '('+re.children[i]+')' );
        show_hide_attachments_impl( show, Object( child ));
    }
}

function show_hide_attachments( show, message_idx ) {

    var re = last_search_result[message_idx];

    // Show attachments only for those messages whose body are presently shown
    // (on the current page.)
    //
    var bid = 'message_body_'+re.id;

    if( document.getElementById( bid ).style.display != 'none' ) {
        show_hide_attachments_impl( show, Object( re ));
    }
}

function show_hide_all_attachments( show ) {
    for( var message_idx = last_message_idx-1; message_idx >= first_message_idx; message_idx-- )
        show_hide_attachments( show, message_idx );
}

function AttachmentLoader( a, aid ) {
    this.type = a.type.split('/');
    this.id = a.id;
    this.aid = aid;
    this.loaded = false;
    this.load = function () {
        if( this.loaded ) return;
        this.loaded = true;
        var a_elem = document.getElementById( this.aid );
        if( this.type[0] == 'image' ) {
            a_elem.innerHTML =
            '<img max-width="800" src="ShowAttachment.php?id='+this.id+'" />';
        } else if( this.type[0] == 'text' ) {
            var aid4text = 'attachment_id_'+this.id+'_txt';
            a_elem.innerHTML =
            '<div style="max-width:800px; min-height:40px; max-height:200px; overflow:auto; border:solid 1px;"><textbox><pre id="'+aid4text+'"></pre></textbox></div>';
            load( 'ShowAttachment.php?id='+this.id, aid4text );
        } else if( this.type[0] == 'application' && this.type[1] == 'pdf' ) {
            a_elem.innerHTML =
            '<object data="ShowAttachment.php?id='+this.id+'" type="application/pdf" width="800" height="600"></object>';
        } else if( this.type[0] == 'application' && this.type[1] == 'rtf' ) {
            a_elem.innerHTML =
            '<object data="ShowAttachment.php?id='+this.id+'" type="application/rtf" width="800" height="600"></object>';
        } else {
            a_elem.innerHTML =
            '<img src="images/NoPreview.png" />';
        }
    }
}

var attachment_descriptors=[];
function attachment_expander_toggle( attachment_idx ) {

    var attachment_descr = attachment_descriptors[String(attachment_idx)];
    var ahid = attachment_descr.ahid;
    var aid = attachment_descr.aid;
    var h = document.getElementById(ahid);
    var a = document.getElementById(aid);
    if( a.style.display == 'none') {
        attachment_descr.loader.load();
        h.innerHTML='<b>V</b>';
        a.style.display = 'block';
    } else {
        h.innerHTML='<b>&gt;</b>';
        a.style.display = 'none';
    }
    a.style.width='14px';
}

function child_message_expander_toggle( h, b ) {
    if( b.style.display == 'none') {
        h.innerHTML='<b>V</b>';
        b.style.display = 'block';
    } else {
        h.innerHTML='<b>&gt;</b>';
        b.style.display = 'none';
    }
    h.style.width='14px';
}

function child2html( re ) {

    // Initialize and register identifiers for group operations like hiding/expanding
    // all messages.
    //
    var mid = 'message_'+re.id;
    var hid = 'message_header_'+re.id;
    var bid = 'message_body_'+re.id;
    var lid = 'message_link_dialog_'+re.id;
    var rid = 'message_reply_dialog_'+re.id;
    var eid = 'message_edit_dialog_'+re.id;
    var did = 'message_delete_dialog_'+re.id;

    var run = re.run == '' ? '' : 'run: <em style="padding:2px;">'+re.run+'</em>';
    var shift = re.shift == '' ? '' : 'shift: <em style="padding:2px;">'+re.shift+'</em>';
    var run_shift = re.run == '' && re.shift == '' ? '' : ' - '+run+' '+shift;

    var attachment_sign = '<img src="images/attachment.png" height="18" />';
    var attachment_signs = '';
    var attachments='';
    if( re.attachments.length > 0 ) {
        attachments +=
        //'        <div style="padding:10px; background-color:#ffffdd;">';
        '        <div style="padding:10px; padding-left:0px; background-color:#ffffdd;">';
        for( var i=0; i < re.attachments.length; i++ ) {
            attachment_signs += attachment_sign;
            var a = re.attachments[i];

            var ahid = 'attachment_header_id_'+a.id;
            var aid = 'attachment_id_'+a.id;

            attachments +=
                '<div style="margin-top:0px; margin-left:0px;">'+
                '  <span id="'+ahid+'" style="border:solid 1px #c0c0c0; width:14px; height:14px; padding-left:2px; padding-right:2px; margin-right:5px; font-size:14px; text-align:center; cursor:pointer;"'+
                '     onclick="javascript:attachment_expander_toggle('+a.id+')"'+
                '     onmouseover="javascript:expander_highlight(this)" '+
                '     onmouseout="javascript:expander_unhighlight(this,document.bgColor)">'+
                '     <b>&gt;</b>'+'</span>'+
                '  <span><b>'+a.url+'</b></span>'+
                '  ( <span>type:<b> '+a.type+'</b></span> '+
                '  <span>size: <b> '+a.size+'</b></span> )'+
                '  '+attachment_sign+
                '</div>'+
                '<div id="'+aid+'" style="margin-left:0px; padding:17px; display:none;">'+
                '</div>';

            var descr = { 'ahid' : ahid, 'aid' : aid, 'loader' : new AttachmentLoader( a, aid )};
            attachment_descriptors[String(a.id)] = descr;
        }
        attachments +=
        '</div>';
    }

    var child_sign = '';
    if( re.children.length > 0 )
        child_sign = '&nbsp;<sup><b>&crarr;</b></sup>';

    var result =
        '<div id="'+mid+'" style="display:block;" style="margin-top:0px;" >'+
        '  <div style="margin-top:20px; margin-left:0px; padding:2px;">'+
        '    <span id="'+hid+'" style="border:solid 1px #c0c0c0; width:14px; height:14px; padding-left:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '      onclick="child_message_expander_toggle('+hid+','+bid+')"'+
        '      onmouseover="expander_highlight(this)" '+
        '      onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '      title="Open/close the message body">'+
        '      <b>&gt;</b>'+
        '    </span>'+
        '    <span>'+
        '      <b><em style="padding:2px;">'+re.event_time+'</em></b>'+
        '      by: <b><em style="padding:2px;">'+re.author+'</em></b>'+
        '      - <em class="lb_msg_subject" style="padding:2px;">'+re.subject+'</em>'+run_shift+
        '    </span>'+
        attachment_signs+
        child_sign+
        '  </div>'+
        '  <div id="'+bid+'" style="margin-left:7px; margin-bottom:20px;">'+
        '    <div style="padding:10px; padding-left:20px; padding-bottom:0px;  padding-right:0px;">'+
        '      <div style="float:right; position:relative; top:-16px;">';
    result +=
        '        <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '          onclick="create_message_link_dialog('+lid+','+re.id+')"'+
        '          onmouseover="expander_highlight(this)" '+
        '          onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '          title="Show a persistent URL for this message. The link can be embedded into other documents.">'+
        '          Link'+
        '        </span>';
    if( isAuthorizedFor( 'canPostNewMessages' )) {
        result +=
        '        <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '          onclick="create_message_reply_dialog('+rid+','+re.id+')"'+
        '          onmouseover="expander_highlight(this)" '+
        '          onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '          title="Open a dialog to reply to the message">'+
        '          Reply'+
        '        </span>';
    }
    if( isAuthorizedFor( 'canEditMessages' )) {
        result +=
        '        <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '          onclick="create_message_edit_dialog('+eid+','+re.id+')"'+
        '          onmouseover="expander_highlight(this)" '+
        '          onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '          title="Open a dialog to edit the message text">'+
        '          Edit'+
        '        </span>';
    }
    if( isAuthorizedFor( 'canDeleteMessages' )) {
        result +=
        '        <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '          onclick="create_message_delete_dialog('+did+','+re.id+')"'+
        '          onmouseover="expander_highlight(this)" '+
        '          onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '          title="Delete the whole message and its children">'+
        '          Delete'+
        '        </span>';
    }
    result +=
        '      </div>'+
        '      <div style="margin-top:0px; margin-right:0px; margin-bottom:0px; background-color:#efefef;">'+re.html+'</div>'+
        '      <div id="'+lid+'" style="border-top:dashed 1px #000000; padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        '      <div id="'+rid+'" style="border-top:dashed 1px #000000; padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        '      <div id="'+eid+'" style="border-top:dashed 1px #000000; padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        '      <div id="'+did+'" style="padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        attachments+
        '      <div style="padding-left:20px;">';

    if( re.children.length > 0 )
        for( var i = re.children.length-1; i >= 0 ; i-- ) {
            var child = eval( "("+re.children[i]+")" );
            result += child2html( Object( child ));
        }

    result +=
        '      </div>'+
        '    </div>'+
        '  </div>'+
        '</div>';
    return result;
}

function event2html( message_idx ) {

    var re = last_search_result[message_idx];

    // Initialize and register identifiers for group operations like hiding/expanding
    // all messages.
    //
    var mid = 'message_'+re.id;
    var hid = 'message_header_'+re.id;
    var bid = 'message_body_'+re.id;
    var lid = 'message_link_dialog_'+re.id;
    var rid = 'message_reply_dialog_'+re.id;
    var eid = 'message_edit_dialog_'+re.id;
    var did = 'message_delete_dialog_'+re.id;

    var run = re.run == '' ? '' : 'run: <em style="padding:2px;">'+re.run+'</em>';
    var shift = re.shift == '' ? '' : 'shift: <em style="padding:2px;">'+re.shift+'</em>';
    var run_shift = re.run == '' && re.shift == '' ? '' : ' - '+run+' '+shift;

    var tag_sign = '';
    var tags='';
    if( re.tags.length > 0 ) {
        tag_sign = '&nbsp;<sup><b>T</b></sup>';
        tags +=
        '<div style="padding:2px; background-color:#ddffff;">';
        for( var i=0; i < re.tags.length; i++ ) {
            var t = re.tags[i];
            tags +=
                '<div style="margin-top:4px; margin-left:0px;">'+
                '  <span style="margin-left:4px;"><b>'+t.tag+'</b>=<i>'+t.value+'</i></span>'+
                '</div>';
        }
        tags +=
        '</div>';
    }

    var attachment_sign = '<img src="images/attachment.png" height="18" />';
    var attachment_signs = '';
    var attachments='';
    if( re.attachments.length > 0 ) {
        attachments +=
        //'        <div style="padding:10px; background-color:#ffffdd;">';
        '        <div style="padding:10px; padding-left:0px; background-color:#ffffdd;">';
        for( var i=0; i < re.attachments.length; i++ ) {
            attachment_signs += attachment_sign;
            var a = re.attachments[i];

            var ahid = 'attachment_header_id_'+a.id;
            var aid = 'attachment_id_'+a.id;

            attachments +=
                '<div style="margin-top:0px; margin-left:0px;">'+
                '  <span id="'+ahid+'" style="border:solid 1px #c0c0c0; width:14px; height:14px; padding-left:2px; padding-right:2px; margin-right:5px; font-size:14px; text-align:center; cursor:pointer;"'+
                '     onclick="javascript:attachment_expander_toggle('+a.id+')"'+
                '     onmouseover="javascript:expander_highlight(this)" '+
                '     onmouseout="javascript:expander_unhighlight(this,document.bgColor)">'+
                '     <b>&gt;</b>'+'</span>'+
                '  <span><b>'+a.url+'</b></span>'+
                '  ( <span>type:<b> '+a.type+'</b></span> '+
                '  <span>size: <b> '+a.size+'</b></span> )'+
                '  '+attachment_sign+
                '</div>'+
                '<div id="'+aid+'" style="margin-left:0px; padding:17px; display:none;">'+
                '</div>';

            var descr = { 'ahid' : ahid, 'aid' : aid, 'loader' : new AttachmentLoader( a, aid )};
            attachment_descriptors[String(a.id)] = descr;
        }
        attachments +=
        '</div>';
    }

    var child_sign = '';
    if( re.children.length > 0 )
        child_sign = '&nbsp;<sup><b>&crarr;</sup></b>';

    var result =
        '<div id="'+mid+'" style="display:none;" style="position:relative;" >'+
        '  <div style="margin-top:10px; margin-left:10px; padding:2px;">'+
        '    <span id="'+hid+'" style="border:solid 1px #c0c0c0; width:14px; height:14px; padding-left:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '      onclick="message_expander_toggle('+message_idx+')"'+
        '      onmouseover="expander_highlight(this)" '+
        '      onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '      title="Open/close the message body">'+
        '      <b>&gt;</b>'+
        '    </span>'+
        '    <span>'+
        '      <b><em style="padding:2px;">'+re.event_time+'</em></b>'+
        '      by: <b><em style="padding:2px;">'+re.author+'</em></b>'+
        '      - <em class="lb_msg_subject" style="padding:2px;">'+re.subject+'</em>'+run_shift+
        '    </span>'+
        attachment_signs+
        child_sign+
        tag_sign+
        '  </div>'+
        '  <div id="'+bid+'" style="display:none; margin-left:17px; margin-bottom:20px;">'+
        '    <div style="padding:10px; padding-left:20px; padding-bottom:0px;  padding-right:0px; border-left:solid 1px #c0c0c0;">'+
        '    <div style="float:right; position:relative; top:-8px;">';
    result +=
        '        <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '          onclick="create_message_link_dialog('+lid+','+re.id+')"'+
        '          onmouseover="expander_highlight(this)" '+
        '          onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '          title="Show a persistent URL for this message. The link can be embedded into other documents.">'+
        '          Link'+
        '        </span>';
    if( isAuthorizedFor( 'canPostNewMessages' )) {
        result +=
        '      <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '        onclick="create_message_reply_dialog('+rid+','+re.id+')"'+
        '        onmouseover="expander_highlight(this)" '+
        '        onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '        title="Open a dialog to reply to the message">'+
        '        Reply'+
        '      </span>';
    }
    if( isAuthorizedFor( 'canEditMessages' )) {
        result +=
        '      <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '        onclick="create_message_edit_dialog('+eid+','+re.id+')"'+
        '        onmouseover="expander_highlight(this)" '+
        '        onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '        title="Open a dialog to edit the message text">'+
        '        Edit'+
        '      </span>';
    }
    if( isAuthorizedFor( 'canDeleteMessages' )) {
        result +=
        '      <span style="border:solid 1px #c0c0c0; height:14px; padding-left:2px; padding-right:2px; margin-right:4px; font-size:14px; text-align:center; cursor:pointer;"'+
        '        onclick="create_message_delete_dialog('+did+','+re.id+')"'+
        '        onmouseover="expander_highlight(this)" '+
        '        onmouseout="expander_unhighlight(this,document.bgColor)"'+
        '        title="Delete the whole message and its children">'+
        '        Delete'+
        '      </span>';
    }
    result +=
        '    </div>'+
        '    <div style="margin-top:8px; margin-right:0px; margin-bottom:0px; background-color:#efefef;">'+re.html+'</div>'+
        '    <div id="'+lid+'" style="border-top:dashed 1px #000000; padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        '    <div id="'+rid+'" style="border-top:dashed 1px #000000; padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        '    <div id="'+eid+'" style="border-top:dashed 1px #000000; padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        '    <div id="'+did+'" style="padding:10px; background-color:#e0e0e0; display:none;"></div>'+
        attachments+
        tags+
        '    <div style="padding-left:20px;">';

    if( re.children.length > 0 )
        for( var i = re.children.length-1; i >= 0 ; i-- ) {
            var child = eval( "("+re.children[i]+")" );
            result += child2html( Object( child ));
        }

    result +=
        '      </div>'+
        '    </div>'+
        '  </div>'+
        '</div>';
    return result;
}

var last_search_result = null;
var limit_per_page = null;
var first_message_idx = null;
var last_message_idx = null;

function display_messages_page() {

    document.getElementById('messages_total').innerHTML = last_search_result.length;
    document.getElementById('messages_showing_from_id').innerHTML = last_search_result.length - last_message_idx + 1;
    document.getElementById('messages_showing_through_id').innerHTML = last_search_result.length - first_message_idx;

    last_message_page_button.set( 'disabled', first_message_idx == 0 );
    next_message_page_button.set( 'disabled', first_message_idx == 0 );
    prev_message_page_button.set( 'disabled', last_message_idx >= last_search_result.length );
    first_message_page_button.set( 'disabled', last_message_idx >= last_search_result.length );

    for( var i = 0; i < last_search_result.length; i++ ) {
        var re = last_search_result[i];
        var mid = 'message_'+re.id;
        var m = document.getElementById(mid);

        // Show the messages from the current page and hide the others
        //
        if(( first_message_idx <= i ) && ( i < last_message_idx )) {
            m.style.display = 'block';
        } else {
            m.style.display = 'none';
        }
    }
}


var first_message_page_button = null;
function first_message_page() {
    first_message_idx = Math.max( 0, last_search_result.length-limit_per_page );
    last_message_idx  = last_search_result.length;
    display_messages_page();
}

var next_message_page_button = null;
function next_message_page() {
    last_message_idx  = first_message_idx;
    first_message_idx = Math.max( 0, first_message_idx - limit_per_page );
    display_messages_page();
}

var prev_message_page_button = null;
function prev_message_page() {
    first_message_idx = last_message_idx;
    last_message_idx  = Math.min( last_search_result.length, last_message_idx + limit_per_page );
    display_messages_page();
}

var last_message_page_button = null;
function last_message_page() {
    first_message_idx = 0;
    if( limit_per_page == last_search_result.length ) {
        last_message_idx = last_search_result.length;
    } else {
        last_message_idx = last_search_result.length % limit_per_page;
        if( last_message_idx == 0 )
            last_message_idx = limit_per_page;
    }
    display_messages_page();
}

function display_messages() {
    if( last_search_result == null ) {
        alert( 'display_messages(): application logic error - no messages downloaded' );
        return;
    }
    var limit_per_page_str = document.search_display_form.limit_per_page.options[document.search_display_form.limit_per_page.selectedIndex].value;
    limit_per_page = limit_per_page_str == 'all' ? last_search_result.length : Number( limit_per_page_str );

    first_message_idx = Math.max( 0, last_search_result.length-limit_per_page );
    last_message_idx  = last_search_result.length;

    display_messages_page();
}

var new_message_button=null;

var messages_refresh_timer = null;
function scheduleNexRefreshOfMessagesTable() {
    messages_refresh_timer = window.setTimeout( 'refreshMessagesTable()', 5000 );
}
function stopRefreshOfMessagesTable() {
    if( messages_refresh_timer != null ) {
        window.clearTimeout( messages_refresh_timer );
        messages_refresh_timer = null;
    }
}
function refreshMessagesTable() {
    if( current_messages_table != null ) {
        current_messages_table.refresh();
    }
}

function auto_refresh_togle( e ) {
    if( e.checked ) scheduleNexRefreshOfMessagesTable();
    else stopRefreshOfMessagesTable();
}

var last_display_scope,
    last_display_text2search,
    last_display_search_in_messages,
    last_display_search_in_tags,
    last_display_search_in_values,
    last_display_posted_at_experiment,
    last_display_posted_at_shifts,
    last_display_posted_at_runs,
    last_display_begin,
    last_display_end,
    last_display_tag,
    last_display_author,
    last_display_html_new_message,
    last_display_auto_refresh,
    last_limit_per_view;

function re_display_messages_table() {
	last_limit_per_view = document.search_display_form.limit_per_view.options[document.search_display_form.limit_per_view.selectedIndex].value;
    display_messages_table(
        last_display_scope,
        last_display_text2search,
        last_display_search_in_messages,
        last_display_search_in_tags,
        last_display_search_in_values,
        last_display_posted_at_experiment,
        last_display_posted_at_shifts,
        last_display_posted_at_runs,
        last_display_begin,
        last_display_end,
        last_display_tag,
        last_display_author,
        last_display_html_new_message,
        last_display_auto_refresh,
        last_limit_per_view );
}

function display_messages_table(
    scope,
    text2search,
    search_in_messages,
    search_in_tags,
    search_in_values,
    posted_at_experiment,
    posted_at_shifts,
    posted_at_runs,
    begin,
    end,
    tag,
    author,
    html_new_message,
    auto_refresh,
    limit_per_view ) {

    stopRefreshOfMessagesTable();

    last_display_scope = scope;
    last_display_text2search = text2search;
    last_display_search_in_messages = search_in_messages;
    last_display_search_in_tags = search_in_tags;
    last_display_search_in_values = search_in_values;
    last_display_posted_at_experiment = posted_at_experiment;
    last_display_posted_at_shifts = posted_at_shifts;
    last_display_posted_at_runs = posted_at_runs,
    last_display_begin = begin;
    last_display_end = end;
    last_display_tag = tag;
    last_display_author = author;
    last_display_html_new_message = html_new_message;
    last_display_auto_refresh = auto_refresh;
    last_limit_per_view = limit_per_view;

    this.url='Search.php?id='+current_selection.experiment.id+
        (scope == '' ? '' : '&'+scope)+
        '&format=detailed'+
        '&text2search='+encodeURIComponent(text2search)+
        '&search_in_messages='+(search_in_messages ? '1' : '0')+
        '&search_in_tags='+(search_in_tags ? '1' : '0')+
        '&search_in_values='+(search_in_values ? '1' : '0')+
        '&posted_at_experiment='+(posted_at_experiment ? '1' : '0')+
        '&posted_at_shifts='+(posted_at_shifts ? '1' : '0')+
        '&posted_at_runs='+(posted_at_runs ? '1' : '0')+
        '&begin='+encodeURIComponent(begin)+
        '&end='+encodeURIComponent(end)+
        '&tag='+encodeURIComponent(tag)+
        '&author='+encodeURIComponent(author)+
        (limit_per_view == null ? '' : '&limit='+limit_per_view);

    var html=
        '<div>'+
        '  <img src="images/OCSMessages.png" />'+
        '</div>'+
        '<div style="margin-left:10px;">';
    if( html_new_message != '' ) {
        html +=
        html_new_message+
        '  <div style="margin-top:10px; padding-bottom:20px;">';
    } else {
        html +=
        '  <div style="margin-top:10px; margin-bottom:10px; padding-bottom:10px;">';
    }
    var auto_refresh_checked = auto_refresh ? 'checked="checked"' : '';
    html +=
        '  <form name="search_display_form">'+
        '  <div style="text-align:right; margin-bottom:5px;">'+
        '    <b><em id="messages_showing_from_id">0</em></b> - <b><em id="messages_showing_through_id">0</em></b> of <b><em id="messages_total">0</em></b> ';
    if( limit_per_view != null )
    html +=
        '       <select align="center" type="text" name="limit_per_view" style="padding:1px;" onchange="re_display_messages_table()">'+
        '         <option value="all"'+(limit_per_view=='all'?' selected="selected"':'')+'>all known threads</option>'+
        '         <option value="25"' +(limit_per_view=='25' ?' selected="selected"':'')+'>latest threads only</option>'+
        '       </select>'+
        '       '+(limit_per_view=='all'?'':' <b style="color:red;">!!!</b>');
    html +=
        '  </div>'+
        '  <div style="margin-left:10px; margin-bottom:10px; padding-top:10px; padding-bottom:10px; border-top:solid 2px #e0e0e0; border-bottom:solid 2px #e0e0e0;">'+
        '    <form name="search_display_form">'+
        '      <table><tbody>'+
        '        <tr>'+
        '          <td valign="top">'+
        '            <button id="expand_button">Expand</button>'+
        '            <button id="collapse_button">Collapse</button>'+
        '              <button id="show_attachments_button">View Attach</button>'+
        '              <button id="hide_attachments_button">Hide Attach</button>'+
        '              <button id="new_message_button"><em style="font-weight:bold;">New Message</em></button>'+
        '          </td>'+
        '          <td valign="top">'+
        '            <div style="padding-left:40px;">'+
        '              <select align="center" type="text" name="limit_per_page" style="padding:1px;" onchange="display_messages()">'+
        '                <option value="all">all</option>'+
        '                <option value="5">5</option>'+
        '                <option value="10">10</option>'+
        '                <option value="20">20</option>'+
        '                <option value="50">50</option>'+
        '                <option value="100">100</option>'+
        '              </select> / page'+
        '            </div>'+
        '            <div style="padding-left:40px;padding-top:10px;">'+
        '              <input type="checkbox" name="autorefresh" value="Autorefresh" '+auto_refresh_checked+' onchange="auto_refresh_togle(this)" />&nbsp;auto refresh'+
        '            </div>'+
        '          </td>'+
        '          <td valign="top">'+
        '            <div style="padding-left:20px;">'+
        '              <button id="first_message_page_button">&lt;&lt; First</button>'+
        '              <button id="prev_message_page_button">&lt; Prev</button>'+
        '              <button id="next_message_page_button">Next &gt;</button>'+
        '              <button id="last_message_page_button">Last &gt;&gt;</button>'+
        '            </div>'+
        '          </td>'+
        '        </tr>'+
        '      </tbody></table></center>'+
        '  </div>'+
        '  </form>'+
        '  <div id="messages_area">'+
        '    <img src="images/ajaxloader.gif" />&nbsp;searching messages...'+
        '  </div>'+
        '</div>';

    document.getElementById('messagesarea').innerHTML=html;

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
    // ATTENTION: this button is made global to allow re-enabling it from
    // the dialog window when it gets closed.
    //
    new_message_button = new YAHOO.widget.Button( "new_message_button" );
    new_message_button.on (
        "click",
        function( p_oEvent ) {
            document.getElementById('new_message_dialog').style.display='block';
            new_message_button.set( 'disabled', true );
        }
    );
    if( html_new_message == '' ) {
        new_message_button.set( 'disabled', true );
    }

    first_message_page_button = new YAHOO.widget.Button( "first_message_page_button" );
    first_message_page_button.on (
        "click",
        function( p_oEvent ) {
            first_message_page();
        }
    );
    //first_message_page_button.set( 'disabled', true );

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
                
    last_message_page_button = new YAHOO.widget.Button( "last_message_page_button" );
    last_message_page_button.on (
        "click",
        function( p_oEvent ) {
            last_message_page();
        }
    );
    //last_message_page_button.set( 'disabled', true );

    function callback_on_load( result ) {
        if( result.ResultSet.Status != "success" ) {
            document.getElementById('messages_area').innerHTML = result.ResultSet.Message;
        } else {
            last_search_result = result.ResultSet.Result;
            var html1 = '';
            for( var i=last_search_result.length-1; i >= 0 ; i-- )
                html1 += event2html(i);
            document.getElementById('messages_area').innerHTML = html1;
            display_messages();
            if( document.search_display_form.autorefresh.checked )
                scheduleNexRefreshOfMessagesTable();
        }
    }
    function callback_on_failure( http_status ) {
        document.getElementById('messages_area').innerHTML=
            '<b><em style="color:red;" >Error</em></b>&nbsp;Request failed. HTTP status: '+http_status;
    }
    load_then_call( this.url, callback_on_load, callback_on_failure );

    function callback_on_refresh( result ) {
        if( result.ResultSet.Status != "success" ) {
            document.getElementById('messages_area').innerHTML = result.ResultSet.Message;
        } else {
            var this_search_result = result.ResultSet.Result;
            if( this_search_result.length > 0 ) {
                var html1 = '';
                for( var i=0; i < this_search_result.length; i++ ) {
                    last_search_result.push(this_search_result[i])
                    html1 += event2html( last_search_result.length-1 );
                }
                var html_old = document.getElementById('messages_area').innerHTML;
                document.getElementById('messages_area').innerHTML = html1 + html_old;
                display_messages();
            }
            if( document.search_display_form.autorefresh.checked )
                scheduleNexRefreshOfMessagesTable();
        }
    }
    this.refresh = function() {
        if( last_search_result.length > 0 )
            load_then_call(
                this.url+'&since='+encodeURIComponent(last_search_result[last_search_result.length-1].event_time),
                callback_on_refresh,
                callback_on_failure );
        else
            load_then_call(
                this.url,
                callback_on_load,
                callback_on_failure );
    }
    return this;
}

function search_and_display() {

    var scope='';
    var tag = document.search_form.tag.options[document.search_form.tag.selectedIndex].value;
    var author = document.search_form.author.options[document.search_form.author.selectedIndex].value;

    display_messages_table(
        scope,
        document.search_form.text2search.value,
        document.search_form.search_in_messages.checked,
        document.search_form.search_in_tags.checked,
        document.search_form.search_in_values.checked,
        document.search_form.posted_at_experiment.checked,
        document.search_form.posted_at_shifts.checked,
        document.search_form.posted_at_runs.checked,
        document.search_form.begin.value,
        document.search_form.end.value,
        tag,
        author,
        '',
        false );
}

function search_contents() {

    set_context (
        '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
        'Find Messages >' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    workarea.innerHTML=
        '<div id="messagesarea">'+
        '  <div style="margin-bottom:8px; padding:2px; background-color:#e0e0e0;">'+
        '    <center><b>How to use the tool</b></center>'+
        '  </div>'+
        '  <div id="search_instructions" style="margin:10px;"></div>'+
        '</div>';

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    navarea.innerHTML=
        '<div style="margin-bottom:20px;">'+
        '  <img src="images/Filter.png" />'+
        '</div>'+
        '<form name="search_form" action="javascript:search_and_display()">'+
        '  <div id="search_form_params"></div>'+
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

function subscribe() {
    set_context (
            '<a href="javascript:display_experiment()">'+experiment_or_facility( current_selection.experiment.id )+'</a> > '+
            'Manage Subscriptions >' );

    reset_navarea();
    reset_workarea();

    document.getElementById('workarea').innerHTML=
        '<div class="help_area">'+
        '  <p class="help">The mechanism of subscriptions allows registered users to recieve instant e-mail notifications'+
        '  on any new messages posted in a context of the experiment.</p>'+
        '  <p class="help"><b>Who can subscribe:</b> The subscriptions can be made by any user who has read'+
        '  privileges in the experiment. Logged LogBook users can register as many e-mail addresses of any'+
        '  recipients as they want.</p>'+
        '  <p class="help"><b>Message delivery:</b> The notification service will make its best effort'+
        '  to deliver (send) notifications on all relevant updates to recipients who were registered at'+
        '  a time the updates were posted. However the service can <b>NOT</b> guarantee that the notifications'+
        '  will be actiually delivered to addressees. It is up to the users to ensure that specified e-mail'+
        '  addresses are correct. Also keep in mind that some anti-spam filters may block these messages.</p>'+
        '  <p class="help"><b>Register/unregister:</b> Make required modifications and press the <b>Save</b> button'+
        '  Upon a successful registration a greeting message will be sent to each new recipient.'+
        '  A similar message will be sent to an addressee after their e-mail address will be unregistered.</p>'+
        '</div>'+
        '<br />'+
        '<div style="margin-left:20px;">'+
        '  <button id="subscribers_save_button">Save</button>'+
        '  <button id="subscribers_cancel_button" style="margin-lefty:10px;">Cancel</button>'+
        '  <div style="margin-top:10px; padding:20px; background-color:#e0e0e0;">'+
        '    <div>'+
        '      <div id="subscribers_table">'+
        '        <div id="subscribers_table_paginator"></div>'+
        '        <div id="subscribers_table_body"></div>'+
        '      </div>'+
        '      <div style="margin-top:8px;" align="left" >'+
        '        <button id="subscribers_add_button">Add Entry</button>'+
        '        <button id="subscribers_update_button">Delete Selected</button>'+
        '      </div>'+
        '    </div>'+
        '  </div>'+
        '</div>'+
        '<form name="manage_subscriptions_form" action="UpdateSubscriptions.php" method="post">'+
        '  <input type="hidden" name="exper_id" value="'+current_selection.experiment.id+'" />'+
        '  <input type="hidden" name="actionSuccess" value="select_experiment" />'+
        '  <input type="hidden" name="subscriptions" value="" />'+
        '</form>';

    var subscribers = null;

    var url = 'RequestSubscriptions.php?exper_id='+current_selection.experiment.id+'&subscribed_by='+auth_remote_user;
 
    function create_table() {
        if( subscribers == null )
            return new Table (
                "subscribers_table",
                [ { key: "delete", formatter: "checkbox" },
                  { key: "address",   sortable: true, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true})},
                  { key: "subscriber", sortable: true, resizeable: true },
                  { key: "subscribed_time", sortable: true, resizeable: true },
                  { key: "subscribed_host", sortable: true, resizeable: true } ],
                url,
                false,
                10
            );
        else
            return new TableLocal (
                "subscribers_table",
                [ { key: "delete", formatter: "checkbox" },
                  { key: "address",   sortable: true, resizeable: true,
                    editor: new YAHOO.widget.TextboxCellEditor({disableBtns:true})},
                  { key: "subscriber", sortable: true, resizeable: true },
                  { key: "subscribed_time", sortable: true, resizeable: true },
                  { key: "subscribed_host", sortable: true, resizeable: true } ],
                subscribers,
                false
            );
    }
    var subscribers_table = create_table();

    function AddAndRefreshSubscribersTable() {
        subscribers_table.dataTable.addRow (
            { "address": auth_remote_user+'@slac.stanford.edu',
              "subscriber" : auth_remote_user,
              "subscribed_time" : '',
              "subscribed_host" : '' },
            0 );
    }
    function synchronize_subscribers() {
        var rs = subscribers_table.dataTable.getRecordSet();
        var rs_length = rs.getLength();
        subscribers = [];
        for( var i = 0; i < rs_length; i++ ) {
            var r = rs.getRecord(i);
            if( !r.getData('delete'))
                subscribers.push ( {
                    "address": r.getData("address"),
                    "subscriber" : r.getData("subscriber"),
                    "subscribed_time" : r.getData("subscribed_time"),
                    "subscribed_host" : r.getData("subscribed_host")} );
        }
    }
    function toJSON() {
        var result = [];
        synchronize_subscribers();
        if( subscribers != null) {
            for( var i = 0; i < subscribers.length; i++ ) {
                var r = subscribers[i];
                result.push ( r["address"] );
            }
        } else {
            // Pass this special string to indicate that we don't have any
            // information on subscribers.
            //
            return 'null';
        }
        return JSON.stringify( result );
    };

    var subscribers_save_button = new YAHOO.widget.Button( "subscribers_save_button" );
    subscribers_save_button.on(
        "click",
        function( p_oEvent ) {
            document.manage_subscriptions_form.subscriptions.value = toJSON();
            document.manage_subscriptions_form.submit();
        }
    );

    var subscribers_cancel_button = new YAHOO.widget.Button( "subscribers_cancel_button" );
    subscribers_cancel_button.on(
        "click",
        function( p_oEvent ) { display_experiment(); }
    );

    var subscribers_add_button = new YAHOO.widget.Button( "subscribers_add_button" );
    subscribers_add_button.on (
        "click",
        function( p_oEvent ) { AddAndRefreshSubscribersTable(); }
    );

    var subscribers_update_button = new YAHOO.widget.Button( "subscribers_update_button" );
    subscribers_update_button.on (
        "click",
        function( p_oEvent ) {
            synchronize_subscribers();
            subscribers_table.dataTable.destroy();
            subscribers_table = create_table();
        }
    );
}

    </script>
</head>
<body class="yui-skin-sam" id="body" onload="init()">
<div id="application_header">
<div>
<div style="float: left;">
<p id="application_title" style="text-align: left;"><em>Electronic
LogBook: </em> <em id="application_subtitle"></em> <!--
            <em id="application_subtitle"><a href="javascript:list_experiments()">select &gt;</a></em>
            --></p>
</div>
<div style="float: right; height: 50px;"><?php
if( $auth_svc->authName() == '' ) {
    echo <<<HERE
          <br>
          <br>
          <a href="../../apps/logbook"><p title="login here to proceed to the full version of the application">login</p></a>
HERE;
} else {
    $logout = $auth_svc->authType() == 'WebAuth' ? '[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]' : '&nbsp;';
    echo <<<HERE
          <table><tbody>
            <tr>
              <td></td>
              <td>{$logout}</td>
            </tr>
            <tr>
              <td>Welcome,&nbsp;</td>
              <td><p><b>{$auth_svc->authName()}</b></p></td>
            </tr>
            <tr>
              <td>Session expires in:&nbsp;</td>
              <td><p id="auth_expiration_info"><b>00:00.00</b></p></td>
            </tr>
          </tbody></table>
HERE;
}
?></div>
<div style="height: 40px;">&nbsp;</div>
</div>
</div>
<div id="menubar" class="yuimenubar yuimenubarnav"></div>
<p id="context"></p>
<br>
<div id="popupdialogs"></div>
<div id="nav-and-work-areas" align="left">
<table>
    <tbody>
        <tr>
            <td valign="top"><!--
              Optional navigation tools (menus, trees, etc.) can be placed
              below.
              -->
            <div id="navarea"></div>
            </td>
            <td valign="top"><!--
              Here comes the main viwing area of the application.
              -->
            <div id="workarea"></div>
            </td>
        </tr>
    </tbody>
</table>
</div>
</body>
</html>

<?php

$logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>
