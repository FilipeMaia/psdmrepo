<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title>File Explorer</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">

<!--
Standard reset, fonts and grids
-->
<link rel="stylesheet" type="text/css" href="/yui/build/reset-fonts-grids/reset-fonts-grids.css" />

<!--
CSS for YUI
-->
<link rel="stylesheet" type="text/css" href="/yui/build/fonts/fonts-min.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/menu/assets/skins/sam/menu.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/paginator/assets/skins/sam/paginator.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/datatable/assets/skins/sam/datatable.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/button/assets/skins/sam/button.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/container/assets/skins/sam/container.css" />
<link rel="stylesheet" type="text/css" href="/yui/build/treeview/assets/skins/sam/treeview.css" />
<!--
<link rel="stylesheet" type="text/css" href="/yui/examples/treeview/assets/css/folders/tree.css" />
-->

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
        margin-left:20px;
        margin-top:20px;
        font-size:16px;
        text-align:left;
    }
    #nav-and-work-areas {
        margin-top:30px;
        margin-bottom:10px;
        margin-left:10px;
        width:100%;
    }
    #navarea {
        padding:20px;
        padding-top:0px;
        overflow:auto;
        display:none;
        min-width:250px;
        float:left;
    }
    #workarea {
        padding:20px;
        padding-top:0px;
        float:left;
    }

    .first_col_hdr {
      padding-left:5px;
      padding-bottom:5px;
      padding-right:5px;
      font-weight:bold;
      text-align:righ;
    }
    .col_hdr {
      padding-bottom:5px;
      padding-left:15px;
      padding-right:5px;
      font-weight:bold;
      text-align:left;
    }
    .col_hdr_right {
      padding-bottom:5px;
      padding-left:15px;
      padding-right:5px;
      font-weight:bold;
      text-align:right;
    }

    .first_separator {
      height:5px;
      border-top:solid 1px #c0c0c0;
    }
    .separator {
      height:5px;
      margin-left:10px;
      border-top:solid 1px #c0c0c0;
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
<script type="text/javascript" src="/yui/build/button/button-min.js"></script>

<script type="text/javascript" src="/yui/build/yahoo/yahoo-min.js"></script>
<script type="text/javascript" src="/yui/build/dom/dom-min.js"></script>
<script type="text/javascript" src="/yui/build/treeview/treeview-min.js"></script>

<!--
Custom JavaScript
-->
<script type="text/javascript" src="js/Menubar.js"></script>
<script type="text/javascript" src="js/Dialogs.js"></script>
<script type="text/javascript" src="js/JSON.js"></script>
<script type="text/javascript" src="../webfwk/js/Loader.js"></script>
<script type="text/javascript" src="../webfwk/js/Utilities.js"></script>


<!--
PHP Generated JavaScript with initialization parameters
-->
<script type="text/javascript">

<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

try {
    $auth_svc = AuthDB::instance();
    
    echo <<<HERE

/* Authentication and authorization context
 */
var auth_type="{$auth_svc->authType()}";
var auth_remote_user="{$auth_svc->authName()}";
 
var auth_webauth_token_creation="{$_SERVER['WEBAUTH_TOKEN_CREATION']}";
var auth_webauth_token_expiration="{$_SERVER['WEBAUTH_TOKEN_EXPIRATION']}";

var auth_granted = {};

function refresh_page() {
    window.location = "{$_SERVER['REQUEST_URI']}";
}

HERE;

// Initial action dispatcher's generator
//
echo <<<HERE

function init() {
  load( 'help/Welcome.html', 'workarea' );;
  auth_timer_restart();

}

HERE;
?>

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
        { text: "Electronic Log Book", url: "../logbook/" },
        { text: "File Explorer", url: "../explorer/" } ],
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
var menubar_group_browse = menubar_data.length;
menubar_data.push ( {
    id:    'browse',
    href:  '#browse',
    title: 'Browse',
    title_style: null,
    itemdata: [
        { text: "Data files of experiments",     url: "javascript:browse_experiments()", disabled: false },
        { text: "iRODS (file manager) catalogs", url: "javascript:browse_catalogs()", disabled: false } ],
    disabled: false }
);
var menubar_group_manage = menubar_data.length;
menubar_data.push ( {
    id:    'manage',
    href:  '#manage',
    title: 'Manage',
    title_style: null,
    itemdata: [
        { text: "HDF5 translation",     url: "javascript:manage_hdf5_translation()", disabled: false } ],
    disabled: false }
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

function set_subtitle( subtitle ) {
    //document.getElementById('application_subtitle').innerHTML = subtitle;
}

function set_context( context ) {
    document.getElementById('context').innerHTML = context;
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

/* Directory browser and relevant operations.
 *
 * NOTE: Ideally these would need to be declared as 'const'. Unfortunatelly,
 * ECMAScript won't support this (the code won't work on MS Explorer). Only
 * Mozilla family of browsers will.
 */
var BROWSE_INSTRUMENTS = 1,
    BROWSE_EXPERIMENTS = 2,
    BROWSE_FILES       = 3;

var instrument = null;
var experiment = null;
var path = null;

var browse_tree = null;

function browse_experiments() {

	set_subtitle( 'Data Files of Experimens' );
    set_context( 'Select Experiment > ' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    workarea.style.borderLeft = 'solid 1px #c0c0c0';
    workarea.innerHTML=
        '<div>'+
        '  <div style="float:left; width:430px;">'+
        '    <img src="images/ExpSummary.png" />'+
        '    <div id="experiment_summary" style="margin-top:20px; padding-left:20px; width:430px; height:180px;"></div>'+
        '  </div>'+
        '  <div style="float:left; width:400px; padding-left:80px; padding-top:60px;">'+
        '    <img src="images/Filter.png" />'+
        '    <div id="filter" style="margin-top:20px; padding-left:20px; width:350px; height:140px;"></div>'+
        '  </div>'+
        '  <div style="clear:both;">'+
        '  </div>'+
        '</div>'+
        '<div style="margin-top:30px;">'+
        '  <img src="images/Files.png" />'+
        '  <div style="margin-top:20px; padding-left:20px;">'+
        '    <table><thead>'+
        '      <tr>'+
        '        <td style="width: 35px;" class="first_col_hdr">Run</td>'+
        '        <td style="width:250px;" class="col_hdr"><b>File Name</b></td>'+
        '        <td style="width: 50px;" class="col_hdr"><b>Type</b></td>'+
        '        <td style="width:135px;" class="col_hdr_right"><b>Size</b></td>'+
        '        <td style="width:150px;" class="col_hdr"><b>Created</b></td>'+
        '        <td style="width: 60px;" class="col_hdr"><b>Archived</b></td>'+
        '        <td style="width: 40px;" class="col_hdr"><b>Disk</b></td>'+
        '      </tr>'+
        '      <tr>'+
        '        <td><div class="first_separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '      </tr>'+
        '    </thead></table>'+
        '    <br>'+
        '    <div id="experiment_files" style="width:870px; height:340px; overflow:auto;"></div>'+
        '  </div>'+
        '</div>';

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    navarea.innerHTML=
        '<img src="images/Experiment.png" />'+
        '<div id="browse_tree" style="margin-top:20px; padding-left:20px;"></div>';

    browse_tree = new YAHOO.widget.TreeView( "browse_tree" );

    // The whole tree will be built dynamically
    //
    var root_node = new YAHOO.widget.TextNode(
        {   label:    'Instruments/Experiments',
            expanded: false,
            title:    'Expand and select a folder',
            type:     BROWSE_INSTRUMENTS },
        browse_tree.getRoot());

    var currentIconMode = 0;
    root_node.setDynamicLoad( loadNodeData, currentIconMode );
    root_node.toggle();     // Force the node to be instantly open. this will also
                            // trigger the dynamic loading of its children (if any).

    browse_tree.subscribe( "labelClick", onNodeSelection );
    browse_tree.subscribe( "enterKeyPressed", onNodeSelection );
    browse_tree.draw();

    // Dispatch clicks on selected nodes to the corresponding
    // functions.
    //
    function onNodeSelection( node ) {
        if(        node.data.type == BROWSE_INSTRUMENTS) { ;
        } else if( node.data.type == BROWSE_EXPERIMENTS) { ;
        } else if( node.data.type == BROWSE_FILES   ) {
            display_experiment_files(
                node.data.instrument, node.data.experiment, node.data.exper_id );
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
        var sUrl = "../explorer/ws/RequestExperiments.php?type="+node.data.type;
        switch( node.data.type ) {
            case BROWSE_FILES:
                sUrl += '&path='+node.data.path;
            case BROWSE_EXPERIMENTS:
                sUrl += '&instrument='+node.data.instrument;
            case BROWSE_INSTRUMENTS:
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

function apply_filter( exper_id ) {

    var types = '';
    if( document.filter_form.xtc.checked ) types = '&types=xtc';
    if( document.filter_form.hdf5.checked ) {
        if( types == '' ) types = '&types=hdf5';
        else types += ',hdf5';
    }

    var runs = '';
    for( var i=0; i < document.filter_form.runs.length; i++ ) {
        if( document.filter_form.runs[i].checked && document.filter_form.runs[i].value == 'range' ) {
            runs = '&runs='+document.filter_form.runs_range.value;
            break;
        }
    }

    var archived = '';
    for( var i=0; i < document.filter_form.archived.length; i++ ) {
        if( document.filter_form.archived[i].checked ) {
            if( document.filter_form.archived[i].value == 'yes' ) {
                archived = '&archived=1';
            } else if( document.filter_form.archived[i].value == 'no' ) {
            	archived = '&archived=0';
            }
            break;
        }
    }

    var local = '';
    for( var i=0; i < document.filter_form.local.length; i++ ) {
        if( document.filter_form.local[i].checked ) {
            if( document.filter_form.local[i].value == 'yes' ) {
            	local = '&local=1';
            } else if( document.filter_form.local[i].value == 'no' ) {
            	local = '&local=0';
            }
            break;
        }
    }

    var url = '../explorer/ws/RequestExperimentFiles.php?exper_id='+exper_id + types + runs + archived + local;

    document.getElementById('experiment_files').innerHTML='Updating...';

    load( url, 'experiment_files' );
}

function import_list( exper_id ) {

    var types = '';
    if( document.filter_form.xtc.checked ) types = '&types=xtc';
    if( document.filter_form.hdf5.checked ) {
        if( types == '' ) types = '&types=hdf5';
        else types += ',hdf5';
    }

    var runs = '';
    for( var i=0; i < document.filter_form.runs.length; i++ ) {
        if( document.filter_form.runs[i].checked && document.filter_form.runs[i].value == 'range' ) {
            runs = '&runs='+document.filter_form.runs_range.value;
            break;
        }
    }

    var archived = '';
    for( var i=0; i < document.filter_form.archived.length; i++ ) {
        if( document.filter_form.archived[i].checked ) {
            if( document.filter_form.archived[i].value == 'yes' ) {
                archived = '&archived=1';
            } else if( document.filter_form.archived[i].value == 'no' ) {
            	archived = '&archived=0';
            }
            break;
        }
    }

    var local = '';
    for( var i=0; i < document.filter_form.local.length; i++ ) {
        if( document.filter_form.local[i].checked ) {
            if( document.filter_form.local[i].value == 'yes' ) {
            	local = '&local=1';
            } else if( document.filter_form.local[i].value == 'no' ) {
            	local = '&local=0';
            }
            break;
        }
    }
    var url = '../explorer/ws/RequestExperimentFiles.php?exper_id='+exper_id + types + runs + archived + local + '&import';

    window.open( url );
}

function display_experiment_files( instrument, experiment, exper_id ) {

    set_subtitle( 'Data Files of Experiment - <b>'+instrument+' / '+experiment+'</b>' );
    set_context(
        '<a href="javascript:browse_experiments()">Select Experiment</a> &gt; '+
        '<b>'+instrument+' / '+experiment+'</b> [ ID='+exper_id+' ]' );

    document.getElementById('experiment_summary').innerHTML='Loading...';
    document.getElementById('filter').innerHTML=
        '<form name="filter_form" action="javascript:apply_filter('+exper_id+')">'+
        '  <div id="filter_form_params">Loading...</div>'+
        '</form>';
    document.getElementById('experiment_files').innerHTML='Loading...';

    load( '../explorer/ws/ExperimentSummary.php?id='+exper_id, 'experiment_summary' );
    load( '../explorer/ws/Filter.php?exper_id='+exper_id, 'filter_form_params' );
    load( '../explorer/ws/RequestExperimentFiles.php?exper_id='+exper_id, 'experiment_files' );

    YAHOO.util.Event.onContentReady (
        "reset_filter_button",
        function () {
            var default_runs_range = document.filter_form.runs_range.value;
            var reset_filter_button = new YAHOO.widget.Button( "reset_filter_button" );
            reset_filter_button.on (
                "click",
                function( p_oEvent ) {

                	for( var i=0; i < document.filter_form.runs.length; i++ ) document.filter_form.runs[i].checked=!i; 
                    document.filter_form.runs_range.value=default_runs_range;

                    for( var i=0; i < document.filter_form.archived.length; i++ ) document.filter_form.archived[i].checked=!i; 
                    for( var i=0; i < document.filter_form.local.length; i++ ) document.filter_form.local[i].checked=!i; 

                    document.filter_form.xtc.checked=true;
                    document.filter_form.hdf5.checked=true;
                    document.filter_form.other.checked=true;
                }
            );
        }
    );
    YAHOO.util.Event.onContentReady (
        "submit_filter_button",
        function () {
            var submit_filter_button = new YAHOO.widget.Button( "submit_filter_button" );
            submit_filter_button.on (
                "click",
                function( p_oEvent ) {
                    apply_filter( exper_id );
                }
            );
        }
    );
    YAHOO.util.Event.onContentReady (
        "import_list_button",
        function () {
            var import_list_button = new YAHOO.widget.Button( "import_list_button" );
            import_list_button.on (
                "click",
                function( p_oEvent ) {
                    import_list( exper_id );
                }
            );
        }
    );
}

function display_path( path ) {
	var path_length = path.length + 2;
	post_simple_message(
        'popupdialogs',
        'PCDS file path',
        '<textarea rows="1" cols="'+path_length+'" style="padding:4px;">'+path+'</textarea>'
    );
}

/* HDF5 translation manager
 */
var MANAGE_INSTRUMENTS = 1,
    MANAGE_EXPERIMENTS = 2,
    MANAGE_FILES       = 3;

var manage_instrument = null;
var manage_experiment = null;
var manage_path = null;

var manage_tree = null;

function manage_hdf5_translation() {

    set_subtitle( 'Manage HDF5 Translation of Experimens' );
    set_context( 'Select Experiment > ' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    workarea.style.borderLeft = 'solid 1px #c0c0c0';
    workarea.innerHTML=
        '<div>'+
        '  <div style="float:left; width:430px;">'+
        '    <img src="images/ExpSummary.png" />'+
        '    <div id="system_summary" style="margin-top:20px; padding-left:20px; width:430px; height:180px;"></div>'+
        '  </div>'+
        '  <div style="clear:both;">'+
        '  </div>'+
        '</div>'+
        '<div style="margin-top:30px;">'+
        '  <img src="images/Files.png" />'+
        '  <div style="margin-top:20px; padding-left:20px;">'+
        '    <table><thead>'+
        '      <tr>'+
        '        <td style="width: 35px;" class="first_col_hdr">Run</td>'+
        '        <td style="width:250px;" class="col_hdr"><b>File Name</b></td>'+
        '        <td style="width: 50px;" class="col_hdr"><b>Type</b></td>'+
        '        <td style="width:135px;" class="col_hdr_right"><b>Size</b></td>'+
        '        <td style="width:150px;" class="col_hdr"><b>Created</b></td>'+
        '        <td style="width: 60px;" class="col_hdr"><b>xxxxxxxx</b></td>'+
        '        <td style="width: 40px;" class="col_hdr"><b>xxxx</b></td>'+
        '      </tr>'+
        '      <tr>'+
        '        <td><div class="first_separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '      </tr>'+
        '    </thead></table>'+
        '    <br>'+
        '    <div id="experiment_files" style="width:870px; height:340px; overflow:auto;"></div>'+
        '  </div>'+
        '</div>';

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    navarea.innerHTML=
        '<img src="images/Experiment.png" />'+
        '<div id="manage_tree" style="margin-top:20px; padding-left:20px;"></div>';

    manage_tree = new YAHOO.widget.TreeView( "manage_tree" );

    // The whole tree will be built dynamically
    //
    var root_node = new YAHOO.widget.TextNode(
        {   label:    'Instruments/Experiments',
            expanded: false,
            title:    'Expand and select a folder',
            type:     MANAGE_INSTRUMENTS },
        manage_tree.getRoot());

    var currentIconMode = 0;
    root_node.setDynamicLoad( loadNodeData, currentIconMode );
    root_node.toggle();  // Force the node to be instantly open. this will also
                         // trigger the dynamic loading of its children (if any).

    manage_tree.subscribe( "labelClick", onNodeSelection );
    manage_tree.subscribe( "enterKeyPressed", onNodeSelection );
    manage_tree.draw();

    // Dispatch clicks on selected nodes to the corresponding
    // functions.
    //
    function onNodeSelection( node ) {
        if(        node.data.type == MANAGE_INSTRUMENTS) { ;
        } else if( node.data.type == MANAGE_EXPERIMENTS) { ;
        } else if( node.data.type == MANAGE_FILES   ) {
            display_experiment_files(
                node.data.instrument, node.data.experiment, node.data.exper_id );
        }
    }

    function loadNodeData( node, fnLoadComplete ) {

        // We'll create child nodes based on what we get back when we
        // use Connection Manager to pass the text label of the
        // expanding node to the Yahoo!
        // Search "related suggestions" API.  Here, we're at the
        // first part of the request -- we'll make the request to the
        // server.  In our Connection Manager success handler, we'll build our new children
        // and then return fnLoadComplete back to the tree.

        // Get the node's label and urlencode it; this is the word/s
        // on which we'll search for related words:
        //
        //     alert( "node: "+node.label+", type: "+node.data.type );
        //     var nodeLabel = encodeURI( node.data.label );

        // prepare URL for XHR request:
        //
        var sUrl = "../explorer/ws/RequestExperiments.php?type="+node.data.type;
        switch( node.data.type ) {
            case MANAGE_FILES:
                sUrl += '&path='+node.data.path;
            case MANAGE_EXPERIMENTS:
                sUrl += '&instrument='+node.data.instrument;
            case MANAGE_INSTRUMENTS:
                break;
        }

        // prepare our callback object
        //
        var callback = {

            // if our XHR call is successful, we want to make use
            // of the returned data and create child nodes.
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

                // When we're done creating child nodes, we execute the node's
                // loadComplete callback method which comes in via the argument
                // in the response object (we could also access it at node.loadComplete,
                // if necessary):
                //
                oResponse.argument.fnLoadComplete();
            },

            // if our XHR call is not successful, we want to
            // fire the TreeView callback and let the Tree
            // proceed with its business.
            //
            failure: function(oResponse) {
                alert( "failed to get the information from server for node: "+node.label+", type: "+node.data.type );
                oResponse.argument.fnLoadComplete();
            },

            // our handlers for the XHR response will need the same
            // argument information we got to loadNodeData, so
            // we'll pass those along:
            //
            argument: {
                "node": node,
                "fnLoadComplete": fnLoadComplete
            },

            // timeout -- if more than 7 seconds go by, we'll abort
            // the transaction and assume there are no children:
            //
            timeout: 7000
        };

        // With our callback object ready, it's now time to
        // make our XHR call using Connection Manager's
        // asyncRequest method:
        //
        YAHOO.util.Connect.asyncRequest( 'GET', sUrl, callback );
    }
}
/*
function display_experiment_files( instrument, experiment, exper_id ) {

    set_subtitle( 'Manage HDF5 Translation of Experiment - <b>'+instrument+' / '+experiment+'</b>' );
    set_context(
        '<a href="javascript:manage_hdf5_translation()">Select Experiment</a> &gt; '+
        '<b>'+instrument+' / '+experiment+'</b> [ ID='+exper_id+' ]' );

    document.getElementById('experiment_summary').innerHTML='Loading...';
    document.getElementById('experiment_files').innerHTML='Loading...';

    load( '../explorer/ws/ExperimentSummary.php?id='+exper_id, 'experiment_summary' );
    load( '../explorer/ws/RequestExperimentFiles.php?exper_id='+exper_id, 'experiment_files' );
}
*/
/*
 * iRODS catalogs browser
 */
var BROWSE_ZONES = 1,
    BROWSE_CATALOGS = 2;

var catalogs_tree = null;

function browse_catalogs() {
	set_subtitle( 'File Manager Catalogs' );
    set_context( 'Select iRODS Catalog > ' );

    reset_navarea();
    reset_workarea();

    var workarea = document.getElementById('workarea');
    //workarea.style.borderLeft = 'solid 1px #c0c0c0';
    workarea.style.borderLeft = '';
    //workarea.style.backgroundColor = '#f0f0f0';
    workarea.innerHTML=
        '<div>'+
        '  <div style="margin-top:50px;">'+
        '    <table><thead>'+
        '      <tr>'+
        '        <td style="width:235px;" class="first_col_hdr"><b>File</b></td>'+
        '        <td style="width: 65px;" class="col_hdr"><b>Owner</b></td>'+
        '        <td style="width:135px;" class="col_hdr_right"><b>Size</b></td>'+
        '        <td style="width:150px;" class="col_hdr"><b>Created</b></td>'+
        '        <td style="width:100px;" class="col_hdr"><b>Resource</b></td>'+
        '        <td style="width: 40px;" class="col_hdr"><b>Location</b></td>'+
        '      </tr>'+
        '      <tr>'+
        '        <td><div class="first_separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '        <td><div class="separator"></div></td>'+
        '      </tr>'+
        '    </thead></table>'+
        '    <br>'+
        '    <div id="catalog_files" style="width:900px; height:600px; overflow:auto;"></div>'+
        '  </div>'+
        '</div>';

    var navarea = document.getElementById('navarea');
    navarea.style.display = 'block';
    //navarea.style.borderRight = 'solid 1px #c0c0c0';
    navarea.innerHTML=
        '<img src="images/Catalogs.png" />'+
        '<div id="catalogs_tree" style="margin-top:20px; padding-left:20px;"></div>';

    catalogs_tree = new YAHOO.widget.TreeView( "catalogs_tree" );

    // The whole tree will be built dynamically
    //
    var root_node = new YAHOO.widget.TextNode(
        {   label:    'Zones',
            expanded: false,
            title:    'Expand and select a zone',
            type:     BROWSE_ZONES },
        catalogs_tree.getRoot());

    var currentIconMode = 0;
    root_node.setDynamicLoad( loadNodeData, currentIconMode );
    root_node.toggle();     // Force the node to be instantly open. this will also
                            // trigger the dynamic loading of its children (if any).

    catalogs_tree.subscribe( "labelClick", onNodeSelection );
    catalogs_tree.subscribe( "enterKeyPressed", onNodeSelection );
    catalogs_tree.draw();

    // Dispatch clicks on selected nodes to the corresponding
    // functions.
    //
    function onNodeSelection( node ) {
        if(        node.data.type == BROWSE_ZONES) { ;
        } else if( node.data.type == BROWSE_CATALOGS   ) {
            display_catalog_files( node.data );
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
        var sUrl = "../explorer/ws/RequestCatalogs.php?type="+node.data.type;
        switch( node.data.type ) {
            case BROWSE_ZONES:
                sUrl += '&path=';
                break;
            case BROWSE_CATALOGS:
                sUrl += '&path='+node.data.path;
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

function display_catalog_files( node_data ) {

    set_subtitle( 'iRODS Catalog - <b>'+node_data.path+'</b>' );
    set_context(
        '<a href="javascript:browse_catalogs()">Select iRODS Catalog</a> &gt; <b>'+node_data.context+'</b>' );

    document.getElementById('catalog_files').innerHTML='Loading...';

    load( '../explorer/ws/RequestCatalogFiles.php?path='+node_data.path, 'catalog_files' );
}

</script>

</head>
<body class="yui-skin-sam" id="body" onload="init()">
    <div id="application_header">
      <div>
        <div style="float:left;">
          <p id="application_title" style="text-align:left;">
            <em>Data Files Explorer: </em>
            <em id="application_subtitle">LCLS Controls and Data Systems</em>
            <em id="application_subtitle"></em>
          </p>
        </div>
        <div style="float:right; height:50px;">
<?php
if( $auth_svc->authName() == '' ) {
    echo <<<HERE
          <br>
          <br>
          <a href="../../apps/filemgr"><p title="login here to proceed to the full version of the application">login</p></a>
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
?>
        </div>
        <div style="height:40px;">&nbsp;</div>
      </div>
    </div>
    <div id="menubar" class="yuimenubar yuimenubarnav"></div>
    <div id="context"></div>
    <div id="nav-and-work-areas" align="left">
      <div id="navarea"></div>
      <div id="workarea"></div>
    </div>
    <div id="popupdialogs"></div>

</body>
</html>

<?php

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>