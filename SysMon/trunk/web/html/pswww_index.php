<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

$title    = 'LCLS Controls & Data Systems' ;
$subtitle = 'Web Tools & Documentation Catalog' ;

$groups = array (
    array (
        'name'  => 'Applications for User Experiments' ,
        'links' => array (
            array (
                'name'  => '<b>Data Manager</b> (formerly: <b>Web Portal</b>)' ,
                'href'  => '/apps/portal' ,
                'title' =>
                    "This experiment-centric application provides a collection \n" .
                    "of tools for viewing/managing data and metadata of an experiment. \n" .
                    "The tools includes: Experiment's group manager, Electronic LogBook, \n" .
                    "Run Tables, File Manager, HDF5 Translation ordering system. \n" .
                    "Each experiment gets its own instace of the Data Manager. \n" .
                    "Members of the experiments get access to the application after \n" .
                    "being ibncluded into the experiment's POSIX group by the PI." ,
                'login' => True
            )
        )
    ) ,
    array (
        'name'  => 'Experiment/Hutch Management' ,
        'links' => array (
            array (
                'name'  => 'Experiment <b>Switch</b>' ,
                'href'  => '/apps/portal/experiment_switch' ,
                'title' =>
                    "The application is meant to activate select experiments for \n" .
                    "data taking at the corresponding instruments. Note, that only \n" .
                    "one experiment at a time can be in the ACTIVE state at \n " .
                    "a particular instance of the DAQ system." ,
                'login' => True
            ) ,
            array (
                'name'  =>  '<b>Shift/Hutch</b> Manager' ,
                'href'  => '/apps/shiftmgr' ,
                'title' =>
                    "The Shift/Hutch management application keeps a track of various \n" .
                    "activities in the instrument hutches, the amount of time spent for \n" .
                    "each activity (such as experiment tuning, system alignment, data \n" .
                    "taking, etc.) as well as a summary of problems during the shifts. \n" .
                    "The tool can be also used to generate summary reports (presently \n" .
                    "MS Excel Spreadsheets) on the use of time in the hutches over \n" .
                    "an extended period of time." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Experiment <b>Registry</b> Database' ,
                'href'  => '/apps/regdb' ,
                'title' =>
                    "This is presently the main (and the only) tool for tegistering \n" .
                    "or modifying experiment entries in LCLS. Experiments are associated \n" .
                    "with beam time proposals which are managed by SSRL/LCLS Users Office. \n" .
                    "Each experiment is associated with a UNIX/POSIX group, and it has \n" .
                    "some description, the contact information (including PI name and \n" .
                    "e-mail address). The experiment entries are used in many areas \n" .
                    "throughout LCLS, such as: data taking (DAQ), Data Management, \n" .
                    "Access Control to the data and Web applicatons, etc. \n" .
                    "Only the authorized personnel can register experiments." ,
                'login' => True
            ) ,
            array (
                'name'  => '<b>Authorization</b> Database' ,
                'href'  => '/apps/authdb' ,
                'title' =>
                    "The tool for managing the role-based access control to the data \n" .
                    "and Web applications. The tools also provides an interface for \n" .
                    "viewing/managing POSIX groups in the PCDS LDAP server, looking up \n" .
                    "for user accounts. The management operations are available to \n" .
                    "the authorized personnel only." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Check and correct authorizations for <b>instrument groups</b>' ,
                'href'  => '/apps/authdb/AuthInstrGroups' ,
                'title' =>
                    "This script displays/manages e-Log and LDAP authorizations for \n" .
                    "instrument-specific POSIX groups. The management operations are \n" .
                    "available to the authorized personnel only." ,
                'login' => True
            ) ,
            array (
                'name'  => 'POSIX <b>Groups Manager</b>' ,
                'href'  => '/apps/authdb/manage_my_groups' ,
                'title' =>
                    "View/manage members of select POSIX groups. Only LCLS specific \n" .
                    "groups are allowed to be managed by the tool. The management \n" .
                    "operations are available to the authorized personnel only." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Group Management <b>Log</b>' ,
                'href'  => '/apps/authdb/LDAP_group_management_log' ,
                'title' =>
                    "View a log of the POSIX group management operations. Note, that \n" .
                    "this page has a restricted access." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Manage <b>Auto-Translation</b> option for <b>HDF5</b> files (<b>STANDARD</b> translation)' ,
                'href'  => '/apps/regdb/AutoTranslate2HDF5?service=STANDARD' ,
                'title' =>
                    "View/Modify HDF5 translation options across all experiments. \n" .
                    "The management operations are available to the authorized \n" .
                    "personnel only." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Manage <b>Auto-Translation</b> option for <b>HDF5</b> files (<b>MONITORING</b> translation)' ,
                'href'  => '/apps/regdb/AutoTranslate2HDF5?service=MONITORING' ,
                'title' =>
                    "View/Modify HDF5 translation options across all experiments. \n" .
                    "The management operations are available to the authorized \n" .
                    "personnel only." ,
                'login' => True
            )
        )
    ) ,
    array (
        'name'  => 'Data Management' ,
        'links' => array (
            array (
                'name'  => 'Default <b>Data Path</b> of Experiments' ,
                'href'  => '/apps/regdb/ExperimentDataPath' ,
                'title' =>
                    "View/manage data placement of experiments. The applications \n" .
                    "maps the logical data path of an experiment to an underlying \n" .
                    "file system(s). The management operations are available \n" .
                    "to the authorized personnel only." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Data <b>Retention Policy</b> Manager' ,
                'href'  => '/apps-dev/regdb/StoragePolicyMgr' ,
                'title' =>
                    "View/adjust parameters of the Data Retention Policy, view \n" .
                    "storage usage statistics, perform disk cleanup for expired \n" .
                    "content. Note, that this page has a restricted access." ,
                'login' => True
            ) ,
            array (
                'name'  => '<b>Data Migration</b> Monitor/Notifier' ,
                'href'  => '/apps-dev/sysmon/dmmon?app=File%20Migration:E-mail%20Notifications' ,
                'title' =>
                    "View delays in the data migration activities, subscribe for \n" .
                    "e-mail notifications for the delayed migrations.",
                'login' => True
            ) ,
            array (
                'name'  => 'Monitoring <b>Tape Restore</b> Requests' ,
                'href' => '/apps-dev/regdb/SimpleTapeMonitor' ,
                'title' =>
                    "View a history of past file restore requests" ,
                'login' => True
            ) ,
            array (
                'name'  => 'Monitoring <b>Data Movers</b> Performance' ,
                'href' => '/apps-dev/sysmon/dmmon' ,
                'title' =>
                    "The application for the real-time monitoring of the data movers. \n" .
                    "Results are presented as plots or tables.",
                'login' => True
            ) ,
            array (
                'name'  => 'Data Collection <b>Statistics</b>' ,
                'href' => '/apps/portal/statistics' ,
                'title' =>
                    "Compile and present data collection statistics accross \n" .
                    "instruments and experiments." ,
                'login' => True
            )
        )
    ) ,
    array (
        'name'  => 'Computing System' ,
        'links' => array (
            array (
                'name'  => '<b>Ganglia</b> Monitoring' ,
                'href'  => '/system/ganglia/' ,
                'title' =>
                    "Scalable distributed monitoring system for the PCDS \n" .
                    "computing system." ,
                'login' => True
            ) ,
            array (
                'name'  => '<b>M/Monit</b> Monitoring' ,
                'href'  => '/system/mmonit/' ,
                'title' =>
                    "Monitoring system for various computing, networking \n" .
                    "and storage services at PCDS. Access is allowed to \n" .
                    "authorized personnel only." ,
                'login' => True
            ) ,
            array (
                'name'  => 'Inventory' ,
                'href'  => '/inventory/' ,
                'title' =>
                    "The inventory of computing resources and services at PCDS. \n" .
                    "Access is allowed to authorized personnel only." ,
                'login' => True
            )
        )
    ) ,
    array (
        'name'  => 'Engineering Databases' ,
        'links' => array (
            array (
                'name'  => 'Cable Ordering & Management (<b>NeoCAPTAR</b>)' ,
                'href'  => '/apps/neocaptar' ,
                'title' =>
                    "The application to support a complete workflow of cable \n" .
                    "ordering, fabrication and installation at PCDS. " ,
                'login' => True
            ) ,
            array (
                'name'  => 'Inventory Database of Electronic Equipment (<b>IREP</b>)' ,
                'href'  => '/apps/irep' ,
                'title' =>
                    "The inventory database for electronic equipment." ,
                'login' => True
            )
        )
    ) ,
    array (
        'name'  => 'Development Support Tools & Documentation' ,
        'links' => array (
            array (
                'name'  => '<b>PCDS</b> Computing Documentation in Confluence' ,
                'href'  => 'https://confluence.slac.stanford.edu/display/PCDS/Computing' ,
                'title' =>
                    "Computing and storage resources for LCLS users, computer \n" .
                    "accounts, Data Retention Policy, data management operations." ,
                'login' => False
            ) ,
            array (
                'name'  => '<b>Analysis</b> Documentation in Confluence' ,
                'href'  => 'https://confluence.slac.stanford.edu/display/PSDM' ,
                'title' =>
                    "Comprehensive documentation on analysis frameworks, algorithms \n" .
                    "and tools which are provided to users at LCLS." ,
                'login' => True
            ) ,
            array (
                'name'  => 'TRAC Issue Tracker: <b>Analysis & Data Management</b> (obsolete)' ,
                'href'  => '/trac/psdm/' ,
                'title' =>
                    "The old issue tracker which was in use befor switching to JIRA \n" .
                    "in August 2014. See JIRA projects for the on-going development." ,
                'login' => True
            ) ,
            array (
                'name'  => 'TRAC Issue Tracker: <b>Data Acquisition Systems</b>' ,
                'href'  => '/trac/daq/' ,
                'title' =>
                    "The issue tracker is still in use." ,
                'login' => True
            ) ,
            array (
                'name'  => 'TRAC Issue Tracker: <b>Controls</b>' ,
                'href'  => '/trac/controls/' ,
                'title' =>
                    "This issue tracker is being actively used by the Controls group." ,
                'login' => True
            ) ,
            array (
                'name'  => 'JIRA Issue Tracker: <b>Software Release Tools</b>' ,
                'href'  => 'https://jira.slac.stanford.edu/browse/PSRT' ,
                'title' =>
                    "Supporting on-going software developmment efforts in \n" .
                    "an area of the core tools and code integration infrastructure. \n" .
                    "The tracker is available for read access to everyone (no SLAC UNIX \n" .
                    "account is needed to explore its content) and it is write-enabled \n" .
                    "for members of the core computing team of LCLS.",
                'login' => True
            ) ,
            array (
                'name'  => 'JIRA Issue Tracker: <b>Analysis Software</b>' ,
                'href'  => 'https://jira.slac.stanford.edu/browse/PSAS' ,
                'title' =>
                    "Supporting on-going software developmment efforts in \n" .
                    "an area of the analysis frameworks, tools, algorithms and methods. \n" .
                    "The tracker is available for read access to everyone (no SLAC UNIX \n" .
                    "account is needed to explore its content) and it is write-enabled \n" .
                    "for members of the core computing team of LCLS.",
                'login' => True
            ) ,
            array (
                'name'  => 'JIRA Issue Tracker: <b>Data Management</b>' ,
                'href'  => 'https://jira.slac.stanford.edu/browse/PSDH' ,
                'title' =>
                    "Supporting on-going software developmment efforts in \n" .
                    "an area of the LCLS Data Management applications, as well as \n" .
                    "Data Management operations. \n" .
                    "The tracker is available for read access to everyone (no SLAC UNIX \n" .
                    "account is needed to explore its content) and it is write-enabled \n" .
                    "for members of the core computing team of LCLS.",
                'login' => True
            ) ,
            array (
                'name'  => 'JIRA Issue Tracker: <b>Web Applications</b>' ,
                'href'  => 'https://jira.slac.stanford.edu/browse/PSWA' ,
                'title' =>
                    "Supporting on-going software developmment efforts in \n" .
                    "an area of the Web applications and Web-based tools. \n" .
                    "The tracker is available for read access to everyone (no SLAC UNIX \n" .
                    "account is needed to explore its content) and it is write-enabled \n" .
                    "for members of the core computing team of LCLS.",
                'login' => True
            ) ,
            array (
                'name'  => 'JIRA Issue Tracker: <b>Infrastructure</b>' ,
                'href'  => 'https://jira.slac.stanford.edu/browse/PCI' ,
                'title' =>
                    "Supporting maintainance efforts for the services infrastructure \n" .
                    "needed by the Analysis and Data Management applications. \n" .
                    "The tracker is available for read access to everyone (no SLAC UNIX \n" .
                    "account is needed to explore its content) and it is write-enabled \n" .
                    "for members of the core computing team of LCLS.",
                'login' => True
            ) ,
            array (
                'name'  => 'BUILDBOT: <b>Release Build Service</b>' ,
                'href'  => '/buildbot/' ,
                'title' =>
                    "The Web interfaces to our installation of an open-source framework \n" .
                    "for automating  software build, test, and release processes." ,
                'login' => True
            ) ,
            array (
                'name'  => '<b>Release Comparision</b> Tool' ,
                'href'  => '/apps/websrt/releases' ,
                'title' =>
                    "See a history of all release builds, and release notes. Compare tags \n" .
                    "of select two releases, go to the SVN repository through the Web interface \n" .
                    "to see code changes between two releases." ,
                'login' => True
            )
        )
    ) ,
    array (
        'name'  => 'Code & Downloads' ,
        'links' => array (
            array (
                'name'  => 'RPMS of the <b>Analysis Releases</b> and <b>External Packages</b>' ,
                'href'  => '/psdm-repo' ,
                'title' =>
                    "A collection of RPMs at this site is used by release \n" .
                    "exportation tools." ,
                'login' => False
            ) ,
            array (
                'name'  => 'Virtual Machine Images: <b>Analysis Software</b>' ,
                'href'  => '/psdm-vm' ,
                'title' =>
                    "Virtual machine images availble for downloading by LCLS users. \n" .
                    "The machines are preloaded with analysis release(s) and other \n" .
                    "infrastructure which is sufficient to begin doing (learning \n" .
                    "how to do) data analysis on the LCLS data. See detail instructions \n" .
                    "on using the VMs at the corresponding Confluence page." ,
                'login' => False
            ) ,
            array (
                'name'  => 'SVN Repository: <b>Analysis & Data Management</b>' ,
                'href'  => '/svn/psdmrepo/' ,
                'title' =>
                    "This path to the SVN repository should be used in order to \n" .
                    "contribute code into the repository. Note, this URL requires \n" .
                    "a valid Kerberos V5 token in realm: SLAC.STANFORD.EDU." ,
                'login' => True
            ) ,
            array (
                'name'  => 'SVN Repository: <b>Analysis & Data Management</b> (read-only)' ,
                'href'  => '/svn-readonly/psdmrepo/' ,
                'title' =>
                    "The read-only access to the repository w/o any authentication." ,
                'login' => False
            ) ,
            array (
                'name'  => 'SVN Repository Web Browser: <b>Analysis & Data Management</b> (read-only)' ,
                'href'  => 'http://java.freehep.org/svn/repos/psdm/list/' ,
                'title' =>
                    "User-friendly Web UI for browsing the content of the repository. \n" .
                    "Please, don't be confused with the site to where this link is \n" .
                    "pointing to! That site is just a proxy to the repository. This proxy \n" .
                    "will eventually move to PCDS",
                'login' => False
            )
        )
    )
) ;

//print_r(phpinfo()) ;

?>
<!DOCTYPE html>
<html>

<head>

<title><?="{$title}: {$subtitle}"?></title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
<meta name="viewport" content="width=device-width, initial-scale=1">

<link rel="icon" href="/apps/webfwk/img/Portal_favicon.ico"/>


<style>
    body {
        margin:         0;
        padding:        0;
    }
    #top {
        padding:        0px;
        /*
        font-family:    "Linux Libertine",Georgia,Times,serif;
        */
        font-family: 'Segoe UI',Tahoma,Helvetica,Arial,Verdana,sans-serif;
        /*
        font-family:    "Myriad Set Pro","Lucida Grande",Helvetica,Arial,Verdana,sans-serif;
        */
        /*
        font-family:    'lucida grande',tahoma,verdana,arial,sans-serif;
        */
    }
    #header {
        padding:            14px;
        background-color:   #8c1515;
        color:              #f0f0f0;
        box-shadow:         0 8px 10px rgba(0,0,0,0.36);
    }
    #title {
        text-align:         left;
        font-weight:        bold;
        font-size:          32px;
    }
    #login {
        font-size:          12px;
    }
    #login button {
        color:              red;
        background-image:   0 !important;
        border-radius:      2px !important;
        border-color:       darkgrey !important;
    }
    #center {
        padding:        20px;
        padding-top:    10px;
        overflow-y:     auto;
        font-family:    'Segoe UI',Tahoma,Helvetica,Arial,Verdana,sans-serif;
    }
    .group {
        float:          left;
        margin:         10px 20px 10px 0;
        padding:        10px 20px 20px 10px;
        box-shadow:     0 5px 10px rgba(0,0,0,0.36);
    }
    .group > .name {
        margin-bottom:  10px;
        border-bottom:  1px solid #a0a0a0;
        font-size:      20px;
        font-weight:    bold;
    }
    .group > .entry {
        padding:        5px 5px 5px 15px;
        width:          100%;
    }
    .group > .entry:hover {
        background-color:   aliceblue;
    }
    .group > .entry > a.link {
        text-decoration:    none;
        color:              #2e2d29; /*#0071bc;*/
    }
    .group > .entry > a.link:hover {
        color:              #000000;
    }
    .annotated[data]:hover:after {
      content:          attr(data);
      padding:          8px;
      color:            #000000;
      position:         absolute;
      left:             -20;
      top:              -60;
      white-space:      pre;
      max-width:        520px;
      z-index:          2;
      border-radius:    3px ;
      background-color: lemonchiffon;      
      font-weight:      normal;
      font-size:        13px;
    }
    .end-of-groups {
        clear:          both;
    }
    @media screen and (max-width: 800px){
        #header {
            box-shadow: none;
            border:     0;
        }
        #title {
            font-size:  24px;
            max-width:  80%;
        }
        #center {
            padding:    0px;
            border:     0;
        }
        .group {
            float:      none;
            width:      100%;
            box-shadow: none;
            border:     0;
            margin:     0;
            padding:    0;
            padding-bottom: 15px;
        }
        .group > .name {
            font-size:  18px;
            background-color:   #f0f0f0;
            background: linear-gradient(#e0e0e0, #ffffff);
            padding:    10px;
            margin:     0;
            border:     0;
        }
        .group > .entry {
            padding: 5px 5px 0px 15px;
        }
        .group > .entry > a.link {
            font-size:  14px;
        }
        .annotated[data]:hover:after {
            font-size:  12px;
        }
    }
</style>

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
        
<script type="text/javascript" src="/apps/webfwk/js/Utilities.js"></script>
<script type="text/javascript">

/* ----------------------------------------------
 * Window geometry (compatible with all browsers)
 * ----------------------------------------------
 */
function document_inner_geometry() {
    var winW = 630, winH = 460;
    if (document.body && document.body.offsetWidth) {
    winW = document.body.offsetWidth;
    winH = document.body.offsetHeight;
    }
    if (document.compatMode=='CSS1Compat' &&
        document.documentElement &&
        document.documentElement.offsetWidth ) {
        winW = document.documentElement.offsetWidth;
        winH = document.documentElement.offsetHeight;
    }
    if (window.innerWidth && window.innerHeight) {
        winW = window.innerWidth;
        winH = window.innerHeight;
    }
    var result = {width: winW, height: winH };
    return result;
}
/* ----------------------------------------
 * Authentication and authorization context
 * ----------------------------------------
 */
var auth_is_authenticated=1;
var auth_type            ="WebAuth";
var auth_remote_user     ="<?=$_SERVER['WEBAUTH_USER']?>";

var auth_webauth_token_creation   ="<?=$_SERVER['WEBAUTH_TOKEN_CREATION']?>";
var auth_webauth_token_expiration ="<?=$_SERVER['WEBAUTH_TOKEN_EXPIRATION']?>";

function refresh_page() {
    window.location = "<?=$_SERVER['REQUEST_URI']?>";
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
        $('#popupdialogs').html(
            '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
            'Your WebAuth session has expired. Press <b>Ok</b> or use <b>Refresh</b> button'+
            'of the browser to renew your credentials.</p>'
        );
        $('#popupdialogs').dialog({
            resizable: false,
            modal: true,
            buttons: {
                "Ok": function() {
                    $( this ).dialog( "close" );
                    refresh_page();
                }
            },
            title: 'Session Expiration Notification'
        });
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
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
        'This will log yout out from the current WebAuth session. Are you sure?</p>'
     );
    $('#popupdialogs').dialog( {
        resizable: false,
        modal: true,
        buttons: {
            "Yes": function() {
                $( this ).dialog( "close" );
                document.cookie = 'webauth_wpt_krb5=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
                document.cookie = 'webauth_at=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
                refresh_page();
            },
            Cancel: function() {
                $( this ).dialog( "close" );
            }
        },
        title: 'Session Logout Warning'
    } );
}
$(document).ready(
    function() {
        $('#logout').button().click(logout);
        auth_timer_restart();
    }
);
</script>


</head>

<body>
    <div id="top" >
        <div id="header">
            <div id="title" style="float:left;" ><?=$title?> : <?=$subtitle?></div>
            <div id="login" style="float:right;" >
                <table><tbody>
                    <tr>
                        <td>Logged as</td>
                        <td><b><?=$_SERVER['WEBAUTH_USER']?></b></td>
                        <td><button id="logout"
                                    title="close the current WebAuth session">LOGOUT</button></td>
                    </tr>
                    <tr>
                        <td>Session expires in : </td>
                        <td><span id="auth_expiration_info"><b>00:00.00</b></span></td>
                    </tr>
                </tbody></table>
            </div>
            <div style="clear:both;"></div>
        </div>
    </div>
    <div id="center" >
<?php   foreach ($groups as $g) { ?>
        <div class="group" >
            <div class="name" ><?=$g['name']?></div>
<?php       foreach ($g['links'] as $l) { ?>
                <div class="entry annotated" data="<?=$l['title']?>" ><a class="link" href="<?=$l['href']?>" ><?=$l['name']?></a></div>
<?php   } ?>
        </div>
<?php   } ?>
        <div class="end-of-groups"></div>
    </div>
    <div id="popupdialogs"></div>
</body>
</html>

