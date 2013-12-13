<?php


/* Let a user to select an experiment first if no valid experiment
 * identifier is supplied to the script.
 */
function select_experiment () {
    header("Location: select_experiment.php") ;
    exit ;
}

$exper_id = isset($_GET['exper_id']) ? intval(trim($_GET['exper_id'])) : 0 ;
if (!$exper_id) select_experiment() ;

/* Redirections to support legacy URLs.
 */
$known_apps = array (

    'experiment' => array (
        'name'             => 'Experiment' ,
        'context1_default' => 'Info' ,
        'context1'         => array (
            'summary'          => 'Info' ,
            'manage'           => 'Group Manager' )) ,

    'elog' => array (
        'name'             => 'e-Log' ,
        'context1_default' => 'Recent (Live)' ,
        'context1'         => array (
            'recent'           => 'Recent (Live)' ,
            'post'             => 'Post' ,
            'search'           => 'Search' ,
            'shifts'           => 'Shifts' ,
            'runs'             => 'Runs' ,
            'attachments'      => 'Attachments' ,
            'subscribe'        => 'Subscribe' )) ,

    'runtables' => array (
        'name'             => 'Run Tables' ,
        'context1_default' => 'Calibrations' ,
        'context1'         => array (
            'calibrations'     => 'Calibrations' ,
            'detectors'        => 'DAQ Detectors' ,
            'epics'            => 'EPICS' )) ,

    'datafiles' => array (
        'name'             => 'File Manager' ,
        'context1_default' => 'Summary' ,
        'context1'         => array (
            'summary'          => 'Summary' ,
            'files'            => 'XTC HDF5' )) ,

    'hdf' => array (
        'name'             => 'HDF5 Translation' ,
        'context1_default' => 'Manage' ,
        'context1'         => array (
            'manage'           => 'Manage' )) ,

    'shiftmgr' => array (
        'name'             => 'Shift Manager' ,
        'context1_default' => 'Reports' ,
        'context1'         => array (
            'reports'          => 'Reports' ,
            'history'          => 'History' ,
            'notifications'    => 'E-mail Notifications' )) ,
) ;

$select_app = null ;
$select_app_context1 = null ;

if (isset( $_GET['app'])) {
    $app_path = explode(':', strtolower(trim($_GET['app']))) ;
    $application = $app_path[0] ;
    if (array_key_exists($application, $known_apps)) {
        $select_app = $known_apps[$application]['name'] ;
        if (count($app_path) > 1) {
            $context1 = $app_path[1] ;
            if (array_key_exists($context1, $known_apps[$application]['context1'])) {
                $select_app_context1 = $known_apps[$application]['context1'][$context1] ;
            }
        }
        if (is_null($select_app_context1)) $select_app_context1 = $known_apps[$application]['context1_default'] ;
    }
}
if (is_null($select_app)) {
    $select_app = 'Experiment' ;
    $select_app_context1 = 'Info' ;
}
/* Parse optional parameters which may be used by applications. The parameters
 * will be passed directly into applications for further analysis (syntax, values,
 * etc.).
 */
if (isset($_GET['params'])) {
    $params = explode(',', trim( $_GET['params'])) ;
}

require_once 'authdb/authdb.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use AuthDb\AuthDb ;
use LogBook\LogBook ;
use LogBook\LogBookAuth ;
use RegDB\RegDB ;
use RegDB\RegDBAuth ;

try {
    AuthDb::instance()->begin() ;
    RegDB::instance()->begin() ;
    LogBook::instance()->begin() ;

    $is_data_administrator = AuthDb::instance()->hasPrivilege (
        AuthDb::instance()->authName() ,
        null ,
        'StoragePolicyMgr' ,
        'edit'
    ) ;

    $logbook_experiment = LogBook::instance()->find_experiment_by_id($exper_id) ;
    if (is_null($logbook_experiment)) select_experiment() ;

    $experiment = $logbook_experiment->regdb_experiment() ;
    $instrument = $experiment->instrument() ;

    $title        = $experiment->is_facility() ? 'E-Log of Facility:' : 'Web Portal of Experiment' ;
    $subtitle     = $experiment->instrument()->name().'/'.$experiment->name() ;
    $subtitle_url = '<a href="select_experiment.php" title="Switch to another experiment">'.$experiment->instrument()->name().'&nbsp;/&nbsp;'.$experiment->name().'</a>' ;

    $user = RegDB::instance()->find_user_account(AuthDb::instance()->authName()) ;
    if (!$user) die("Sorry, can't safely run this application due to a broken authentication system.") ;

    $experiment_can_manage_group = false ;
    foreach (array_keys( RegDB::instance()->experiment_specific_groups()) as $g) {
        if ($g === $experiment->POSIX_gid()) {
            $experiment_can_manage_group = RegDBAuth::instance()->canManageLDAPGroup($g) ;
            break ;
        }
    }
    $elog_can_read_messages   = LogBookAuth::instance()->canRead           ($logbook_experiment->id()) ;
    $elog_can_post_messages   = LogBookAuth::instance()->canPostNewMessages($logbook_experiment->id()) ;
    $elog_can_edit_messages   = LogBookAuth::instance()->canEditMessages   ($logbook_experiment->id()) ;
    $elog_can_delete_messages = LogBookAuth::instance()->canDeleteMessages ($logbook_experiment->id()) ;
    $elog_can_manage_shifts   = LogBookAuth::instance()->canManageShifts   ($logbook_experiment->id()) ;

    $calibrations_can_edit = 
        RegDB::instance()->is_member_of_posix_group('ps-data', AuthDb::instance()->authName()) ||
        (!$experiment->is_facility() && RegDB::instance()->is_member_of_posix_group('ps-'.strtolower($instrument->name()), AuthDb::instance()->authName())) ;

    $experiment_can_read_data =
        $experiment_can_manage_group ||
        $elog_can_read_messages || $elog_can_post_messages || $elog_can_edit_messages || $elog_can_delete_messages || $elog_can_manage_shifts ||
        $calibrations_can_edit ||
        RegDB::instance()->is_member_of_posix_group('ps-data', AuthDb::instance()->authName()) ||
        RegDB::instance()->is_member_of_posix_group($logbook_experiment->POSIX_gid(), AuthDb::instance()->authName()) ||
        (!$experiment->is_facility() && RegDB::instance()->is_member_of_posix_group('ps-'.strtolower( $instrument->name()), AuthDb::instance()->authName())) ;
    
    $shiftmgr_can_edit = AuthDb::instance()->hasRole (
        AuthDb::instance()->authName() ,
        null ,
        'ShiftMgr' ,
        "Manage_{$instrument->name()}"
    ) ;
?>

<!DOCTYPE html>
<html>

<head>

<title><?=$title?> / <?=$subtitle?></title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />
<link type="text/css" href="/jquery/css/jquery-ui-timepicker-addon.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Fwk.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/Stack.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/SmartTable.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/Experiment_Info.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Experiment_Group.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_MessageViewer.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Live.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Post.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Search.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Shifts.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Runs.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Attachments.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ELog_Subscribe.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Runtables_Calibrations.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Runtables_Detectors.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Runtables_EPICS.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Filemanager_Summary.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Filemanager_Files.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Filemanager_Files_USR.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/HDF5_Manage.css" rel="Stylesheet" />

<link type="text/css" href="../shiftmgr/css/shiftmgr.css" rel="Stylesheet" />


<style>

span.toggler {
  background-color: #ffffff;
  border: 1px solid #c0c0c0;
  border-radius: 4px;
  -moz-border-radius: 4px;
  cursor: pointer;
}

button.control-button,
label.control-label {
  font-size: 10px;
  color: black;
}

.shift-reports,
.shift-history-reports,
.shift-notifications {
  padding: 15px 20px 20px 20px;
}

</style>

<script type="text/javascript" src="/jquery/js/jquery-1.8.2.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-timepicker-addon.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.resize.js"></script>

<script type="text/javascript" src="../webfwk/js/Class.js" ></script>
<script type="text/javascript" src="../webfwk/js/Widget.js" ></script>
<script type="text/javascript" src="../webfwk/js/StackOfRows.js" ></script>
<script type="text/javascript" src="../webfwk/js/Fwk.js"></script>
<script type="text/javascript" src="../webfwk/js/Table.js"></script>
<script type="text/javascript" src="../webfwk/js/SmartTable.js" ></script>

<script type="text/javascript" src="../portal/js/Experiment_Info.js"></script>
<script type="text/javascript" src="../portal/js/Experiment_Group.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Utils.js"></script>
<script type="text/javascript" src="../portal/js/ELog_MessageViewer.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Live.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Post.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Search.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Shifts.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Runs.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Attachments.js"></script>
<script type="text/javascript" src="../portal/js/ELog_Subscribe.js"></script>
<script type="text/javascript" src="../portal/js/Runtables_Calibrations.js"></script>
<script type="text/javascript" src="../portal/js/Runtables_Detectors.js"></script>
<script type="text/javascript" src="../portal/js/Runtables_EPICS.js"></script>
<script type="text/javascript" src="../portal/js/Filemanager_Summary.js"></script>
<script type="text/javascript" src="../portal/js/Filemanager_Files.js"></script>
<script type="text/javascript" src="../portal/js/Filemanager_Files_USR.js"></script>
<script type="text/javascript" src="../portal/js/HDF5_Manage.js"></script>

<script type="text/javascript" src="../shiftmgr/js/Definitions.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Reports.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Notifications.js"></script>
<script type="text/javascript" src="../shiftmgr/js/History.js"></script>

<script type="text/javascript">

var experiment = {
    id          :  <?= $experiment->id() ?> ,
    name        : '<?= $experiment->name() ?>' ,
    contact_uid : '<?= $experiment->leader_Account() ?>' ,
    posix_group : '<?= $experiment->POSIX_gid() ?>' ,
    is_facility :  <?= $experiment->is_facility() ? 1 : 0 ?> ,
    instrument : {
        id   :  <?= $instrument->id()   ?> ,
        name : '<?= $instrument->name() ?>'
    }
}

var access_list = {
    user : {
        uid   : '<?= $user["uid"]   ?>' ,
        gecos : '<?= htmlspecialchars($user["gecos"], ENT_QUOTES | ENT_HTML5) ?>' ,
        email : '<?= htmlspecialchars($user["email"], ENT_QUOTES | ENT_HTML5) ?>'
    } ,
    experiment : {
        view_info    : <?= $experiment_can_read_data    ? 1 : 0 ?> ,
        manage_group : <?= $experiment_can_manage_group ? 1 : 0 ?>
    } ,
    elog : {
        read_messages   : <?= $elog_can_read_messages   ? 1 : 0 ?> ,
        post_messages   : <?= $elog_can_post_messages   ? 1 : 0 ?> ,
        edit_messages   : <?= $elog_can_edit_messages   ? 1 : 0 ?> ,
        delete_messages : <?= $elog_can_delete_messages ? 1 : 0 ?> ,
        manage_shifts   : <?= $elog_can_manage_shifts   ? 1 : 0 ?>
    } ,
    runtables : {
        read :              <?= $experiment_can_read_data ? 1 : 0 ?> ,
        edit_calibrations : <?= $calibrations_can_edit    ? 1 : 0 ?>
    } ,
    datafiles : {
        read                  : <?=$experiment_can_read_data ? 1 : 0 ?> ,
        manage                : <?=$elog_can_post_messages   ? 1 : 0 ?> ,
        is_data_administrator : <?=$is_data_administrator    ? 1 : 0 ?>
    } ,
    hdf5 : {
        read   : <?=$experiment_can_read_data ? 1 : 0 ?> ,
        manage : <?=$elog_can_post_messages   ? 1 : 0 ?>
    } ,
    shiftmgr : {
        can_edit : <?php echo $shiftmgr_can_edit ? 1 : 0 ; ?>
    } ,
    no_page_access_html :
'<br><br>' +
'<center>' +
'  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">' +
'    A c c e s s &nbsp; E r r o r' +
'  </span>' +
'</center>' +
'<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;"> ' +
'  We\'re sorry! Our records indicate that your SLAC UNIX account has no proper permissions to access this page.' +
'  You\'re encoureged to directly contact the PI of the experiment to resolve this issue.' +
'</div>'
} ;

var select_app = '<?=$select_app?>' ;
var select_app_context1 = '<?=$select_app_context1?>' ;

var global_extra_params = {} ;
<?php
    if (isset($params)) {
        foreach ($params as $p) {
            $kv = explode(':', $p) ;
            switch (count($kv)) {
            case 0 :
                break;
            case 1 :
                $k = $kv[0] ;
                echo "global_extra_params['{$k}'] = true ;\n" ;
                break ;
            default :
                $k = $kv[0] ;
                $v = $kv[1] ;
                echo "global_extra_params['{$k}'] = '{$v}' ;\n" ;
                break ;
            }
        }
    }
?>

$(function() {

    var menus = [{

        name: 'Experiment' ,
        menu: [{
            name: 'Info' ,
            application: new Experiment_Info(experiment, access_list) } , {
 
            name: 'Group Manager' ,
            application: new Experiment_Group(experiment, access_list) }]} , {
    
        name: 'e-Log',
        menu: [{
            name: 'Recent (Live)' ,
            application: new ELog_Live(experiment, access_list) } , {

            name: 'Post' ,
            application: new ELog_Post(experiment, access_list) } , {

            name: 'Search' ,
            application: new ELog_Search(experiment, access_list) } , {

            name: 'Shifts' ,
            application: new ELog_Shifts(experiment, access_list) } , {

            name: 'Runs' ,
            application: new ELog_Runs(experiment, access_list) } , {

            name: 'Attachments' ,
            application: new ELog_Attachments(experiment, access_list) } , {

            name: 'Subscribe' ,
            application: new ELog_Subscribe(experiment, access_list) }]}] ;

    if (!experiment.is_facility) {
        menus.push ({
        
            name: 'Run Tables',
            menu: [{
                name: 'Calibrations' ,
                application: new Runtables_Calibrations(experiment, access_list) } , {

                name: 'DAQ Detectors' ,
                application: new Runtables_Detectors(experiment, access_list) } , {

                name: 'EPICS' ,
                application: new Runtables_EPICS(experiment, access_list) }]} , {

            name: 'File Manager',
            menu: [{
                name: 'Summary' ,
                application: new Filemanager_Summary(experiment, access_list) } , {

                name: 'XTC HDF5' ,
                application: new Filemanager_Files(experiment, access_list) } , {

                name: 'USR' ,
                application: new Filemanager_Files_USR(experiment, access_list) }]} , {

            name: 'HDF5 Translation',
            menu: [{
                name: 'Manage' ,
                application: new HDF5_Manage(experiment, access_list) }]}) ;

        if (access_list.shiftmgr.can_edit) {
            menus.push ({
        
                name: 'Shift Manager',
                menu: [{
                    name: 'Reports' ,
                    application: new Reports(experiment.instrument.name, access_list.shiftmgr.can_edit) ,
                    html_container: 'shift-reports-'+experiment.instrument.name} , {

                    name: 'History' ,
                    application: new History(experiment.instrument.name),
                    html_container: 'shift-history-'+experiment.instrument.name} , {

                    name: 'E-mail Notifications' ,
                    application: new Notifications(experiment.instrument.namet) ,
                    html_container: 'shift-notifications-'+experiment.instrument.name}]}) ;
        }
    }

    Fwk.build (

        '<?=$title?>' ,
        '<?=$subtitle_url?>' ,

        menus ,

        function (text2search) {
            global_elog_search_message_by_text (text2search) ; } ,

        function () {
            Fwk.activate(select_app, select_app_context1) ; }
    ) ;
});

// Redirections which may be required by the legacy code generated
// by Web services.

function show_email (user, addr) { Fwk.show_email(user, addr) ; }

function global_elog_search_message_by_text (text2search) {
    var application = Fwk.activate('e-Log', 'Search') ;
    if (application) application.search_message_by_text(text2search) ;
    else console.log('global_elog_search_message_by_text(): not implemented') ;
} 

function global_elog_search_message_by_id (id, show_in_vicinity) {
    var application = Fwk.activate('e-Log', 'Search') ;
    if (application) application.search_message_by_id(id, show_in_vicinity) ;
    else console.log('global_elog_search_message_by_id(): not implemented') ;
} 

function global_elog_search_run_by_num (num, show_in_vicinity) {
    var application = Fwk.activate('e-Log', 'Search') ;
    if (application) application.search_run_by_num (num, show_in_vicinity) ;
    else console.log('global_elog_search_run_by_num(): not implemented') ;
}

function display_path(filepath) { Fwk.show_path(filepath) ; }

</script>

</head>

  <body>

      Loading...

<?php { $instr_name = $instrument->name() ; ?>

      <div id="shift-reports-<?php echo $instr_name ; ?>" style="display:none">

        <div class="shift-reports">

          <!-- Controls for selecting shifts for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-search-controls"  style="float:left;" >

            <div class="shifts-search-filters" >

              <div class="shifts-search-filter-group" >
                <div class="header" >Time range</div>
                <div class="cell-1" >
                  <select class="filter" name="range" style="padding:1px;">
                    <option value="week"  >Last 7 days</option>
                    <option value="month" >Last month</option>
                    <option value="range" >Specific range</option>
                  </select>
                </div>
                <div class="cell-2" >
                  <input class="filter" type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />
                  <input class="filter" type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" />
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Stopper out</div>
                <div class="cell-2">
                  <select class="filter" name="stopper" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="header-1">Door open</div>
                <div class="cell-2" >
                  <select class="filter" name="door" style="padding:1px;">
                    <option value="" ></option>
                    <option value="100" >&lt; 100 %</option>
                    <option value="99"  >&lt; 99 %</option>
                    <option value="98"  >&lt; 98 %</option>
                    <option value="97"  >&lt; 97 %</option>
                    <option value="96"  >&lt; 96 %</option>
                    <option value="95"  >&lt; 95 %</option>
                  </select>
                </div>
                <div class="header-1" >LCLS beam</div>
                <div class="cell-2">
                  <select class="filter"t name="lcls" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="header-1">Data taking</div>
                <div class="cell-2" >
                  <select class="filter" name="daq" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Shift types</div>
                <div class="cell-2">
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="USER"     title="if enabled it will include shifts of this type" /></div><div class="cell-4">USER</div>
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="MD"       title="if enabled it will include shifts of this type" /></div><div class="cell-4">MD</div>
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="IN-HOUSE" title="if enabled it will include shifts of this type" /></div><div class="cell-4">IN-HOUSE</div>
                </div>
                <div class="terminator" ></div>
              </div>

            </div>
            <div class="shifts-search-buttons" >
              <button name="reset"  title="reset the search form to the default state">RESET</button>
            </div>
            <div class="shifts-search-filter-terminator" ></div>
          </div>

<?php if ($shiftmgr_can_edit) { ?>

          <div id="new-shift-controls" style="float:left; margin-left:10px;">
            <button name="new-shift" title="open a dialog for creating a new shift" >CREATE NEW SHIFT</button>
            <div id="new-shift-con" class="new-shift-hdn" style="background-color:#f0f0f0; margin-top:5px; padding:1px 10px 5px 10px; border-radius:5px;" >
              <div style="max-width:460px;">
                <p>Note that shifts are usually created automatically based on rules defined
                in the Administrative section of this application. You may still want to create
                your own shift if that shift happens to be an exception from the rules.
                Possible cases would be: non-planned shift, very short shift, etc. In all
                other cases please see if there is a possibility to reuse an empty shift slot
                by checking "Display all shifts" checkbox on the left.</p>
              </div>
              <div style="float:left;">
                <table style="font-size:90%;"><tbody>
                  <tr>
                    <td class="shift-grid-hdr " valign="center" >Type:</td>
                    <td class="shift-grid-val " valign="center" >
                      <select name="type" >
                        <option value="USER"     >USER</option>
                        <option value="MD"       >MD</option>
                        <option value="IN-HOUSE" >IN-HOUSE</option>
                      </select></td>
                  </tr>
                  <tr>
                    <td class="shift-grid-hdr " valign="center" >Begin:</td>
                    <td class="shift-grid-val " valign="center" >
                      <input name="begin-day" type="text" size=8 title="specify the begin date of the shift" />
                      <input name="begin-h"   type="text" size=1 title="hour: 0..23" />
                      <input name="begin-m"   type="text" size=1 title="minute: 0..59" /></td>
                  </tr>
                  <tr>
                    <td class="shift-grid-hdr " valign="center" >End:</td>
                    <td class="shift-grid-val " valign="center" >
                      <input name="end-day" type="text" size=8 title="specify the end date of the shift" />
                      <input name="end-h"   type="text" size=1 title="hour: 0..23" />
                      <input name="end-m"   type="text" size=1 title="minute: 0..59" /></td>
                  </tr>
                </tbody></table>
              </div>
              <div style="float:left; margin-left:20px; margin-top:40px; padding-top:5px;">
                <button name="save"   title="submit modifications and open the editing dialog for the new shift">SAVE</button>
                <button name="cancel" title="discard modifications and close this dialog">CANCEL</button>
              </div>
              <div style="clear:both;"></div>
            </div>
          </div>

<?php } ?>

          <div style="clear:both;"></div>
          <div style="float:right;" id="shifts-search-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-search-display"> </div>

        </div>
      </div>

      <div id="shift-history-<?php echo $instr_name ; ?>" style="display:none">

        <div class="shift-history-reports">

          <!-- Controls for selecting an interval for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-history-controls" style="float:left;" >
            <div>
              <table><tbody>
                <tr>
                  <td><b>Range:</b></td>
                  <td><select name="range" style="padding:1px;">
                        <option value="week"  >Last 7 days</option>
                        <option value="month" >Last month</option>
                        <option value="range" >Specific range</option>
                      </select></td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td><input type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />
                      <b>&mdash;</b>
                      <input type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" /></td>
                  <td><div style="width:20px;">&nbsp;</div></td>
                  <td><button name="reset"  title="reset the search form to the default state">Reset</button></td>
                </tr>
              </tbody></table>
            </div>
            <div style="margin-top:5px;" >
              <table><tbody>
                <tr>
                  <td><b>Display:</b></td>
                  <td class="annotated"
                      data="if enabled the table below will display shift creation events">
                    <input type="checkbox"
                           name="display-create-shift"
                           checked="checked" />CREATE SHIFT</td>
                </tr>
                <tr>
                  <td>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display shift modifications">
                    <input type="checkbox"
                           name="display-modify-shift"
                           checked="checked" />MODIFY SHIFT</td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display area modifications">
                    <input type="checkbox"
                           name="display-modify-area"
                           checked="checked" />MODIFY AREA</td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display timer allocation modifications">
                    <input type="checkbox"
                           name="display-modify-time"
                           checked="checked" />MODIFY TIME ALLOCATION</td>
                </tr>
              </tbody></table>
            </div>
          </div>
          <div style="clear:both;"></div>
          <div style="float:right;" id="shifts-history-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-history-display"></div>

        </div>
      </div>

      <div id="shift-notifications-<?php echo $instr_name ; ?>" style="display:none">

        <div class="shift-notifications">
          View and manage push notifications: who will get an event and what kind of events (new shift created, data updated, etc.
        </div>

      </div>

<?php } ?>

  </body>

</html>

<?php
} catch (Exception $e) { print $e ; }
?>