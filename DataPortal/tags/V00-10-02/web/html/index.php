<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

// Let a user to select an experiment first if no valid experiment
// identifier is supplied to the script.

function select_experiment () {
    header("Location: select_experiment.php") ;
    exit ;
}

$exper_id = isset($_GET['exper_id']) ? intval(trim($_GET['exper_id'])) : 0 ;
if (!$exper_id) select_experiment() ;

// Redirections to support legacy URLs.

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
            'daq'              => 'DAQ' ,
            'epics'            => 'EPICS' ,
            'user'             => 'User' )) ,

    'datafiles' => array (
        'name'             => 'File Manager' ,
        'context1_default' => 'Summary' ,
        'context1'         => array (
            'summary'          => 'Summary' ,
            'files'            => 'XTC HDF5' )) ,

    'hdf' => array (
        'name'             => 'HDF5 Translation' ,
        'context1_default' => 'Standard' ,
        'context1'         => array (
            'standard'         => 'Standard' ,
            'config'           => 'Monitoring' )) ,

    'shiftmgr' => array (
        'name'             => 'Hutch Manager' ,
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

// Parse optional parameters which may be used by applications. The parameters
// will be passed directly into applications for further analysis (syntax, values,
// etc.).

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

    $title        = $experiment->is_facility() ? 'E-Log of Facility:' : 'Data Manager of Experiment' ;
    $subtitle     = $experiment->instrument()->name().'/'.$experiment->name() ;
    $subtitle_url = '<a class="link" href="select_experiment.php" title="Switch to another experiment">'.$experiment->instrument()->name().'&nbsp;/&nbsp;'.$experiment->name().'</a>' ;

    $user = RegDB::instance()->find_user_account(AuthDb::instance()->authName()) ;
    if (!$user) die("Sorry, can't safely run this application due to a broken authentication system.") ;

    $experiment_can_manage_group = false ;
    foreach (array_keys( RegDB::instance()->experiment_specific_groups()) as $g) {
        if ($g === $experiment->POSIX_gid()) {
            $experiment_can_manage_group = RegDBAuth::instance()->canManageLDAPGroup($g) ;
            break ;
        }
    }

    $operator_uid = $experiment->operator_uid() ;

    $elog_can_read_messages   = LogBookAuth::instance()->canRead           ($logbook_experiment->id()) ;
    $elog_can_post_messages   = LogBookAuth::instance()->canPostNewMessages($logbook_experiment->id()) ;
    $elog_can_edit_messages   = LogBookAuth::instance()->canEditMessages   ($logbook_experiment->id()) ;
    $elog_can_delete_messages = LogBookAuth::instance()->canDeleteMessages ($logbook_experiment->id()) ;
    $elog_can_manage_shifts   = LogBookAuth::instance()->canManageShifts   ($logbook_experiment->id()) ;

    $is_member_of_instr_group = !$experiment->is_facility() && RegDB::instance()->is_member_of_posix_group('ps-'.strtolower($instrument->name()), AuthDb::instance()->authName()) ;
    $is_member_of_data_group  = RegDB::instance()->is_member_of_posix_group('ps-data', AuthDb::instance()->authName()) ;

    $calibrations_can_edit = 
        $elog_can_post_messages ||
        $is_member_of_data_group ||
        $is_member_of_instr_group ;

    $experiment_can_read_data =
        $experiment_can_manage_group ||
        $elog_can_read_messages || $elog_can_post_messages || $elog_can_edit_messages || $elog_can_delete_messages || $elog_can_manage_shifts ||
        $calibrations_can_edit ||
        $is_member_of_data_group ||
        RegDB::instance()->is_member_of_posix_group($logbook_experiment->POSIX_gid(), AuthDb::instance()->authName()) ||
        $is_member_of_instr_group ;
    
    $hdf5_can_retranslate =
        $is_data_administrator ||
        $is_member_of_data_group ||
        $is_member_of_instr_group ||
        $elog_can_post_messages ;

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

<link rel="icon" href="../webfwk/img/Portal_favicon.ico"/>

<!--The old-style Table is still required to display experiment status (CSS only),
    group management table (CSS only), and Run Tables for 'Detectors' and 'Calibratons'.
    The later two require the old Table class.

    TODO: migrate them all to the SimpleTable widget.-->

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />
<script type="text/javascript" src="../webfwk/js/Table.js"></script>

<script type="text/javascript">

// Application configuration needs to be passed to the Fwk initialization
// procedure run upon loading RequireJS.

var app_config = {

    title :        '<?=$title?>' ,
    subtitle_url : '<?=$subtitle_url?>' ,
        
    experiment : {
        id           :  <?= $experiment->id() ?> ,
        name         : '<?= $experiment->name() ?>' ,
        contact_uid  : '<?= $experiment->leader_Account() ?>' ,
        posix_group  : '<?= $experiment->POSIX_gid() ?>' ,
        is_facility  :  <?= $experiment->is_facility() ? 1 : 0 ?> ,
        instrument   : {
            id       :  <?= $instrument->id()   ?> ,
            name     : '<?= $instrument->name() ?>'
        } ,
        operator_uid : '<?= $operator_uid ? $operator_uid : '' ?>'
    } ,

    access_list : {
        user : {
            uid   : '<?= $user["uid"]   ?>' ,
            gecos : '<?= htmlspecialchars($user["gecos"], ENT_QUOTES) ?>' ,
            email : '<?= htmlspecialchars($user["email"], ENT_QUOTES) ?>'
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
            read              : <?= $experiment_can_read_data ? 1 : 0 ?> ,
            edit_calibrations : <?= $calibrations_can_edit    ? 1 : 0 ?> ,
            edit              : <?= $elog_can_post_messages   ? 1 : 0 ?>
        } ,
        datafiles : {
            read                  : <?= $experiment_can_read_data ? 1 : 0 ?> ,
            manage                : <?= $elog_can_post_messages   ? 1 : 0 ?> ,
            is_data_administrator : <?= $is_data_administrator    ? 1 : 0 ?>
        } ,
        hdf5 : {
            read                  : <?= $experiment_can_read_data ? 1 : 0 ?> ,
            manage                : <?= $elog_can_post_messages   ? 1 : 0 ?> ,
            is_data_administrator : <?= $hdf5_can_retranslate     ? 1 : 0 ?>
        } ,
        shiftmgr : {
            can_edit : <?= $shiftmgr_can_edit ? 1 : 0 ; ?>
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
'</div>'} ,

    select_app : '<?=$select_app?>' ,
    select_app_context1 : '<?=$select_app_context1?>' ,

    global_extra_params : {
<?php
    if (isset($params)) {
        foreach ($params as $p) {
            $kv = explode(':', $p) ;
            switch (count($kv)) {
            case 0 :
                break;
            case 1 :
                $k = $kv[0] ;
                echo "'{$k}' : true ,\n" ;
                break ;
            default :
                $k = $kv[0] ;
                $v = $kv[1] ;
                echo "'{$k}' : '{$v}' ,\n" ;
                break ;
            }
        }
    }
?>
    }
} ;

</script>

<script data-main="../portal/js/index_main.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

</head>
<body></body>
</html>

<?php
} catch (Exception $e) { print $e ; }
?>