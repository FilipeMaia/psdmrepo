<?php
# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once 'authdb/authdb.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use AuthDB\AuthDB ;
use RegDB\RegDB ;

function report_error($msg, $ex=null) {
    print          '<span style="font-weight:bold; font-size:16px;">Error:     </span>'.$msg.'<br>' ;
    if ($ex) print '<span style="font-weight:bold; font-size:16px;">Exception: </span>'.$ex.'<br>' ;
}

$title = 'Experiment Switch' ;

try {

    AuthDB::instance()->begin() ;
    RegDB::instance()->begin() ;

    // Parse and evaluate optional parameters

    $select_instr_name = isset($_GET['instr_name']) ? trim  ($_GET['instr_name']) : '' ;
    $select_station    = isset($_GET['station'])    ? intval($_GET['station'])    : 0 ;
    
    if ($select_instr_name && !RegDB::instance()->find_instrument_by_name($select_instr_name))
        report_error('unknown instrument name provided to the script') ;

    $no_page_access_html = <<<HERE
<br><br>
<center>
  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">
    A c c e s s &nbsp; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">
  We\'re sorry! Our records indicate that your SLAC UNIX account has no proper permissions to access this page.
</div>
HERE;

    $instruments = array() ;
    foreach (RegDB::instance()->instruments() as $instr) {

        if( !$instr->is_standard()) continue ;

        $num_stations = $instr->find_param_by_name('num_stations') ;
        if (!$num_stations || !$num_stations->value())
            report_error("instrument {$instr->name()} is not properly configured" ) ;

        // Permission to change active experiments

        $can_manage =
            RegDB::instance()->is_member_of_posix_group('ps-data', AuthDb::instance()->authName()) ||
            RegDB::instance()->is_member_of_posix_group('ps-'.strtolower($instr->name()), AuthDb::instance()->authName()) ||
            AuthDb::instance()->hasPrivilege (
                AuthDb::instance()->authName() ,        // user
                null ,                                  // any instrument/experiment
                "ExperimentSwitch_{$instr->name()}" ,   // role
                'change') ;                             // privilege

        array_push($instruments, array (
            'name'         => $instr->name() ,
            'num_stations' => intval($num_stations->value()) ,
            'operator_uid' => strtolower($instr->name()).'opr' ,
            'access_list'  => array (
                'can_manage'          => $can_manage ? 1 : 0 ,
                'can_read'            => $can_manage ? 1 : 0) ,
                'no_page_access_html' => $no_page_access_html)) ;

        if ($select_instr_name) {
            $select_instr_name = $instr->name() ;
            $select_station = 0 ;
        }
    }
?>

<!DOCTYPE html>
<html>

<head>

<title><?=$title?></title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link rel="icon" href="../webfwk/img/Portal_favicon.ico"/>

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="../webfwk/js/Table.js"></script>

<script type="text/javascript">

// Application configuration needs to be passed to the Fwk initialization
// procedure run after loading RequireJS.

var app_config = {

    title : '<?=$title?>' ,
    subtitle : 'Activate Experiments for DAQ' ,

    select_app : '<?=$select_instr_name?>' ,
    select_app_context1 : 'Station <?=$select_station?>' ,

    instruments : <?=json_encode($instruments)?>
} ;

</script>

<script data-main="../portal/js/experiment_switch_main.js" src="/require/require.js"></script>

</head>

<body></body>

</html>

<?php
} catch (Exception $e) {
    report_error('Operation failed', $e) ;
}
?>
