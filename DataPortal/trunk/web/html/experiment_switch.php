<?php

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
            'access_list'  => array (
                'can_manage' => $can_manage ? 1 : 0 ,
                'can_read'   => $can_manage ? 1 : 0))) ;

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

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />
<link type="text/css" href="/jquery/css/jquery-ui-timepicker-addon.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Fwk.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/PropList.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/CheckTable.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/ExpSwitch_Station.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/ExpSwitch_History.css" rel="Stylesheet" />


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

button {
  background: rgba(240, 248, 255, 0.39) !important;
  border-radius: 2px !important;
}

</style>

<script type="text/javascript" src="/jquery/js/jquery-1.8.2.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-timepicker-addon.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.resize.js"></script>

<script type="text/javascript" src="/underscore/underscore-min.js"></script>

<script type="text/javascript" src="../webfwk/js/Class.js" ></script>
<script type="text/javascript" src="../webfwk/js/Widget.js" ></script>
<script type="text/javascript" src="../webfwk/js/Fwk.js"></script>
<script type="text/javascript" src="../webfwk/js/Table.js"></script>
<script type="text/javascript" src="../webfwk/js/PropList.js"></script>
<script type="text/javascript" src="../webfwk/js/CheckTable.js"></script>

<script type="text/javascript" src="../portal/js/ExpSwitch_Station.js"></script>
<script type="text/javascript" src="../portal/js/ExpSwitch_History.js"></script>

<script type="text/javascript">

var select_app = '<?=$select_instr_name?>' ;
var select_app_context1 = 'Station <?=$select_station?>' ;

var instruments = <?=json_encode($instruments)?> ;

$(function() {

    var menus = [] ;
    for (var i in instruments) {

        var instr = instruments[i] ;
        var instr_tab = {
            name: instr.name ,
            menu: []} ;

        for (var station = 0; station < instr.num_stations; ++station)
            instr_tab.menu.push ({
                name: 'Station '+station ,
                application: new ExpSwitch_Station(instr.name, station, instr.access_list)}) ;

            instr_tab.menu.push ({
                name: 'History' ,
                application: new ExpSwitch_History(instr.name,  instr.access_list)}) ;

        menus.push(instr_tab) ;
    }
    Fwk.build (

        '<?=$title?>' ,
        'Activate Experiments for DAQ' ,

        menus ,

        null ,  // no quick search for this application

        function () {
            Fwk.activate(select_app, select_app_context1) ; }
    ) ;
}) ;

// Redirections which may be required by the legacy code generated
// by Web services.

function show_email (user, addr) { Fwk.show_email(user, addr) ; }

</script>
</head>
<body>
</body>
</html>

<?php
} catch (Exception $e) {
    report_error('Operation failed', $e) ;
}
?>
