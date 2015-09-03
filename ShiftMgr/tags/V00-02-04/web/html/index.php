<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once 'authdb/authdb.inc.php' ;

use AuthDb\AuthDb ;

AuthDb::instance()->begin() ;

$instruments = array('AMO','SXR','XPP','XCS','CXI','MEC') ;
$instr2editor = array() ;
foreach ($instruments as $instr_name) {
    $instr2editor[$instr_name] =
        AuthDb::instance()->hasRole (
            AuthDb::instance()->authName() ,
            null ,
            'ShiftMgr' ,
            "Manage_{$instr_name}"
        ) ;
}
?>

<!DOCTYPE html>
<html>

<head>

<title>Shift Manager</title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />
<link type="text/css" href="/jquery/css/jquery-ui-timepicker-addon.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery-1.8.2.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-timepicker-addon.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="../webfwk/js/Table.js"></script>

<script type="text/javascript">

// Application configuration needs to be passed to the Fwk initialization
// procedure run upon loading RequireJS.

var app_config = {

    title :    'PCDS Shift Manager' ,
    subtitle : 'Instrument Hutches' ,

    instruments :  <?php echo json_encode($instruments) ?> ,
    instr2editor : <?php echo json_encode($instr2editor) ?>
} ;

</script>

<script data-main="../shiftmgr/js/index_main.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

</head>
<body></body>
</html>

