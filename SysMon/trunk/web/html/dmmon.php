<?php

require_once 'authdb/authdb.inc.php' ;

use AuthDB\AuthDB ;

$title    = 'System Monitoring' ;
$subtitle = 'Data Movers' ;

$instruments = array('AMO','SXR','XPP','XCS','CXI','MEC') ;

$select_app          = $instruments[0] ;
$select_app_context1 = "Live" ;
?>

<!DOCTYPE html>
<html>

<head>

<title><?="{$title} : {$subtitle}"?></title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link rel="icon" href="../webfwk/img/Portal_favicon.ico"/>

<script type="text/javascript">

// Application configuration needs to be passed to the Fwk initialization
// procedure run upon loading RequireJS.

var app_config = {

    title:    '<?= $title ?>' ,
    subtitle: '<?= $subtitle ?>' ,

    uid: '<?= AuthDB::instance()->authName() ?>' ,

    instruments: <?php echo json_encode($instruments) ?> ,

    no_page_access_html:
'<br><br>' +
'<center>' +
'  <span style="color: red; font-size: 175%; font-weight: bold; font-family: Times, sans-serif;">' +
'    A c c e s s &nbsp; E r r o r' +
'  </span>' +
'</center>' +
'<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">' +
'  We\'re sorry! Our records indicate that your SLAC UNIX account has no proper permissions to access this page.' +
'</div>' ,

    select_app:          '<?=$select_app?>' ,
    select_app_context1: '<?=$select_app_context1?>' ,

    select_params: {
    }
} ;

</script>

<script data-main="../sysmon/js/dmmon_main.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

</head>
<body></body>
</html>

