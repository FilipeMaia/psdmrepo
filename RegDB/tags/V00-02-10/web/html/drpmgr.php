<?php

require_once 'authdb/authdb.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use AuthDB\AuthDB ;
use RegDB\RegDB ;

$title    = 'System Management' ;
$subtitle = 'Data Retention Policy' ;

RegDB::instance()->begin() ;
$instruments = array() ;
foreach (RegDB::instance()->instruments() as $instr) {
    if ($instr->is_location()) continue ;
    array_push($instruments, $instr->name()) ;
}

$select_app          = 'Policies' ;
$select_app_context1 = 'General' ;

if (isset( $_GET['app'])) {
    $app_path   = explode(':', trim($_GET['app'])) ;
    $select_app = $app_path[0] ;
    if (count($app_path) > 1)
        $select_app_context1 = $app_path[1] ;
}

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

<script data-main="../regdb/js/drpmgr_main.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

</head>
<body></body>
</html>

