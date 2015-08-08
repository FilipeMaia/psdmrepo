<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once 'authdb/authdb.inc.php' ;
require_once 'irep/irep.inc.php' ;

use AuthDB\AuthDB ;

use Irep\Irep ;

$title = 'PCDS Inventory & Repair' ;
$subtitle = 'Electronic Equipment' ;


try {

    $authdb = AuthDB::instance() ;
    $authdb->begin() ;

    $irep = Irep::instance() ;
    $irep->begin() ;

    $known_apps = array (

        'equipment'             => array (
            'name'              => 'Equipment' ,
            'context1_default'  => 'Inventory' ,
            'context1'          => array (
                'inventory'     => 'Inventory' ,
                'add'           => 'Add New Equipment')
        ) ,
        'issues'                => array (
            'name'              => 'Issues' ,
            'context1_default'  => 'Search' ,
            'context1'          => array (
                'search'        => 'Search' ,
                'reports'       => 'Reports')
        ) ,
        'dictionary'            => array (
            'name'              => 'Dictionary' ,
            'context1_default'  => 'Equipment' ,
            'context1'          => array (
                'manufacturers' => 'Equipment' ,
                'locations'     => 'Locations' ,
                'statuses'      => 'Statuses')
        ) ,
        'admin'                 => array (
            'name'              => 'Admin' ,
            'context1_default'  => 'Access Control' ,
            'context1'          => array (
                'access'        => 'Access Control' ,
                'notifications' => 'E-mail Notifications' ,
                'slacid'        => 'SLACid Numbers')
        )
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
        $select_app = 'Equipment' ;
        $select_app_context1 = 'Inventory' ;
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
// procedure run after loading RequireJS.

var app_config = {

    title:    '<?=$title?>' ,
    subtitle: '<?=$subtitle?>' ,

    current_user : {
        uid:               '<?php echo $authdb->authName        () ;         ?>' ,
        is_other:           <?php echo $irep->is_other          ()?'1':'0' ; ?>  ,
        is_administrator:   <?php echo $irep->is_administrator  ()?'1':'0' ; ?>  ,
        can_edit_inventory: <?php echo $irep->can_edit_inventory()?'1':'0' ; ?>  ,
        has_dict_priv:      <?php echo $irep->has_dict_priv     ()?'1':'0' ; ?>
    } ,
    users:   [] ,
    editors: [] ,

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
        equipment_id: null
    }
} ;

<?php

    foreach ($irep->users() as $user) {
        echo "app_config.users.push('{$user->uid()}') ;\n" ;
        if ($user->is_administrator() || $user->is_editor()) echo "app_config.editors.push('{$user->uid()}') ;\n" ;
    }
    if (isset($_GET['equipment_id'])) {
        $equipment_id = intval(trim($_GET['equipment_id'])) ;
        if ($equipment_id) echo "app_config.select_params.equipment_id = {$equipment_id} ;\n" ;
    }

?>
 
</script>

<script data-main="../irep/js/index_main.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

</head>

<body></body>

</html>

<?php
} catch (Exception $e) {
    print <<<HERE
<span style="font-weight:bold; font-size:16px;">Error:     </span>Operation failed<br>
<span style="font-weight:bold; font-size:16px;">Exception: </span>{$e}<br>
HERE;
    return 1 ;
}
?>
