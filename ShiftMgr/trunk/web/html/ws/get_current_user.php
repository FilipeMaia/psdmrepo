<?php
require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $current_user = $SVC->shiftmgr()->current_user();
    $SVC->finish(array('current_user' => $current_user));
});
?>
