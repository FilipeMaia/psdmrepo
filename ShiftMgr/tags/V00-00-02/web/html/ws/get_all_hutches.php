<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $hutches = $SVC->shiftmgr()->get_all_hutches();
    $SVC->finish(array('hutches' => $hutches));
});

?>
