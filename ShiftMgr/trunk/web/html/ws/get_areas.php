<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $areas = $SVC->shiftmgr()->get_areas();
    $SVC->finish(array('areas' => $areas));
});

?>
