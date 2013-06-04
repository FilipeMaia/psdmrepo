<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $uses = $SVC->shiftmgr()->get_uses();
    $SVC->finish(array('uses' => $uses));
});

?>
