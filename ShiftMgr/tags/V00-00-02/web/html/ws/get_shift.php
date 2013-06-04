<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $id = $SVC->required_int('id');
    $shift = $SVC->shiftmgr()->get_shift($id);
    $SVC->finish(array('shift' => $shift));
});

?>
