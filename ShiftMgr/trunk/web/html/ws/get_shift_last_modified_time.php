<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $id = $SVC->required_int('id');
    $last_modified_time = $SVC->shiftmgr()->get_shift_last_modified_time($id);
    $SVC->finish(array('last_modified_time' => $last_modified_time));
});

?>
