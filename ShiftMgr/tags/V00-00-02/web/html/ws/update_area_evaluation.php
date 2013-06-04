<?php

/**
 * This service will update a new shift at the specified hutch.
 * 
 * PARAMETERS:
 * 
 *   hutch (string), start_time (sec), end_time (sec), other_notes (string)
 */

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('POST', function ($SVC) {
    $id = $SVC->required_int('id');
    $area = $SVC->required_str('area');
    $ok = $SVC->required_bool('ok') ? 1 : 0;
    $downtime = $SVC->optional_int('downtime', 0);
    $comment = $SVC->optional_str('comment', '');
    $last_modified_time = $SVC->shiftmgr()->update_area_evaluation($id, $area, $ok, $downtime, $comment);
    $SVC->finish(array('last_modified_time' => $last_modified_time));
});

?>
