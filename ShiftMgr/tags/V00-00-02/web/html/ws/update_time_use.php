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
    $use_name = $SVC->required_str('use_name');
    $use_time = $SVC->optional_int('use_time', 0);
    $comment = $SVC->optional_str('comment', '');
    $last_modified_time = $SVC->shiftmgr()->update_time_use($id, $use_name, $use_time, $comment);
    $SVC->finish(array('last_modified_time' => $last_modified_time));
});

?>
