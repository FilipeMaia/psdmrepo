<?php

/**
 * This service will update an existing shift at the specified hutch.
 * 
 * PARAMETERS:
 * 
 *   hutch (string), start_time (sec), end_time (sec), other_notes (string)
 */

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('POST', function ($SVC) {
    $id = $SVC->required_str('id');
    $hutch = $SVC->required_str('hutch');
    if (! $SVC->shiftmgr()->is_manager($hutch)) {
        $SVC->abort("your account is not authorized to update shifts for {$hutch}");
    }
    $start_time = $SVC->required_int('start_time');
    $end_time = $SVC->optional_int('end_time', 0);
    $other_notes = $SVC->optional_str('other_notes', '');

    $last_modified_time = $SVC->shiftmgr()->update_shift($id, $hutch, $start_time, $end_time, $other_notes);
    $SVC->finish(array('last_modified_time' => $last_modified_time));
});

?>
