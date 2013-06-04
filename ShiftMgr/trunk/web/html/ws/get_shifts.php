<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $hutch = $SVC->optional_str('hutch', '');
    $earliest_start_time = $SVC->optional_int('earliest_start_time', 0);
    $latest_start_time = $SVC->optional_int('latest_start_time', 0);
    $shifts = $SVC->shiftmgr()->get_shifts($hutch, $earliest_start_time, $latest_start_time);
    $SVC->finish(array('shifts' => \ShiftMgr\ShiftMgrUtils::shifts2array($shifts)));
});

?>
