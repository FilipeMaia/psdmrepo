<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'shiftmgr/shiftmgr.inc.php';

\DataPortal\ServiceJSON::run_handler('POST', function ($SVC) {
    $hutch = $SVC->required_str('hutch');
    if (! $SVC->shiftmgr()->is_manager($hutch)) {
      // XXX should this instead be in shiftmgr.class.ph?
      $SVC->abort("your account is not authorized to create shifts for {$hutch}");
    }
    $start_time = $SVC->required_int('start_time');
    $end_time = $SVC->required_int('end_time');
    $created_shift = $SVC->shiftmgr()->create_shift($hutch, $start_time, $end_time);
    $SVC->finish(array('shift' => $created_shift));
});

?>
