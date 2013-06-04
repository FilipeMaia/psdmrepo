<?php

require_once 'dataportal/dataportal.inc.php';
require_once 'lusitime/lusitime.inc.php';

use LusiTime\LusiTime;

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $pv = $SVC->required_str('pv');
    $start = new LusiTime($SVC->required_int('start'));
    $stop = new LusiTime($SVC->required_int('stop'));
    $values = $SVC->exptimemon()->beamtime_beam_status($pv, $start, $stop);
    $SVC->finish(array('values' => $values));
});

?>
