<?php

require_once 'dataportal/dataportal.inc.php';

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $masks = $SVC->exptimemon()->beam_destination_masks;
    $SVC->finish(array('masks' => $masks));
});

?>
