<?php

/**
 * This service will return a list of cable number allocation.
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish(array('range' => $SVC->irep()->slacid_ranges()));
}) ;

?>
