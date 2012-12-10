<?php

/**
 * This service will return a dictionary of statuses and related sub-statuses.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish(\Irep\IrepUtils::statuses2array($SVC->irep())) ;
}) ;

?>
