<?php

/**
 * This service will a dictionary of locations.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish(\Irep\IrepUtils::locations2array($SVC->irep())) ;
}) ;

?>
