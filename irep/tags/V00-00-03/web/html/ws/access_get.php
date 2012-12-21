<?php

/**
 * This service will return an access control lists.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish(array ('access' => \Irep\IrepUtils::access2array($SVC->irep()->users()))) ;
}) ;

?>
