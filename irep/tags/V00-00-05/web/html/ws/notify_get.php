<?php

/**
 * This service will return notification lists.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish (\Irep\IrepUtils::notifications2array($SVC->irep())) ;
}) ;
  
?>
