<?php

/**
 * This service will return an access control lists.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'neocaptar/neocaptar.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish (array (
        'access' => \NeoCaptar\NeoCaptarUtils::access2array ($SVC->neocaptar()->users())
    ));
}) ;

?>
