<?php

/**
 * This service will return a dictionary of pinlists (drawings)
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'neocaptar/neocaptar.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $SVC->finish (array ('pinlist' => \NeoCaptar\NeoCaptarUtils::dict_pinlists2array ($SVC->neocaptar()))) ;
}) ;

?>
