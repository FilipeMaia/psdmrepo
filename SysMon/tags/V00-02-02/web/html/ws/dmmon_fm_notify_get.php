<?php

/*
 * Report subscribers for the file migration delays
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $users = array() ;
    foreach ($SVC->sysmon()->fm_delay_subscribers() as $s) {
        array_push($users, $s->jsonSerialize()) ;
    }
    return array ('users' => $users) ;
}) ;

?>
