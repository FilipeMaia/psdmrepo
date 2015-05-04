<?php

/*
 * Report subscribers for the file migration delays
 * 
 * For complete documentation see JIRA ticket:
 * https://jira.slac.stanford.edu/browse/PSDH-35
 *
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    return array ('users' => $SVC->sysmon()->fm_delay_subscribers()) ;
}) ;

?>
