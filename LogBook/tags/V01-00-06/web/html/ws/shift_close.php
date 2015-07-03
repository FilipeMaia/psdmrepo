<?php

/**
 * This script will process a request for closing an on-going shift.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $shift_id = $SVC->required_int ('id') ;

    $shift = LogBook::instance()->find_shift_by_id($shift_id) ;
    if (!$shift) $SVC->abort("no shift found for id={$shift_id}") ;

    if (!$SVC->logbookauth()->canManageShifts($shift->parent()->id()))
        $SVC->abort('You are not authorized to manage shifts of the experiment') ;

    $shift->close( LusiTime::now()) ;

    $SVC->finish() ;
}) ;

?>
