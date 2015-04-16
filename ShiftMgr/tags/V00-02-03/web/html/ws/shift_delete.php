<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {
    $shift_id = $SVC->required_int('shift_id') ;
    $SVC->shiftmgr()->delete_shift_by_id($shift_id) ;
    $SVC->finish() ;
}) ;

?>
