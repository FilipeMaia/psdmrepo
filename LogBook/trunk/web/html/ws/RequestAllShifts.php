<?php

/* This script will process a request for retreiving all shifts of the experiments.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LogBook\LogBookUtils ;
use LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id  = $SVC->required_int('exper_id') ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("experiment for id={$exper_id}") ;

    if (!$SVC->logbookauth()->canRead($experiment->id())) $SVC->abort('not authorized for the operation') ;

    $max_total_seconds = 1 ;    // -- this will nt include the current (active) shift
    $shifts = array() ;

    foreach ($experiment->shifts() as $s) {

        $total_seconds = $s->end_time() ?
            $s->end_time()->sec  - $s->begin_time()->sec :
            LusiTime::now()->sec - $s->begin_time()->sec ;

        // Exclude the current shift from consideration when calculating the maximum
        // duration of a shift

        if ($s->end_time() && ($total_seconds > $max_total_seconds)) $max_total_seconds = $total_seconds ;

        array_push (
            $shifts ,
            LogBookUtils::shift2array($s)
        ) ;
    }

    $SVC->finish (array (
        'Shifts'     => $shifts ,  // the newest shift should go first
        'MaxSeconds' => $max_total_seconds
    )) ;
}) ;

?>
