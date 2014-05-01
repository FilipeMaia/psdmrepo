<?php

/**
 * This script will process a request for creating a new shift.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LogBook\LogBookUtils ;
use LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $exper_id   = $SVC->required_int ('exper_id') ;
    $leader_uid = $SVC->optional_str ('leader', $SVC->authdb()->authName()) ;
    $author_uid = $leader_uid ;
    $goals      = $SVC->optional_str ('goals', '') ;
    $crew       = array() ;

    $experiment = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$experiment) $SVC->abort("no experiment found for id={$exper_id}") ;

    if (!$SVC->logbookauth()->canManageShifts($experiment->id()))
        $SVC->abort('You are not authorized to manage shifts of the experiment') ;


    $begin_time = LusiTime::now() ;

    $shift = $experiment->create_shift($leader_uid, $crew, $begin_time) ;

    $entry = $experiment->create_entry($author_uid, 'TEXT', $goals, $shift->id()) ;
    $entry->add_tag('SHIFT_GOALS', '') ;

    $SVC->finish (
        array (
            'ResultSet' => array (
                'Result' => array(LogBookUtils::shift2array($shift))
            )
        )
   ) ;
}) ;
?>
