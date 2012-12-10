<?php

/**
 * Search and return all known history events for the specified (by its identifier)
 * cable.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'neocaptar/neocaptar.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id = $SVC->required_int ('id') ;

    $cable = $SVC->neocaptar()->find_cable_by_id ($id) ;
    if (is_null($cable)) $SVC->abort ("cable not found for id: {$id}") ;

    $events2return = array () ;
    foreach ($cable->history() as $e)
        array_push ($events2return, \NeoCaptar\NeoCaptarUtils::event2array ($e)) ;

    $SVC->finish (array ('event' => $events2return )) ;
}) ;

?>
