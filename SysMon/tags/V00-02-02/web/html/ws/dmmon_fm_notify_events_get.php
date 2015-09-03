<?php

/*
 * REturn all known notification events for the file migration delays
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    if (PHP_VERSION_ID < 50400) {
        /*
         * JSON-ready object serialization control is provided
         * through a special interface JsonSerializable as
         * of PHP 5.4. The method jsonSerialize will return \stdClass
         * object with members ready for the JSON serialization.
         * Until that we have to call this method explicitly.
         */
        $events = array() ;
        foreach ($SVC->sysmon()->fm_delay_events() as $e)
            array_push($events, $e->jsonSerialize()) ;

        return array ('events' => $events) ;
    }
    return array (
        'events' => $SVC->sysmon()->fm_delay_events()) ;
}) ;

?>
