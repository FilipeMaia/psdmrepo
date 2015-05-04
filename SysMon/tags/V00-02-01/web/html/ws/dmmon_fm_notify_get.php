<?php

/*
 * Report subscribers for the file migration delays
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
        $users = array() ;
        foreach ($SVC->sysmon()->fm_delay_subscribers() as $s)
            array_push($users, $s->jsonSerialize()) ;

        return array ('users' => $users) ;
    }
    return array (
        'users' => $SVC->sysmon()->fm_delay_subscribers()) ;
}) ;

?>
