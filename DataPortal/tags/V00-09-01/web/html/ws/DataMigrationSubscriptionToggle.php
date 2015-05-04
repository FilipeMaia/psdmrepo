<?php

/* The script will toggle subscriptions of the authenticated user to receive
 * e-mail notifications on delayed file migration.
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $user  = $SVC->authdb()->authName() ;
    $email = $user.'@slac.stanford.edu' ;

    $SVC->configdb()->subscribe4migration_if (
        is_null($SVC->configdb()->check_if_subscribed4migration($user, $email)) ,
        $user ,
        $email
    ) ;
}) ;
?>