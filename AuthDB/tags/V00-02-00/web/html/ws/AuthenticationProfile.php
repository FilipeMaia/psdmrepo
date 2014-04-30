<?php

/**
 * This service will return authentication profile of a logged user.
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $profile = array (
        'is_authenticated'         => $SVC->authdb()->isAuthenticated() ? 1 : 0 ,
        'type'                     => $SVC->authdb()->authType() ,
        'user'                     => $SVC->authdb()->authName() ,
        'webauth_token_creation'   => $_SERVER['WEBAUTH_TOKEN_CREATION'] ,
        'webauth_token_expiration' => $_SERVER['WEBAUTH_TOKEN_EXPIRATION']
    ) ;
    $SVC->finish(array('profile'=>$profile)) ;
}) ;

?>
