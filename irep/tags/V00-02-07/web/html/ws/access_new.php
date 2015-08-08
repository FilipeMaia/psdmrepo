<?php

/**
 * This service will create a new user and return an updated access control list.
 * 
 * Parameters:
 * 
 *   <uid> <role>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->is_administrator() or
        $SVC->abort('your account not authorized for the operation') ;

    $uid  = $SVC->required_str('uid') ;
    $role = $SVC->required_str('role') ;

    $user = $SVC->regdb()->find_user_account($uid) ;
    if (is_null($user)) $SVC->abort("no such user: {$uid}") ;

    $SVC->irep()->add_user($user['uid'], $user['gecos'], $role) ;

    $SVC->finish(array ('access' => \Irep\IrepUtils::access2array($SVC->irep()->users()))) ;
}) ;

?>
