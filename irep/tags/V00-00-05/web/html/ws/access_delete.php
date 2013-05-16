<?php

/**
 * This service will delete the specified account of a user account
 * and return an updated list.
 * 
 * Parameters:
 * 
 *   <uid>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->is_administrator() or
        $SVC->abort('your account not authorized for the operation') ;

    $uid  = $SVC->required_str('uid') ;

    $user = $SVC->irep()->find_user_by_uid($uid) ;
    if (is_null($user)) $SVC->abort("no such user: {$uid}") ;

    if ($user->uid() === $SVC->irep()->current_user()->uid())
        $SVC->abort('you can not delete your own account') ;

    $SVC->irep()->delete_user($user->uid()) ;

    $SVC->finish(array ('access' => \Irep\IrepUtils::access2array($SVC->irep()->users()))) ;
}) ;

?>
