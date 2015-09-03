<?php

/**
 * This service will set/toggle the specified privilege of a user account
 * and return an updated list.
 * 
 * Parameters:
 * 
 *   <uid> <name>
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $SVC->irep()->is_administrator() or
        $SVC->abort('your account not authorized for the operation') ;

    $uid  = $SVC->required_str('uid') ;
    $name = $SVC->required_str('name') ;

    $user = $SVC->irep()->find_user_by_uid($uid) ;
    if (is_null($user)) $SVC->abort("no such user: {$uid}") ;

    switch ($name) {
        case 'dict_priv':
            $user->set_dict_priv(!$user->has_dict_priv()) ;
            break ;
        default:
            $SVC->abort("unknown privilege requested: {$name}") ;
    }
    $SVC->finish(array ('access' => \Irep\IrepUtils::access2array($SVC->irep()->users()))) ;
}) ;

?>
