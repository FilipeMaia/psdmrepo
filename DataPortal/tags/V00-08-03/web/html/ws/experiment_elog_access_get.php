<?php

/*
 * Return the information about level of the specified account
 * to the e-Log of the experiment.
 * 
 * PARAMETERS:
 * 
 *   <exper_id> <uid>
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;
    $uid      = $SVC->required_str('uid') ;

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     "no such experiment id={$exper_id}") ;

    $account = $SVC->safe_assign ($SVC->regdb()->find_user_account($uid) ,
                                  "no such account '{$uid}'") ;

    $role = 'NoAccess' ;
    foreach (array('Reader', 'Writer', 'Editor') as $role_name)
        if ($SVC->authdb()->hasRole($uid, $experiment->id(), 'LogBook', $role_name)) {
            $role = $role_name ;
            break ;
        }

    return array('role' => $role) ;
}) ;
?>
