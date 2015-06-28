<?php

/*
 * MOdify the account's role for the e-Log and return the updated status
 * of the account for the specified experiment.
 * 
 * PARAMETERS:
 * 
 *   <exper_id> <uid> <role>
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    // ---------------------------------------------------
    // Get and verify the parameters's syntax and validity

    $exper_id = $SVC->required_int('exper_id') ;
    $uid      = $SVC->required_str('uid') ;
    $role     = $SVC->required_str('role') ;

    $experiment = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                     "no such experiment id={$exper_id}") ;

    $account = $SVC->safe_assign ($SVC->regdb()->find_user_account($uid) ,
                                  "no such account '{$uid}'") ;

    $known_roles = array('Reader', 'Writer', 'Editor') ;

    $SVC->assert ($role === 'NoAccess' || in_array($role, $known_roles) ,
                  "unknown role '{$role}'") ;

    // ------------------------------------------------------------
    // Make sure the caller has the group management privileges for
    // the experiment.

    $is_data_administrator = $SVC->authdb()->hasPrivilege (
        $SVC->authdb()->authName() ,
        null ,
        'StoragePolicyMgr' ,
        'edit'
    ) ;
    $experiment_can_manage_group = false ;
    foreach (array_keys($SVC->regdb()->experiment_specific_groups()) as $g) {
        if ($g === $experiment->POSIX_gid()) {
            $experiment_can_manage_group = $SVC->regdbauth()->canManageLDAPGroup($g) ;
            break ;
        }
    }
    $is_authorized =
        $is_data_administrator ||
        $experiment_can_manage_group ||
        (!$experiment->instrument()->is_standard() && $SVC->regdb()->is_member_of_posix_group('ps-'.strtolower($experiment->instrument()->name()), $SVC->authdb()->authName())) ;

    $SVC->assert ($is_authorized ,
                  'not authorized for the operation') ;

    // ---------------------------------------------------------------------
    // Make sure the operation isn't being requested for an operator account
    // of an active experiment.

    $operator_uid = $experiment->operator_uid() ;
    if ($operator_uid && ($operator_uid === $uid)) {
        if ($experiment->instrument()->is_standard() && $SVC->regdb()->is_active_experiment($exper_id))
            $SVC->assert (in_array($role, array('Writer', 'Editor')) ,
                          "downgrading e-Log access below 'Writer' isn't allowed for the operator accounts of active experiments") ;
    }

    foreach ($known_roles as $role_name)
        if ($SVC->authdb()->hasRole($uid, $experiment->id(), 'LogBook', $role_name))
            $SVC->authdb()->deleteRolePlayer('LogBook', $role_name, $experiment->id(), $uid) ;

    if ($role !== 'NoAccess')
        $SVC->authdb()->createRolePlayer('LogBook', $role, $experiment->id(), $uid) ;

    return array('role' => $role) ;
}) ;
?>
