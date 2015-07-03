<?php

/*
 * Modify e-Log authorizations and return the updated status for the specified instrument.
 * 
 * PARAMETERS:
 * 
 *   <instr_name> auth2remove=<authspec>
 * 
 *   WHERE:
 * 
 *     authspec = [ {<exper_id>, <uid>, <role>} , ... ]
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    // ---------------------------------------------------
    // Get and verify the parameters's syntax and validity

    $instr_name  = $SVC->required_str ('instr_name') ;
    $auth2remove = $SVC->required_json('auth2remove') ;
 
    $instr = $SVC->safe_assign ($SVC->regdb()->find_instrument_by_name($instr_name) ,
                                "no such instrument '{$instr_name}'") ;
    $roles = array() ;
    if ($instr->is_standard()) {

        $operator_uid = strtolower($instr->name()).'opr' ;

        $account = $SVC->safe_assign ($SVC->regdb()->find_user_account($operator_uid) ,
                                      "no such operator account '{$operator_uid}'") ;

        $known_roles = array('Reader', 'Writer', 'Editor') ;

        foreach ($auth2remove as $spec) {

            $exper_id = intval($spec->exper_id) ;
            $uid      =        $spec->uid ;
            $role     =        $spec->role ;

            $experiment = $SVC->safe_assign ($SVC->regdb()->find_experiment_by_id($exper_id) ,
                                             "no such experiment id={$exper_id}") ;

            $SVC->assert (in_array($role, $known_roles) ,
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
            $can_manage_group = false ;
            foreach (array_keys($SVC->regdb()->experiment_specific_groups()) as $g) {
                if ($g === $experiment->POSIX_gid()) {
                    $can_manage_group = $SVC->regdbauth()->canManageLDAPGroup($g) ;
                    break ;
                }
            }
            $is_authorized =
                $is_data_administrator ||
                $can_manage_group ||
                $SVC->regdb()->is_member_of_posix_group('ps-'.strtolower($instr_name), $SVC->authdb()->authName()) ;

            $SVC->assert ($is_authorized ,
                          'not authorized for the operation') ;

            // ---------------------------------------------------------------------
            // Make sure the operation isn't being requested for an operator account
            // of an active experiment.

            if ($operator_uid === $uid)
                $SVC->assert (!$SVC->regdb()->is_active_experiment($exper_id) ,
                              "removing e-Log access isn't allowed for the operator accounts of active experiments") ;

            if ($SVC->authdb()->hasRole($uid, $exper_id, 'LogBook', $role))
                $SVC->authdb()->deleteRolePlayer('LogBook', $role, $exper_id, $uid) ;
        }
        
        // ---------------------------------
        // Find the remaining authorizations

        foreach ($instr->experiments() as $e)
            foreach ($known_roles as $role_name)
                if ($SVC->authdb()->hasRole($operator_uid, $e->id(), 'LogBook', $role_name))
                    array_push($roles, array (
                        'uid'   => $operator_uid ,
                        'role'  => $role_name ,
                        'exper' => array (
                            'name'      => $e->name() ,
                            'id'        => $e->id() ,
                            'is_active' => $SVC->regdb()->is_active_experiment($e->id())))) ;
    }
    return array('roles' => $roles) ;
}) ;
?>
