<?php

/*
 * Return the status of the e-log autorization for the operator account
 * of an instrument.
 * 
 * PARAMETERS:
 * 
 *   <instr_name>
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;

    $instr = $SVC->safe_assign ($SVC->regdb()->find_instrument_by_name($instr_name) ,
                                "no such instrument '{$instr_name}'") ;

    $roles = array() ;
    if ($instr->is_standard()) {

        $operator_uid = strtolower($instr->name()).'opr' ;

        $account = $SVC->safe_assign ($SVC->regdb()->find_user_account($operator_uid) ,
                                      "no such operator account '{$operator_uid}'") ;

        $known_roles = array('Reader', 'Writer', 'Editor') ;

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
