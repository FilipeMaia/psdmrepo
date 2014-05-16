<?php

/*
 * Activate the specified experiment at an instrument and a station
 *
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $instr_name = $SVC->required_str ('instr_name') ;
    $station    = $SVC->required_int ('station') ;
    $exper_id   = $SVC->required_int ('exper_id') ;
    $message    = $SVC->optional_str ('message', '') ;
    $notify     = $SVC->required_JSON('notify') ;

    // Process and verify the input parameters 

    $instr = $SVC->regdb()->find_instrument_by_name($instr_name) ;
    if (!$instr) $SVC->abort("no such instrument found: '{$instr_name}'") ;

    $num_stations_obj = $instr->find_param_by_name('num_stations') ;
    if (!$num_stations_obj || !$num_stations_obj->value())
        $SVC->abort("instrument {$instr->name()} is not properly configured") ;

    $num_stations = intval($num_stations_obj->value()) ;
    if ($station >= $num_stations)
        $SVC->abort("no such station number {$station} at instrument {$instr->name()}") ;

    $exper = $SVC->logbook()->find_experiment_by_id($exper_id) ;
    if (!$exper)
        $SVC->abort("the active experiment with id={$exper_id} is no longer registered") ;

    $notify_list = array() ;
    foreach ($notify as $n)
        array_push($notify_list, array(
            'uid'      => $n->uid ,
            'gecos'    => $n->gecos ,
            'email'    => $n->email ,
            'rank'     => $n->rank ,
            'notified' => $n->notified)) ;

    // Check for proper authorizations

    $requestor_uid = $SVC->authdb()->authName() ;
    $requestor_account = $SVC->regdb()->find_user_account($requestor_uid) ;
    if (!$requestor_account)
        $SVC->abort("user account not found for: {$requestor_uid}") ;

    $requestor_gecos = $requestor_account['gecos'] ;
    $requestor_email = $requestor_account['email'] ;

    $appl = 'ExperimentSwitch' ;
    $role = 'Manage' ;
    $role_inst = "{$role}_{$instr->name()}" ;

    $can_manage =
        $SVC->authdb()->hasRole($requestor_uid, $exper_id, $appl, $role) ||
        $SVC->authdb()->hasRole($requestor_uid, null,      $appl, $role) ||
        $SVC->authdb()->hasRole($requestor_uid, $exper_id, $appl, $role_inst) ||
        $SVC->authdb()->hasRole($requestor_uid, null,      $appl, $role_inst) ;

    if (!$can_manage)
        $SVC->abort('You are not authorized to manage the experiment switch') ;
    
    // Proceed to the switch only if the new experiment differs from
    // the presently active one.
    
    $prev_switch   = $SVC->regdb()->last_experiment_switch($instr->name(), $station);
    $prev_exper_id = $prev_switch ? intval($prev_switch['exper_id']) : 0 ;

    if ($prev_exper_id != $exper_id) {

        $prev_exper = $SVC->logbook()->find_experiment_by_id($prev_exper_id) ;
        if (!$prev_exper)
            $SVC->abort("the previous experiment with id={$prev_exper_id} is no longer registered") ;

        // This is the actual switch

        $SVC->regdb()->switch_experiment($exper->name(), $station, $requestor_uid, $notify_list );

        // Make proper adjustments to the OPR account access privileges
        //
        // 1. remove authorizations for the prior experiment
        // 2. add the 'Writer' role for the new one
        //

        $opr_account = strtolower($instr->name()).'opr' ;
        $appl = 'LogBook' ;
        foreach (array('Reader', 'Writer', 'Editor') as $role)
            if ($SVC->authdb()->hasRole($opr_account, $prev_exper_id, $appl, $role))
                $SVC->authdb()->deleteRolePlayer($appl, $role, $prev_exper_id, $opr_account) ;

        $SVC->authdb()->createRolePlayer($appl, 'Writer', $exper_id, $opr_account) ;
        
        DataPortal\SwitchUtils::notify (
            $instr ,
            $station ,
            $prev_exper ,
            $exper ,
            $message ,
            $requestor_gecos ,
            $requestor_email ,
            $notify_list) ;
    }
    $SVC->finish (array (
        'current' => DataPortal\SwitchUtils::current($SVC, $instr_name, $station) ,
    )) ;
}) ;
?>
