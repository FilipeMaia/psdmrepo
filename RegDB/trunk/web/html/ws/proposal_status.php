<?php

/*
 * Return the data usage statistics for an experiment
 * 
 * AUTHORIZATION: not required
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    // Sanitize the value of the parameter by ensuring it has
    // the following format:
    // 
    //   'Lxxx'
    //   'xxx'
    //
    // Also remove an optional 'L' (low case) from the begininig of
    // the string.
    
    $proposal_number = strtolower($SVC->required_str('number')) ;
    switch (strlen($proposal_number)) {
        case 4:
            if ($proposal_number[0] != 'l') $SVC->abort("Illegal format of the proposal number: {$proposal_number}") ;
            $proposal_number = substr($proposal_number, 1) ;
        case 3:
            break ;
        default:
            $SVC->abort("Illegal length of the proposal number: {$proposal_number}") ;
    }

    // Find an experiment matching the proposal.
    foreach ($SVC->regdb()->experiment_names() as $exper_name) {
        if (preg_match("/^.{3}{$proposal_number}\d{2}$/", $exper_name)) {
            $exper = $SVC->safe_assign (
                $SVC->regdb()->find_experiment_by_unique_name($exper_name) ,
                "no experiment found for: {$exper_name}") ;

            $unix_group_members = array() ;
            foreach ($exper->group_members() as $user) {
                array_push($unix_group_members, array(
                    'uid'   => $user['uid'] ,
                    'gecos' => $user['gecos']
                )) ;
            }

            $group_managers = array() ;
            foreach ($SVC->authdb()->rolePlayers($exper->id(), 'LDAP', 'Admin') as $uid_or_group) {
                // Skip group-based managers. This can be added later if needed.
                if ('gid:' === substr($uid_or_group, 0, 4)) continue ;
                $user = $SVC->safe_assign (
                            $SVC->regdb()->find_user_account($uid_or_group) ,
                            "No user exists for UID: {$uid_or_group}") ;
                array_push($group_managers, array(
                    'uid'   => $user['uid'] ,
                    'gecos' => $user['gecos']
                )) ;
            }
            return array (
                'registered' => 1 ,
                'experiment' => array (
                    'instrument' => $exper->instrument()->name() ,
                    'name'       => $exper->name() ,
                    'id'         => $exper->id() ,
                    'portal_url' => "https://pswww.slac.stanford.edu/apps/portal?exper_id={$exper->id()}" ,
                    'contact'    => $exper->contact_info() ,
                    'unix_group'          => $exper->POSIX_gid() ,
                    'unix_group_members'  => $unix_group_members ,
                    'unix_group_managers' => $group_managers
                )
            ) ;
        }
    }
    
    return array (
        'registered' => 0
    ) ;
}) ;
