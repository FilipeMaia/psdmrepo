<?php

/*
 * Update current user's subscription for the file migration delays
 * 
 * For complete documentation see JIRA ticket:
 * https://jira.slac.stanford.edu/browse/PSDH-35
 *
 */
require_once 'dataportal/dataportal.inc.php' ;


\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    
    $uid           = $SVC->required_str('uid') ;
    $is_subscribed = $SVC->required_int('is_subscribed') ;

    if ($is_subscribed) {

        // Allow a choice from any existing instrument as well as
        // an empty string for all instruments.

        $instr = $SVC->required_enum ('instr' ,
                                      array_merge (
                                          $SVC->regdb()->instrument_names() ,
                                          array('')) ,
                                      array('ignore_case' => true, 'convert' => 'toupper')) ;

        $last_sec  = $SVC->required_int ('last_sec') ;
        $delay_sec = $SVC->required_int ('delay_sec') ;

        // Depending on what already exists add a brand new subscription
        // or update an existing one.
        
        if ($SVC->sysmon()->find_fm_delay_subscriber($uid)) {
            $SVC->sysmon()->update_fm_delay_subscriber (
                $uid ,
                $SVC->authdb()->authName() ,
                $instr ,
                $last_sec ,
                $delay_sec
            ) ;
        } else {
            $SVC->sysmon()->add_fm_delay_subscriber (
                $uid ,
                $SVC->authdb()->authName() ,
                $instr ,
                $last_sec ,
                $delay_sec
            ) ;
        }
    } else {

        // Don't bother if the user is no longer subscribed.

        if ($SVC->sysmon()->find_fm_delay_subscriber($uid))
            $SVC->sysmon()->remove_fm_delay_subscriber ($uid) ;
    }

    return array ('users' => $SVC->sysmon()->fm_delay_subscribers()) ;
}) ;

?>
