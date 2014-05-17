<?php

/*
 * Return the information about candidate experiments for the switch
 *
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name = $SVC->required_str('instr_name') ;
    $station    = $SVC->required_int('station') ;

    $instr_group         = 'ps-'.strtolower($instr_name) ;
    $instr_group_members = array() ;
    foreach ($SVC->regdb()->posix_group_members($instr_group, /* $and_as_primary_group=*/ false) as $account)
        array_push (
            $instr_group_members ,
            array('uid' => $account['uid'], 'gecos' => $account['gecos'], 'email' => $account['email'])) ;

    $data_managers = array (
        array('uid' => 'perazzo', 'gecos' => 'Amedeo Perazzo', 'email' => 'perazzo@slac.stanford.edu') ,
        array('uid' =>   'gapon', 'gecos' => 'Igor Gaponenko', 'email' =>   'gapon@slac.stanford.edu') ,
        array('uid' =>   'wilko', 'gecos' =>  'Wilko Kroeger', 'email' =>   'wilko@slac.stanford.edu')
    ) ;

    $experiments = array() ;
    foreach ($SVC->logbook()->experiments_for_instrument($instr_name) as $exper)
        array_push($experiments, array (
            'id'      => $exper->id() ,
            'name'    => $exper->name() ,
            'contact' => DataPortal\DataPortal::experiment_contact_info($exper))) ;
    
    $SVC->finish (array (
        'current' => DataPortal\SwitchUtils::current($SVC, $instr_name, $station) ,
        'instr_group'         => $instr_group ,
        'instr_group_members' => $instr_group_members ,
        'data_managers'       => $data_managers ,
        'experiments'         => $experiments
    )) ;
}) ;
?>
