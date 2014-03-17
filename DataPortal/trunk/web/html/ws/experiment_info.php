<?php

/*
 * Return the information about an experiment.
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id   = $SVC->optional_int('id',   null) ;
    $name = $SVC->optional_str('name', null) ;

    if     ($id)   $logbook_experiment = $SVC->logbook()->find_experiment_by_id($id) ;
    elseif ($name) $logbook_experiment = $SVC->logbook()->find_experiment_by_name($name) ;
    else           $SVC->abort('no experiment identity parameter found in the requst') ;

    if (!$logbook_experiment) $SVC->abort('no experiment found for '.($id ? "id={$id}" : "name={$name}")) ;

    $experiment = $logbook_experiment->regdb_experiment() ;
    $instrument = $experiment->instrument() ;

    $leader_uid     = $experiment->leader_account() ;
    $leader_account = $SVC->regdb()->find_user_account($leader_uid) ;
    $leader_gecos   = $leader_account['gecos'] ;
    $leader_email   = $leader_account['email'] ;
    
    $num_runs   = $logbook_experiment->num_runs() ;
    $first_run  = $logbook_experiment->find_first_run() ;
    $last_run   = $logbook_experiment->find_last_run() ;

    $first_run_begin_time = $first_run ? $first_run->begin_time() : null ;
    $first_run_end_time   = $first_run ? $first_run->end_time()   : null ;
    $last_run_begin_time  = $last_run  ? $last_run->begin_time()  : null ;
    $last_run_end_time    = $last_run  ? $last_run->end_time()    : null ;

    $last_entry = $logbook_experiment->find_last_entry() ;
        
    $SVC->finish (array (

        'id'          => $experiment->id() ,
        'name'        => $experiment->name() ,
        'description' => $experiment->description() ,

        'instr_id'    => $instrument->id() ,
        'instr_name'  => $instrument->name() ,

        'registration_time'     => $experiment->registration_time()->toStringShort() ,
        'registration_time_64'  => $experiment->registration_time()->to64() ,
        'registration_time_sec' => $experiment->registration_time()->sec ,

        'begin_time'     => $experiment->begin_time()->toStringShort() ,
        'begin_time_64'  => $experiment->begin_time()->to64() ,
        'begin_time_sec' => $experiment->begin_time()->sec ,

        'end_time'     => $experiment->end_time()->toStringShort() ,
        'end_time_64'  => $experiment->end_time()->to64() ,
        'end_time_sec' => $experiment->end_time()->sec ,

        'is_facility' => $experiment->is_facility() ? 1 : 0 ,
        'is_active'   => !$experiment->is_facility() && $SVC->regdb()->is_active_experiment($experiment->id()) ? 1 : 0 ,

        'contact_info'           => $experiment->contact_info() ,
        'contact_info_decorated' => DataPortal\DataPortal::decorated_experiment_contact_info($experiment) ,

        'leader_uid'   => $leader_uid ,
        'leader_gecos' => $leader_gecos ,
        'leader_email' => $leader_email ,

        'posix_gid'     => $experiment->POSIX_gid() ,
        'group_members' => $experiment->group_members() ,

        'num_runs'  => $num_runs ,
        'first_run' => array (

            'num' => $first_run ? $first_run->num() : '' ,

            'begin_time'     => $first_run_begin_time ? $first_run_begin_time->toStringShort() : '' ,
            'begin_time_64'  => $first_run_begin_time ? $first_run_begin_time->to64() : '' ,
            'begin_time_sec' => $first_run_begin_time ? $first_run_begin_time->sec : '' ,

            'end_time'       => $first_run_end_time ? $first_run_end_time->toStringShort() : '' ,
            'end_time_64'    => $first_run_end_time ? $first_run_end_time->to64() : '' ,
            'end_time_sec'   => $first_run_end_time ? $first_run_end_time->sec : '') ,

        'last_run' => array (

            'num' => $last_run ? $last_run->num() : '',

            'begin_time'     => $last_run_begin_time ? $last_run_begin_time->toStringShort() : '' ,
            'begin_time_64'  => $last_run_begin_time ? $last_run_begin_time->to64() : '' ,
            'begin_time_sec' => $last_run_begin_time ? $last_run_begin_time->sec : '' ,

            'end_time'       => $last_run_end_time ? $last_run_end_time->toStringShort() : '' ,
            'end_time_64'    => $last_run_end_time ? $last_run_end_time->to64() : '' ,
            'end_time_sec'   => $last_run_end_time ? $last_run_end_time->sec : '') ,

        'num_shifts' => $logbook_experiment->num_shifts() ,

        'num_elog_entries'       => $logbook_experiment->num_entries() ,
        'last_elog_entry_posted' => $last_entry ? $last_entry->insert_time()->toStringShort() : ''
    )) ;
}) ;
?>
