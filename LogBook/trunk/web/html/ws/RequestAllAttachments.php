<?php

/*
 * Return all attachments of an experiment. Group atatchments by a run.
 * 
 * PARAMETERS:
 * 
 *   <exper_id>
 */

require_once 'dataportal/dataportal.inc.php' ;

function sort_by_time_and_merge ($runs, $attachments) {
    $result = array () ;
    $title4run = "show the run in the e-Log Search panel within the current Portal" ;
    foreach ($runs as $r) {
        array_push (
            $result ,
            array (
                'time64'  => $r->begin_time()->to64() ,
                'type'    => 'r' ,
                'r_id'    => $r->id() ,
                'r_num'   => $r->num() ,
                'r_begin' => $r->begin_time()->toStringShort() ,
                'r_end'   => is_null($r->end_time()) ? '' : $r->end_time()->toStringShort())) ;
    }
    $title = "show the message in the e-Log Search panel within the current Portal" ;
    foreach ($attachments as $a) {
        array_push (
            $result ,
            array (
                'time64'    => $a->parent()->insert_time()->to64() ,
                'type'      => 'a' ,
                'e_id'      => $a->parent()->id() ,
                'e_time'    => $a->parent()->insert_time()->toStringShort() ,
                'e_time_64' => $a->parent()->insert_time()->to64() ,
                'e_author'  => $a->parent()->author() ,
                'a_id'      => $a->id() ,
                'a_name'    => $a->description() ,
                'a_size'    => $a->document_size() ,
                'a_type'    => $a->document_type() ,
                'entry_id'  => $a->parent()->id())) ;
    }
    usort (
        $result ,
        function ($a, $b) {
            return $a['time64'] - $b['time64'] ; }) ;

    return $result ;
}

function find_and_process_entries (&$attachments, $entries) {
    foreach ($entries as $e) {
        foreach ($e->attachments() as $a) array_push($attachments, $a) ;
        find_and_process_entries ($attachments, $e->children()) ;
    }
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->required_int('exper_id') ;

    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($exper_id) ,
                                     "no experiment found for id={$exper_id}") ;

    $SVC->assert ($SVC->logbookauth()->canRead($experiment->id()) ,
                  "not authorized to access experiment id={$experiment->id()}") ;

    $attachments = array () ;
    find_and_process_entries($attachments, $experiment->entries()) ;

    return array (
        'Attachments' => sort_by_time_and_merge($experiment->runs(), $attachments)) ;
}) ;

?>
