<?php

/*
 * Return JSON objects with a list of experiments matching the specified
 * access mode.
 */

require_once 'dataportal/dataportal.inc.php' ;

function access_granted_for_experiment ($SVC, $id, $access) {
    switch ($access) {
        case 'read'          : return $SVC->logbookauth()->canRead           ($id) ;
        case 'post'          : return $SVC->logbookauth()->canPostNewMessages($id) ;
        case 'edit'          : return $SVC->logbookauth()->canEditMessages   ($id) ;
        case 'delete'        : return $SVC->logbookauth()->canDeleteMessages ($id) ;
        case 'manage_shifts' : return $SVC->logbookauth()->canManageShifts   ($id) ;
    }
    $SVC->abort("wrong access mode '{$access}'requested") ;
}

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name  = $SVC->optional_str ('instr', null) ;
    $is_location = $SVC->optional_flag('is_location', false) ;
    $access      = $SVC->optional_str ('access', 'read') ;

    $all_experiments = $instr_name ?
        $SVC->regdb()->experiments_for_instrument($instr_name) :
        $SVC->regdb()->experiments() ;

    // Leave only those experiments the logged user is authorizated to see

    $experiments2array = array() ;
    foreach ($all_experiments as $e)
        if (access_granted_for_experiment($SVC, $e->id(), $access)) {
            if ($is_location xor $e->is_facility()) continue ;
            array_push($experiments2array , $e->is_facility() ?
                array (
                    "location"           => $e->instrument()->name() ,
                    "name"               => $e->name() ,
                    "id"                 => $e->id() ,
                    "registration_time"  => $e->registration_time()->toStringShort() ,
                    "description"        => $e->description()) :
                array (
                    "instrument"        => $e->instrument()->name(),
                    "name"              => $e->name(),
                    "id"                => $e->id(),
                    "begin_time"        => $e->begin_time()->toStringShort(),
                    "end_time"          => $e->end_time()->toStringShort(),
                    "registration_time" => $e->registration_time()->toStringShort(),
                    "description"       => $e->description())) ; 
            }

    return array (
        "ResultSet" => array (
            "Result" => $experiments2array)) ;

}) ;

?>
