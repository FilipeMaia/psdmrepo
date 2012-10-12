<?php

/**
 * This service provides services for exploring and managing project
 * security aspects, such:
 * 
 * - requesting a list of users authorized to manage the project
 * - adding co-managers
 * - removing co-managers
 * 
 * At each service invocation the service will return the most current
 * security status of the project.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'neocaptar/neocaptar.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $project_id =             $SVC->required_int ('project_id') ;
    $action     = strtolower ($SVC->required_str ('action')) ;

    // Non-empty UID is required for modifying operations
    //
    $uid = $SVC->optional_str ('uid', null) ;
    if (is_null($uid)) {
        if (($action == 'add') || ($action == 'remove'))
            $SVC->abort ('no user identifier found in the request') ;
    } else
        $uid = strtolower (trim ($uid)) ;

    $project = $SVC->neocaptar()->find_project_by_id ($project_id) ;
    if (is_null($project)) $SVC->abort ("no project found for id: {$project_id}") ;

    $comanagers = $project->comanagers () ;
    switch ($action) {
        case 'add' :
        case 'remove' :
            if ($uid == $project->owner())
                $SVC->abort ('this operation is not allowed on the project owner') ;
            if (($action == 'add'   ) && !in_array ($uid, $comanagers)) $project->add_comanager    ($uid) ;
            if (($action == 'remove') &&  in_array ($uid, $comanagers)) $project->remove_comanager ($uid) ;
            $comanagers = $project->comanagers () ;
        break;
    }
    $SVC->finish (array ('comanager' => $comanagers)) ;
}) ;

?>