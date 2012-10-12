<?php

/**
 * This service will search for projects and return a list of those.
 * There are three alternative modes of operation of the script:
 * 
 * 1. if an optional project 'id' is specified then the script will make\
 * an attempt to find that project.
 * 
 * 2. if any combination of the optional search criterias will be found and
 * if at least one of those will be non-empty then the result of the search
 * will be narrowed according to the requested criteria(s).
 * 
 * 3. in all other curcumstances a list of all known projects will be returned.
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'neocaptar/neocaptar.inc.php' ;

use \NeoCaptar\NeoCaptarUtils ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    // All parameters of the search request are optional, and if they're empty
    // then they will be ignored as if they were never specified.
    //
    $id      = $SVC->optional_int  ('id',      null) ;
    $title   = $SVC->optional_str  ('title' ,  null) ;
    $owner   = $SVC->optional_str  ('owner',   null) ;
    $coowner = $SVC->optional_str  ('coowner', null) ;
    $job     = $SVC->optional_str  ('job',     null) ;
    $prefix  = $SVC->optional_str  ('prefix',  null) ;
    $begin   = $SVC->optional_time ('begin',   null) ;
    $end     = $SVC->optional_time ('end',     null) ;

    if (!is_null($begin) && !is_null ($end) && $begin->greaterOrEqual ($end))
        $SVC->abort ('invalid project creation interval: begin time must be strictly less than the end one');

    $projects = array () ;
    if (!is_null($id)) {

        $project = $SVC->neocaptar()->find_project_by_id ($id) ;
        if (is_null($project)) $SVC->abort ('no project found for id: '.$id) ;

        array_push ($projects, NeoCaptarUtils::project2array ($project)) ;

    } else if (!is_null($job)) {

        $project = $SVC->neocaptar()->find_project_by_jobnumber ($job) ;
        if (is_null($project)) $SVC->abort ('no project found for job number: '.$job) ;

        array_push ($projects, NeoCaptarUtils::project2array ($project)) ;

    } else if (!is_null($prefix)) {

        foreach ($SVC->neocaptar()->find_projects_by_jobnumber_prefix ($prefix) as $p)
            array_push ($projects, NeoCaptarUtils::project2array ($p)) ;

    } else if (!is_null($coowner)) {

        foreach ($SVC->neocaptar()->projects_by_coowner ($coowner) as $p)
            array_push ($projects, NeoCaptarUtils::project2array ($p)) ;

    } else {

        foreach ($SVC->neocaptar()->projects ($title, $owner, $begin, $end) as $p)
            array_push($projects, NeoCaptarUtils::project2array ($p)) ;
    }

    $SVC->finish (array ('project' => $projects)) ;
}) ;
?>