<?php

/**
 * Return the information about release(s).
 * 
 * NOTE: The service won't return "previous" releases for the returned
 *       releases.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'websrt/websrt.inc.php' ;

use websrt\WebSrt ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $release_names = $SVC->required_JSON('names') ;

    // Make sure releases exist and they all have the same type.
    // Then find them all and sort in the descending order.

    $type = null ;
    $releases = array() ;

    foreach ($release_names as $name) {
        $release = WebSrt::find_release($name) ;
        if (is_null($release))
            $SVC->abort("release not found: '{$name}'") ;

        if (is_null($type)) $type = $release->type() ;
        elseif ($type !== $release->type())
            $SVC->abort("inconsistent release types '{$type}', '{$release->type()}' in the request") ;

        array_push($releases, $release) ;
    }
    $releases = WebSrt::sort_desc($releases) ;

    // Start preparing the data structures to be returned

    $result_releases = array() ;
    $result_notes = array() ;

    $num = count($releases) ;
    switch ($num) {

        case 0:     // -- no releases - no problems

            break ;

        case 1:     // -- another special case for a single release

            $release = $releases[0] ;
            array_push (
                $result_releases ,
                $release->export2array()) ;

            array_push (
                $result_notes ,
                array (
                    'release' => $release->name() ,
                    'notes'   => $release->notes())) ;

            break ;

        default:     // -- a range of releases

            foreach ($releases as $release)
                array_push (
                    $result_releases ,
                    $release->export2array()) ;

            // Gather notes for the inclusive range of requested releases

            $begin_ver = $releases[$num-1]->version() ;
            $end_ver = $releases[0]->version() ;

            $releases4type = WebSrt::releases($type) ;
            foreach ($releases4type[$type] as $release) {

                $ver = $release->version() ;
                if ($ver->less($begin_ver)) continue ;  // -- before the begining of the range
                if ($end_ver->less($ver)) continue ;    // -- past the last release of the range

                array_push (
                    $result_notes ,
                    array (
                        'release' => $release->name() ,
                        'notes'   => $release->notes())) ;
            }
            break;
    }
    $SVC->finish (array (
        'releases' => $result_releases ,    // -- full information for the found releases
        'notes'    => $result_notes         // -- notes for the includive range of the releases
    )) ;
}) ;

?>