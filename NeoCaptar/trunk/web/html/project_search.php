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
require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
    // All parameters of the search request are optional, and if they're empty
    // then they will be ignored as if they were never specified.
    //
    $required    = false;
    $allow_empty = true;

    $id = NeoCaptarUtils::get_param_GET('id',$required,$allow_empty);

    $title = NeoCaptarUtils::get_param_GET('title',$required,$allow_empty);
    if( $title == '' ) $title = null;

    $owner = NeoCaptarUtils::get_param_GET('owner',$required,$allow_empty);
    if( $owner == '' ) $owner = null;

    $job = NeoCaptarUtils::get_param_GET('job',$required,$allow_empty);
    if( $job == '' ) $job = null;

    $prefix = NeoCaptarUtils::get_param_GET('prefix',$required,$allow_empty);
    if( $prefix == '' ) $prefix = null;

    $source_has_hours = false;
    $begin = NeoCaptarUtils::get_param_GET_time('begin',$required,$allow_empty,$source_has_hours);
    $end   = NeoCaptarUtils::get_param_GET_time('end',  $required,$allow_empty,$source_has_hours);

    if(!is_null($begin) && !is_null($end) && $begin->greaterOrEqual($end))
        NeoCaptarUtils::report_error('invalid project creation interval: begin time must be strictly less than the end one');

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$project2array = array();
    if( !is_null($id)) {
        $project = $neocaptar->find_project_by_id($id);
        if(is_null($project)) NeoCaptarUtils::report_error('no project found for id: '.$id);
        array_push($project2array, NeoCaptarUtils::project2array($project));
    } else if(!is_null($job)) {
        $project = $neocaptar->find_project_by_jobnumber($job);
        if(is_null($project)) NeoCaptarUtils::report_error('no project found for job number: '.$job);
        array_push($project2array, NeoCaptarUtils::project2array($project));
    } else if(!is_null($prefix)) {
        foreach( $neocaptar->find_projects_by_jobnumber_prefix($prefix) as $p )
            array_push($project2array, NeoCaptarUtils::project2array($p));
    } else {
        foreach( $neocaptar->projects($title,$owner,$begin,$end) as $p )
            array_push($project2array, NeoCaptarUtils::project2array($p));
    }

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success(array('project' => $project2array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }

?>