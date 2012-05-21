<?php

/**
 * This service will return a newely created project. If a title
 * of an existing project is provided then a clone of that project will be
 * made. Otherwise the project will be created from scratch using project
 * attributes provided to the script.
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

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $project = null;

    $project2clone = NeoCaptarUtils::get_param_GET('project2clone',false,true);
    $owner         = NeoCaptarUtils::get_param_GET('owner');
    $title         = NeoCaptarUtils::get_param_GET('title');
    $description   = NeoCaptarUtils::get_param_GET('description',true,true);
    $due_time_str  = NeoCaptarUtils::get_param_GET('due_time').' 00:00:00';
    $due_time      = null;
    $created_time  = LusiTime::now();
    try {
        $due_time = LusiTime::parse($due_time_str);
        if($created_time->greaterOrEqual($due_time))
            NeoCaptarUtils::report_error("illegal value of due_time: it's older than the present time");
    } catch( LusiTimeException $e ) {
        NeoCaptarUtils::report_error("failed to translate value of the due_time parameter: {$due_time_str}");
    }
    $project = $neocaptar->add_project( $owner, $title, $description, $created_time, $due_time, $created_time );
    if( is_null( $project )) NeoCaptarUtils::report_error("project already exists: {$title}. Please, choose a different title.");

    // If cloning cables from an existing project then find that project.
    //
    if( !is_null($project2clone) && ( $project2clone != '' )) {
        $project2clone = $neocaptar->find_project_by_title($project2clone);
        if( is_null($project2clone)) NeoCaptarUtils::report_error("no such project exists. It may have already been deleted from the database.");
        foreach( $project2clone->cables() as $cable2clone ) {
            $cable = $project->clone_cable($cable2clone);
        }
    }
    $project_as_array = NeoCaptarUtils::project2array($project);

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success(array('project' => $project_as_array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
