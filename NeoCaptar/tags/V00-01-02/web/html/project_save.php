<?php

/**
 * This service will update project attributes return an project.
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

    $id           = NeoCaptarUtils::get_param_POST('id');
    $owner        = NeoCaptarUtils::get_param_POST('owner');
    $title        = NeoCaptarUtils::get_param_POST('title');
    $description  = NeoCaptarUtils::get_param_POST('description',true,true);
    $due_time_str = NeoCaptarUtils::get_param_POST('due_time').' 00:00:00';
    $due_time     = null;
    try {
        $due_time = LusiTime::parse($due_time_str);
        if(LusiTime::now()->greaterOrEqual($due_time))
            NeoCaptarUtils::report_error("illegal value of due_time: it's older than the present time");
    } catch( LusiTimeException $e ) {
        NeoCaptarUtils::report_error("failed to translate value of the due_time parameter: {$due_time_str}");
    }

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $project = $neocaptar->find_project_by_id($id);
    if(is_null($project))
        NeoCaptarUtils::report_error("no such project found in the database");

    $project = $neocaptar->find_project_by_title($title);
    if(!is_null($project) && ($project->id() != $id ))
        NeoCaptarUtils::report_error("another project owned by user '{$project->owner()}' already has this title");

    $project = $neocaptar->update_project($id, $owner, $title, $description, $due_time);

    $project_as_array = NeoCaptarUtils::project2array($project);

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success(array('project' => $project_as_array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
