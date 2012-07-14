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

    $project_id = strtolower( trim( NeoCaptarUtils::get_param_GET('project_id')));
    $action     = strtolower( trim( NeoCaptarUtils::get_param_GET('action')));

    // Non-empty UID is required for modifying operations
    //
    $uid = NeoCaptarUtils::get_param_GET('uid', false, false);
    if( is_null($uid)) {
        if(($action == 'add') || ($action == 'remove'))
            NeoCaptarUtils::report_error('no user identifier found in the request');
    } else
        $uid = strtolower( trim( $uid ));

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$project = $neocaptar->find_project_by_id($project_id);
    if(is_null($project)) NeoCaptarUtils::report_error("no project found for id: {$project_id}");

    $comanagers = $project->comanagers();
    switch($action) {
        case 'add':
        case 'remove':
            if( $uid == $project->owner())
                NeoCaptarUtils::report_error('this operation is not allowed on the project owner');
            if(($action == 'add'   ) && !in_array($uid, $comanagers)) $project->add_comanager   ($uid);
            if(($action == 'remove') &&  in_array($uid, $comanagers)) $project->remove_comanager($uid);
            $comanagers = $project->comanagers();
        break;
    }

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success(array('comanager' => $comanagers));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }

?>