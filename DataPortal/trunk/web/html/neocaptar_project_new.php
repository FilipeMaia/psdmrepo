<?php

/**
 * This service will return a newely created project.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $owner        = NeoCaptarUtils::get_param_GET('owner');
    $title        = NeoCaptarUtils::get_param_GET('title');
    $description  = NeoCaptarUtils::get_param_GET('description',true,true);
    $due_time_str = NeoCaptarUtils::get_param_GET('due_time').' 00:00:00';
    $due_time     = null;
    $created_time = LusiTime::now();
    try {
        $due_time = LusiTime::parse($due_time_str);
        if($created_time->greaterOrEqual($due_time))
            NeoCaptarUtils::report_error("illegal value of due_time: it's older than the present time");
    } catch( LusiTimeException $e ) {
        NeoCaptarUtils::report_error("failed to translate value of the due_time parameter: {$due_time_str}");
    }

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$project = $neocaptar->add_project( $owner, $title, $description, $created_time, $due_time, $created_time );
	if( is_null( $project )) NeoCaptarUtils::report_error("project already exists: {$title}");

    $project_as_array = NeoCaptarUtils::project2array($project);

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success(array('project' => $project_as_array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
