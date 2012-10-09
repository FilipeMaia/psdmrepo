<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTimeException;

/**
 * This service will create a new cable either by cloning an existing one,
 * or by creating a brand new one for the specified project.
*/
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $cable_id = NeoCaptarUtils::get_param_POST('cable_id',false);
    if( is_null($cable_id))
        $project_id = NeoCaptarUtils::get_param_POST('project_id');

	$authdb = AuthDB::instance();
	$authdb->begin();

    $neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$cable = null;
	if( is_null($cable_id)) {
        $project = $neocaptar->find_project_by_id($project_id);
        if( is_null($project)) NeoCaptarUtils::report_error('no project exists for id: '.$project_id);
        $cable = $project->add_cable();
	} else {
        $cable2clone = $neocaptar->find_cable_by_id($cable_id);
        if( is_null($cable2clone)) NeoCaptarUtils::report_error('no cable exists for id: '.$cable_id);
        $cable = $cable2clone->clone_self();
    }
    $cable2return = NeoCaptarUtils::cable2array( $cable );

    $neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success( array( 'cable' => $cable2return ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  
?>
