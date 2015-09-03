<?php

/**
 * Search and return all known history events for the specified (by its identifier)
 * project.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

try {
    $id = NeoCaptarUtils::get_param_GET('id');

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$project = $neocaptar->find_project_by_id( $id );
	if( is_null( $project )) NeoCaptarUtils::report_error("project not found for id: {$id}");

    $events2return = array();
	foreach( $project->history() as $e )
        array_push( $events2return, NeoCaptarUtils::event2array($e));

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success( array( 'event' => $events2return ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
