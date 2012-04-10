<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

/**
 * This service will delete an existing cable and return a JSON object with
 * a completion status of the operation.
*/
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $cable_id = NeoCaptarUtils::get_param_GET('cable_id');

    $authdb = AuthDB::instance();
	$authdb->begin();

    $neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$cable = $neocaptar->find_cable_by_id($cable_id);
    if( !is_null($cable)) $neocaptar->delete_cable_by_id($cable_id);

	$neocaptar->commit();
	$authdb->commit();

    NeoCaptarUtils::report_success();

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  
?>
