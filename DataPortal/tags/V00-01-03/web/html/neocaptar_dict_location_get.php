<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarException;

/**
 * This service will return a dictionary of known locations and racks.
 */

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

/* Package the error message into a JSON object and return the one back
 * to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
	$status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( $msg );
   	print <<< HERE
{
  "status": {$status_encoded},
  "message": {$msg_encoded}
}
HERE;
    exit;
}

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$locations = array();
	foreach( $neocaptar->dict_locations() as $location ) {

		$racks = array();
		foreach( $location->racks() as $rack ) {

			$racks[$rack->name()] = array(
				'id'           => $rack->id(),
				'created_time' => $rack->created_time()->toStringShort(),
				'created_uid'  => $rack->created_uid()
			);
		}
		$locations[$location->name()] = array(
			'id'           => $location->id(),
			'created_time' => $location->created_time()->toStringShort(),
			'created_uid'  => $location->created_uid(),
			'rack'         => $racks
		);
	}

	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "location": '.json_encode( $locations ).
   		'}';

	$authdb->commit();
	$neocaptar->commit();

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { report_error( $e->toHtml()); }
  
?>
