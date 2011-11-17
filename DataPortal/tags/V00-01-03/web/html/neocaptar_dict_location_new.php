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

if( !isset( $_GET['location' ] )) report_error('missing location name');
$location_name = trim( $_GET['location'] );
if( $location_name == '' ) report_error('empty location name');

if( isset( $_GET['rack'] )) {
	$rack_name = trim( $_GET['rack'] );
	if( $rack_name == '' ) report_error('empty rack name');
}

try {

	$created_time = LusiTime::now();

	// Check for proper authorization andf get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();
	$created_uid = $authdb->authName();
	$authdb->commit();

	// Try finding or creating a new location witin a separate transaction to avoid 
 	// a potential collision with other MySQL users who might be attempting to do
 	// the same in parallel. Note that in case of the detcted conflict 'add_dict_location()'
 	// won't throw an esception, ity will just return null to indicate the collision.
 	// In that case we should restart the transaction and make another attempt to read
 	// the database.
 	//
	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$location = $neocaptar->find_dict_location_by_name( $location_name );
	if( is_null( $location )) {
		$location = $neocaptar->add_dict_location( $location_name, $created_time, $created_uid );
		if( is_null( $location )) {
			$neocaptar->commit();
			$neocaptar->begin();
			$location = $neocaptar->find_dict_location_by_name( $location_name );
			if( is_null( $location )) report_error('failed to find or create the specified location');
		}
	}

	// Same approach for racks
	//
	if( isset( $rack_name )) {
		$rack = $location->find_rack_by_name( $rack_name );
		if( is_null( $rack )) {
			$rack = $location->add_rack( $rack_name, $created_time, $created_uid  );
			if( is_null( $rack )) {
				$neocaptar->commit();
				$neocaptar->begin();
				$rack = $location->find_rack_by_name( $rack_name );
				if( is_null( $rack )) report_error('failed to find or create the specified rack');
			}
		}
	}

	$racks = array();
	foreach( $location->racks() as $rack ) {

		$racks[$rack->name()] = array(
			'id'           => $rack->id(),
			'created_time' => $rack->created_time()->toStringShort(),
			'created_uid'  => $rack->created_uid()
		);
	}
	$locations = array(
		$location->name() => array(
			'id'           => $location->id(),
			'created_time' => $location->created_time()->toStringShort(),
			'created_uid'  => $location->created_uid(),
			'rack'    => $racks
		)
	);
	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "location": '.json_encode( $locations ).
   		'}';

	$neocaptar->commit();

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { report_error( $e->toHtml()); }
  
?>
