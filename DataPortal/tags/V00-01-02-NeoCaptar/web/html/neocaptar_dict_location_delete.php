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
 * This service will delete the specified item (location or rack)
 * from the dictionary and return an updated portion of the dictionary
 * for the specified location if the request was made for a rack.
 * If the requeste was made for a location then an empty dictionary will be returned.
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

$scope = $_GET['scope'];
if( !isset( $scope )) report_error('no scope parameter found in the request');
$scope = trim( $scope );

switch( $scope ) {
case 'location':
case 'rack': break;
default: report_error(($scope==''?'empty':'illegal').' value of the scope parameter found in the request');
}

$id = $_GET['id'];
if( !isset( $id )) report_error('no element identifier found in the request');
$id = trim( $id );
if( $id == '' ) report_error('empty element identifier found in the request');

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$location = null;

	switch( $scope ) {

	case 'location':
		if( !is_null( $neocaptar->find_dict_location_by_id( $id ))) $neocaptar->delete_dict_location_by_id( $id );
		break;

	case 'rack':
		$rack = $neocaptar->find_dict_rack_by_id( $id );
		if( !is_null( $rack )) {
			$location = $rack->location();
			$neocaptar->delete_dict_rack_by_id( $id );
		}
		break;
	}

	$locations = array();
	if( !is_null( $location )) {

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
