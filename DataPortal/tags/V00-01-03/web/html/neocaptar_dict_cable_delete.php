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
 * This service will delete the specified item (cable, connector or pinlist)
 * from the dictionary and return an updated portion of the dictionary
 * for the specified cable if the request was made for a connector or pinlist.
 * If the requeste was made for a cable then an empty dictionary will be returned.
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
case 'cable':
case 'connector':
case 'pinlist':	break;
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

	$cable = null;

	switch( $scope ) {

	case 'cable':
		if( !is_null( $neocaptar->find_dict_cable_by_id( $id ))) $neocaptar->delete_dict_cable_by_id( $id );
		break;

	case 'connector':
		$connector = $neocaptar->find_dict_connector_by_id( $id );
		if( !is_null( $connector )) {
			$cable = $connector->cable();
			$neocaptar->delete_dict_connector_by_id( $id );
		}
		break;

	case 'pinlist':
		$pinlist = $neocaptar->find_dict_pinlist_by_id( $id );
		if( !is_null( $pinlist )) {
			$cable = $pinlist->connector()->cable();
			$neocaptar->delete_dict_pinlist_by_id( $id );
		}
		break;
	}

	$cables = array();
	if( !is_null( $cable )) {

		$connectors = array();
		foreach( $cable->connectors() as $connector ) {

			$pinlists = array();
			foreach( $connector->pinlists() as $pinlist ) {
				$pinlists[$pinlist->name()] = array(
					'id'           => $pinlist->id(),
					'created_time' => $pinlist->created_time()->toStringShort(),
					'created_uid'  => $pinlist->created_uid()
				);
			}
			$connectors[$connector->name()] = array(
				'id'           => $connector->id(),
				'created_time' => $connector->created_time()->toStringShort(),
				'created_uid'  => $connector->created_uid(),
				'pinlist'      => $pinlists
			);
		}
		$cables[$cable->name()] = array(
			'id'           => $cable->id(),
			'created_time' => $cable->created_time()->toStringShort(),
			'created_uid'  => $cable->created_uid(),
			'connector'    => $connectors
		);
	}
	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "cable": '.json_encode( $cables ).
   		'}';

	$authdb->commit();
	$neocaptar->commit();

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { report_error( $e->toHtml()); }
  
?>
