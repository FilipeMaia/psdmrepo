<?php

/**
 * This service will delete the specified item (location or rack)
 * from the dictionary and return an updated portion of the dictionary
 * for the specified location if the request was made for a rack.
 * If the requeste was made for a location then an empty dictionary will be returned.
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
    $id    = NeoCaptarUtils::get_param_GET('id');
    $scope = NeoCaptarUtils::get_param_GET('scope');

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

    default:
        NeoCaptarUtils::report_error(($scope==''?'empty':'illegal').' value of the scope parameter found in the request');
	}
	$locations = NeoCaptarUtils::dict_locations2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('location' => $locations));


} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
