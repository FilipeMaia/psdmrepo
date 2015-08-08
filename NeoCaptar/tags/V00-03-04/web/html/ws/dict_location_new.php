<?php

/**
 * This service will return a dictionary of known locations and racks.
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

    $location_name = NeoCaptarUtils::get_param_GET('location');
    $rack_name     = NeoCaptarUtils::get_param_GET('rack',false);  // not required

	$created_time = LusiTime::now();

	// Check for proper authorization and get the current UID
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
			if( is_null( $location )) NeoCaptarUtils::report_error('failed to find or create the specified location');
		}
	}

	// Same approach for racks
	//
	if( !is_null( $rack_name )) {
		$rack = $location->find_rack_by_name( $rack_name );
		if( is_null( $rack )) {
			$rack = $location->add_rack( $rack_name, $created_time, $created_uid  );
			if( is_null( $rack )) {
				$neocaptar->commit();
				$neocaptar->begin();
				$rack = $location->find_rack_by_name( $rack_name );
				if( is_null( $rack )) NeoCaptarUtils::report_error('failed to find or create the specified rack');
			}
		}
	}

	$locations = NeoCaptarUtils::dict_locations2array($neocaptar);

	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('location' => $locations));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
