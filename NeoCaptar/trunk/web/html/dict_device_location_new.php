<?php

/**
 * This service will create a new device location entry in the dictionary and
 * it will return a dictionary of known device locations, regions and components.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $location_name  = NeoCaptarUtils::get_param_GET('location');
    $region_name    = NeoCaptarUtils::get_param_GET('region',false);
    $component_name = is_null($region_name) ? null : NeoCaptarUtils::get_param_GET('component',false);

	$created_time = LusiTime::now();

	// Check for proper authorization andf get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();
	$created_uid = $authdb->authName();
	$authdb->commit();

	// Try finding or creating a new device location witin a separate transaction to avoid 
 	// a potential collision with other MySQL users who might be attempting to do
 	// the same in parallel. Note that in case of the detected conflict 'add_dict_device_location()'
 	// won't throw an esception, it will just return null to indicate the collision.
 	// In that case we should restart the transaction and make another attempt to read
 	// the database.
 	//
	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$location = $neocaptar->find_dict_device_location_by_name( $location_name );
	if( is_null( $location )) {
		$location = $neocaptar->add_dict_device_location( $location_name, $created_time, $created_uid );
		if( is_null( $location )) {
			$neocaptar->commit();
			$neocaptar->begin();
			$location = $neocaptar->find_dict_device_location_by_name( $location_name );
			if( is_null( $location )) NeoCaptarUtils::report_error('failed to find or create the specified device location');
		}
	}

	// Same approach for regions and components
	//
	if( !is_null( $region_name )) {
		$region = $location->find_region_by_name( $region_name );
		if( is_null( $region )) {
			$region = $location->add_region( $region_name, $created_time, $created_uid  );
			if( is_null( $region )) {
				$neocaptar->commit();
				$neocaptar->begin();
				$region = $location->find_region_by_name( $region_name );
				if( is_null( $region )) NeoCaptarUtils::report_error('failed to find or create the specified device region');
			}
		}

		if( !is_null( $component_name )) {
			$component = $region->find_component_by_name( $component_name );
			if( is_null( $component )) {
				$component = $region->add_component( $component_name, $created_time, $created_uid  );
				if( is_null( $component )) {
					$neocaptar->commit();
					$neocaptar->begin();
					$component = $region->find_component_by_name( $component_name );
					if( is_null( $component )) NeoCaptarUtils::report_error('failed to find or create the specified device component');
				}
			}
		}
	}

	$locations = NeoCaptarUtils::dict_devices2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'location' => $locations ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }

?>
