<?php

/**
 * This service will delete the specified item (device location, region or region)
 * from the dictionary and return an updated portion of the dictionary
 * for the specified location if the request was made for a region or region.
 * If the requeste was made for a location then an empty dictionary will be returned.
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

    $scope = NeoCaptarUtils::get_param_GET('scope');
    $id    = NeoCaptarUtils::get_param_GET('id');

	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$location = null;

	switch( $scope ) {

	case 'location':
		if( !is_null( $neocaptar->find_dict_device_location_by_id( $id )))
            $neocaptar->delete_dict_device_location_by_id( $id );
		break;

	case 'region':
		$region = $neocaptar->find_dict_device_region_by_id( $id );
		if( !is_null( $region )) {
			$location = $region->location();
			$neocaptar->delete_dict_device_region_by_id( $id );
		}
		break;

	case 'component':
		$component = $neocaptar->find_dict_device_component_by_id( $id );
		if( !is_null( $component )) {
			$location = $component->region()->location();
			$neocaptar->delete_dict_device_component_by_id( $id );
		}
		break;

    default:
        NeoCaptarUtils::report_error(($scope==''?'empty':'illegal').' value of the scope parameter found in the request');
	}

	$locations = array();
	if( !is_null( $location )) {

		$regions = array();
		foreach( $location->regions() as $region ) {

			$components = array();
			foreach( $region->components() as $component ) {
				$components[$component->name()] = array(
					'id'           => $component->id(),
					'created_time' => $component->created_time()->toStringShort(),
					'created_uid'  => $component->created_uid()
				);
			}
			$regions[$region->name()] = array(
				'id'           => $region->id(),
				'created_time' => $region->created_time()->toStringShort(),
				'created_uid'  => $region->created_uid(),
				'component'      => $components
			);
		}
		$locations[$location->name()] = array(
			'id'           => $location->id(),
			'created_time' => $location->created_time()->toStringShort(),
			'created_uid'  => $location->created_uid(),
			'region'       => $regions
		);
	}

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'location' => $locations ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
