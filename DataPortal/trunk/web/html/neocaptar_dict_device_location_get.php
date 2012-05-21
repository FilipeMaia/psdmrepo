<?php

/**
 * This service will return a dictionary of known device locations,
 * regions and components.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$locations = array();
	foreach( $neocaptar->dict_device_locations() as $location ) {

		$regions = array();
		foreach( $location->regions() as $region ) {

			$components = array();
			foreach( $region->components() as $component ) {
				$components[$component->name()] = array(
					'id'            => $component->id(),
					'created_time'  => $component->created_time()->toStringShort(),
					'created_uid'   => $component->created_uid()
				);
			}
			$regions[$region->name()] = array(
				'id'           => $region->id(),
				'created_time' => $region->created_time()->toStringShort(),
				'created_uid'  => $region->created_uid(),
				'component'    => $components
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

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
