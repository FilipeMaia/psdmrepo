<?php

/**
 * This service will return a dictionary of known locations and racks.
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

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('location' => $locations));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
