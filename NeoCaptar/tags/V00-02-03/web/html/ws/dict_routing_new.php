<?php

/**
 * This service will return a dictionary of known routings.
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

    $name = NeoCaptarUtils::get_param_GET('name');

	$created_time = LusiTime::now();

	// Check for proper authorization andf get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();
	$created_uid = $authdb->authName();
	$authdb->commit();

	// Try finding or creating a new routing witin a separate transaction to avoid 
 	// a potential collision with other MySQL users who might be attempting to do
 	// the same in parallel. Note that in case of the detcted conflict 'add_dict_routing()'
 	// won't throw an esception, ity will just return null to indicate the collision.
 	// In that case we should restart the transaction and make another attempt to read
 	// the database.
 	//
	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$routing = $neocaptar->find_dict_routing_by_name( $name );
	if( is_null( $routing )) {
		$routing = $neocaptar->add_dict_routing( $name, $created_time, $created_uid );
		if( is_null( $routing )) {
			$neocaptar->commit();
			$neocaptar->begin();
			$routing = $neocaptar->find_dict_routing_by_name( $name );
			if( is_null( $routing )) NeoCaptarUtils::report_error('failed to find or create the specified routing');
		}
	}

	$routings = NeoCaptarUtils::dict_routings2array($neocaptar);

	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('routing' => $routings));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
