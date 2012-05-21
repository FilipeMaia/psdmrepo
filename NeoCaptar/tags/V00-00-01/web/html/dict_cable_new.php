<?php

/**
 * This service will return a dictionary of known cable types,
 * connectors and pinlists.
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

    $cable_name            = NeoCaptarUtils::get_param_GET('cable');
    $connector_name        = NeoCaptarUtils::get_param_GET('connector',false);
    $pinlist_name          = is_null($connector_name) ? null : NeoCaptarUtils::get_param_GET('pinlist',false);
    $pinlist_documentation = is_null($pinlist_name) ? null : NeoCaptarUtils::get_param_GET('pinlist_documentation',true,true);

	$created_time = LusiTime::now();

	// Check for proper authorization andf get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();
	$created_uid = $authdb->authName();
	$authdb->commit();

	// Try finding or creating a new cable witin a separate transaction to avoid 
 	// a potential collision with other MySQL users who might be attempting to do
 	// the same in parallel. Note that in case of the detcted conflict 'add_dict_cable()'
 	// won't throw an esception, ity will just return null to indicate the collision.
 	// In that case we should restart the transaction and make another attempt to read
 	// the database.
 	//
	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$cable = $neocaptar->find_dict_cable_by_name( $cable_name );
	if( is_null( $cable )) {
		$cable = $neocaptar->add_dict_cable( $cable_name, $created_time, $created_uid );
		if( is_null( $cable )) {
			$neocaptar->commit();
			$neocaptar->begin();
			$cable = $neocaptar->find_dict_cable_by_name( $cable_name );
			if( is_null( $cable )) NeoCaptarUtils::report_error('failed to find or create the specified cable type');
		}
	}

	// Same approach for connectors and pinlists
	//
	if( !is_null( $connector_name )) {
		$connector = $cable->find_connector_by_name( $connector_name );
		if( is_null( $connector )) {
			$connector = $cable->add_connector( $connector_name, $created_time, $created_uid  );
			if( is_null( $connector )) {
				$neocaptar->commit();
				$neocaptar->begin();
				$connector = $cable->find_connector_by_name( $connector_name );
				if( is_null( $connector )) NeoCaptarUtils::report_error('failed to find or create the specified connector type');
			}
		}

		if( !is_null( $pinlist_name )) {
			$pinlist = $connector->find_pinlist_by_name( $pinlist_name );
			if( is_null( $pinlist )) {
				$pinlist = $connector->add_pinlist( $pinlist_name, $pinlist_documentation, $created_time, $created_uid  );
				if( is_null( $pinlist )) {
					$neocaptar->commit();
					$neocaptar->begin();
					$pinlist = $connector->find_pinlist_by_name( $pinlist_name );
					if( is_null( $pinlist )) NeoCaptarUtils::report_error('failed to find or create the specified pinlist type');
				}
			}
		}
	}

	$connectors = array();
	foreach( $cable->connectors() as $connector ) {

		$pinlists = array();
		foreach( $connector->pinlists() as $pinlist ) {
			$pinlists[$pinlist->name()] = array(
				'id'            => $pinlist->id(),
                'documentation' => $pinlist->documentation(),
				'created_time'  => $pinlist->created_time()->toStringShort(),
				'created_uid'   => $pinlist->created_uid()
			);
		}
		$connectors[$connector->name()] = array(
			'id'           => $connector->id(),
			'created_time' => $connector->created_time()->toStringShort(),
			'created_uid'  => $connector->created_uid(),
			'pinlist'      => $pinlists
		);
	}
	$cables = array(
		$cable->name() => array(
			'id'           => $cable->id(),
			'created_time' => $cable->created_time()->toStringShort(),
			'created_uid'  => $cable->created_uid(),
			'connector'    => $connectors
		)
	);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'cable' => $cables ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
