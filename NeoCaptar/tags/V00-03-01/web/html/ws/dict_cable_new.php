<?php

/**
 * This service will create a new cable type and return a dictionary of known
 * cable and connector types.
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

    $cable_name          = NeoCaptarUtils::get_param_GET('cable_name');
    $cable_documentation = NeoCaptarUtils::get_param_GET('cable_documentation',true,true);
    $connector_name      = NeoCaptarUtils::get_param_GET('connector_name',false,false);
	$created_time        = LusiTime::now();

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
		$cable = $neocaptar->add_dict_cable( $cable_name, $cable_documentation, $created_time, $created_uid );
		if( is_null( $cable )) {
			$neocaptar->commit();
			$neocaptar->begin();
			$cable = $neocaptar->find_dict_cable_by_name( $cable_name );
			if( is_null( $cable )) NeoCaptarUtils::report_error('failed to find or create the specified cable type');
		}
	}
    if( !is_null($connector_name)) {
        $connector = $neocaptar->find_dict_connector_by_name( $connector_name );
        if( is_null($connector)) {
            $connector = $neocaptar->find_dict_connector_by_name( $connector_name );
            if( is_null( $connector )) {
                $connector = $neocaptar->add_dict_connector( $connector_name, '', $created_time, $created_uid  );
                if( is_null( $connector )) {
                    $neocaptar->commit();
                    $neocaptar->begin();
                    $connector = $neocaptar->find_dict_connector_by_name( $connector_name );
                    if( is_null( $connector )) NeoCaptarUtils::report_error('failed to find or create the specified connector type');
                }
            }
        }
        if( !$cable->is_linked($connector->id()))
            $cable->link($connector->id());
    }

	$types = NeoCaptarUtils::dict_types2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'type' => $types ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
