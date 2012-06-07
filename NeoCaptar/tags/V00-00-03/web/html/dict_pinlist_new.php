<?php

/**
 * This service will create a new pinlist and return a dictionary of known pinlists.
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

    $name          = NeoCaptarUtils::get_param_GET('name');
    $documentation = NeoCaptarUtils::get_param_GET('documentation',true,true);

	$created_time = LusiTime::now();

	// Check for proper authorization andf get the current UID
	//
	$authdb = AuthDB::instance();
	$authdb->begin();
	$created_uid = $authdb->authName();
	$authdb->commit();

	// Try finding or creating a new pinlist witin a separate transaction to avoid 
 	// a potential collision with other MySQL users who might be attempting to do
 	// the same in parallel. Note that in case of the detcted conflict 'add_dict_pinlist()'
 	// won't throw an esception, ity will just return null to indicate the collision.
 	// In that case we should restart the transaction and make another attempt to read
 	// the database.
 	//
	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $pinlist = $neocaptar->find_dict_pinlist_by_name( $name );
	if( is_null( $pinlist )) {
		$pinlist = $neocaptar->add_dict_pinlist( $name, $documentation, $created_time, $created_uid  );
		if( is_null( $pinlist )) {
			$neocaptar->commit();
			$neocaptar->begin();
			$pinlist = $neocaptar->find_dict_pinlist_by_name( $name );
			if( is_null( $pinlist )) NeoCaptarUtils::report_error('failed to find or create the specified pinlist type');
		}
	}

    $pinlists = NeoCaptarUtils::dict_pinlists2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'pinlist' => $pinlists ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
