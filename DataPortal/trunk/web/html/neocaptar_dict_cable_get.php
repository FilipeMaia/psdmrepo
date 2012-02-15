<?php

/**
 * This service will return a dictionary of known cable types,
 * connectors and pinlists.
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

	$cables = array();
	foreach( $neocaptar->dict_cables() as $cable ) {

		$connectors = array();
		foreach( $cable->connectors() as $connector ) {

			$pinlists = array();
			foreach( $connector->pinlists() as $pinlist ) {
				$pinlists[$pinlist->name()] = array(
					'id'           => $pinlist->id(),
					'created_time' => $pinlist->created_time()->toStringShort(),
					'created_uid'  => $pinlist->created_uid()
				);
			}
			$connectors[$connector->name()] = array(
				'id'           => $connector->id(),
				'created_time' => $connector->created_time()->toStringShort(),
				'created_uid'  => $connector->created_uid(),
				'pinlist'      => $pinlists
			);
		}
		$cables[$cable->name()] = array(
			'id'           => $cable->id(),
			'created_time' => $cable->created_time()->toStringShort(),
			'created_uid'  => $cable->created_uid(),
			'connector'    => $connectors
		);
	}

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'cable' => $cables ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
