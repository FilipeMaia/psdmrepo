<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarException;

/**
 * This service will return a dictionary of known cable types,
 * connectors and pinlists.
 */

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

/* Package the error message into a JSON object and return the one back
 * to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
	$status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( $msg );
   	print <<< HERE
{
  "status": {$status_encoded},
  "message": {$msg_encoded}
}
HERE;
    exit;
}

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

	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "cable": '.json_encode( $cables ).
   		'}';

	$authdb->commit();
	$neocaptar->commit();

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { report_error( $e->toHtml()); }
  
?>
