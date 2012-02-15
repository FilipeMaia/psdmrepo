<?php

/**
 * This service will return a dictionary of known routings.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$routings = array();
	foreach( $neocaptar->dict_routings() as $routing ) {
		$routings[$routing->name()] = array(
			'id'           => $routing->id(),
			'created_time' => $routing->created_time()->toStringShort(),
			'created_uid'  => $routing->created_uid()
		);
	}

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('routing' => $routings));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
