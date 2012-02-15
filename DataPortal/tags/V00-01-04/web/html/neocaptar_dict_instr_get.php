<?php

/**
 * This service will return a dictionary of known instrs.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

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

	$instrs = array();
	foreach( $neocaptar->dict_instrs() as $instr ) {
		$instrs[$instr->name()] = array(
			'id'           => $instr->id(),
			'created_time' => $instr->created_time()->toStringShort(),
			'created_uid'  => $instr->created_uid()
		);
	}

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'instr' => $instrs ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
