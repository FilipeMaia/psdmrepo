<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/**
 * This service will create a new cable either by cloning an existing one,
 * or by creating a brand new one for the specified project.
 * 
 * PARAMETERS:
 * 
 *   cid - if a cable ID is spcified then a clone will be returned
 *   pid - otherwise a project ID is required
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

$cid = $_GET['cid'];
if( !isset($cid)) {
	$pid = $_GET['pid'];
	if( !isset($pid)) report_error('missing identifier of a project for the new cable');
}

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$cable = null;
	if( isset($cid)) {
		$cid += 1000;
		$cable = array (
			'id'           => $cid,
			'status'       => 'Planned',
			'jobnum'       => '',
			'cablenum'     => '',
			'system'       => 'CXI-DG2-PIP-01',
			'func'         => 'R52 To Cxi-Dg2 Pip-01 Hv Feed (CLONE)',
			'type'         => $cid % 3 ? 'CNT195FR' : 'CAT6TLN',
			'length'       => 12 * ( $cid % 14 ),
			'routing'      => 'HV:TDFEHF02',
				'origin'   => array (
				'name'     => 'CXI-DG2-PIP-01-X',
				'loc'      => 'CXI',
				'rack'     => 'DG2',
				'ele'      => '',
				'side'     => '',
				'slot'     => 'PIP-01',
				'connum'   => 'X',
				'pinlist'  => '',
				'station'  => '',
				'conntype' => 'STARCEL',
				'instr'    => 5
			),
			'destination'  => array (
				'name'     => 'B999-5605-PCI-J501',
				'loc'      => 'B999',
				'rack'     => '56',
				'ele'      => '05',
				'side'     => 'B',
				'slot'     => 'PCI',
				'connum'   => 'J501',
				'pinlist'  => '',
				'station'  => '',
				'conntype' => 'SHV-10K',
				'instr'    => 5
			)
		);
	} else {
		$cid += $pid * 1000;
		$cable = array (
			'id'           => $cid,
			'status'       => 'Planned',
			'jobnum'       => '',
			'cablenum'     => '',
			'system'       => '',
			'func'         => '',
			'type'         => '',
			'length'       => 0,
			'routing'      => '',
				'origin'   => array (
				'name'     => '',
				'loc'      => '',
				'rack'     => '',
				'ele'      => '',
				'side'     => '',
				'slot'     => '',
				'connum'   => '',
				'pinlist'  => '',
				'station'  => '',
				'conntype' => '',
				'instr'    => 0
			),
			'destination'  => array (
				'name'     => '',
				'loc'      => '',
				'rack'     => '',
				'ele'      => '',
				'side'     => '',
				'slot'     => '',
				'connum'   => '',
				'pinlist'  => '',
				'station'  => '',
				'conntype' => '',
				'instr'    => 0
			)
		);
	}

	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "cable": '.json_encode( $cable ).
   		'}';

	$authdb->commit();
	
} catch( AuthDBException     $e ) { print $e->toHtml(); }
  catch( LusiTimeException   $e ) { print $e->toHtml(); }
  
?>
