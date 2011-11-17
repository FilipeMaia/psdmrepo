<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

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

	$cables = array();
	for( $i = 0; $i < 55; $i++ ) {
		$status = '';
		switch( $i % 9 ) {
		case 0 : $status = 'Planned';      break;
		case 1 : $status = 'Registered';   break;
		case 2 : $status = 'Labeled';      break;
		case 3 : $status = 'Fabrication';  break;
		case 4 : $status = 'Ready';        break;
		case 5 : $status = 'Installed';    break;
		case 6 : $status = 'Commissioned'; break;
		case 7 : $status = 'Damaged';      break;
		case 8 : $status = 'Retired';      break;
		}
		array_push (
			$cables,
			array (
				'id'           => $i,
				'status'       => $status,
				'jobnum'       => '',
				'cablenum'     =>  '',
				'system'       => 'CXI-DG2-PIP-01',
				'func'         => 'R52 To Cxi-Dg2 Pip-01 Hv Feed',
				'type'         => $i % 3 ? 'CNT195FR' : 'CAT6TLN',
				'length'       => 12 * ( $i % 14 ),
				'routing'      => 'HV:TDFEHF02',
				'origin'       => array (
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
			)
		);
	}
	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "cables": '.json_encode( $cables ).
   		'}';

	$authdb->commit();
	
} catch( AuthDBException     $e ) { print $e->toHtml(); }
  catch( LusiTimeException   $e ) { print $e->toHtml(); }
  
?>
