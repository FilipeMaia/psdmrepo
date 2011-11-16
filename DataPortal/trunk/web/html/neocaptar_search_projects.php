<?php

require_once( 'lusitime/lusitime.inc.php' );

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

	$pid = 1;
	$projects = array();
	array_push(
		$projects,
		array(
			'id'               => $pid++,
			'created_sec'      => LusiTime::parse('2011-03-23 10:44:23')->sec,
			'created'          => LusiTime::parse('2011-03-23 10:44:23')->toStringDay(),
			'title'            => 'IB Network for FEH',
			'owner'            => 'gapon',
			'status'           => array (
				'total'        => 23,
				'Planned'      => 23,
				'Registered'   => 0,
				'Labeled'      => 0,
				'Fabrication'  => 0,
				'Ready'        => 0,
				'Installed'    => 0,
				'Commissioned' => 0,
				'Damaged'      => 0,
				'Retired'      => 0
			),
			'is_submitted'     => 0,
			'submitted_sec'    => 0,
			'submitted'        => '',
			'due_sec'          => LusiTime::parse('2011-10-02 08:00:00')->sec,
			'due'              => LusiTime::parse('2011-10-02 08:00:00')->toStringDay()
		)
	);
	array_push(
		$projects,
		array(
			'id'            => $pid++,
			'created_sec'   => LusiTime::parse('2011-08-30 17:04:36')->sec,
			'created'       => LusiTime::parse('2011-08-30 17:04:36')->toStringDay(),
			'title'         => 'CXI HV',
			'owner'         => 'perazzo',
			'status'           => array (
				'total'        => 16,
				'Planned'      => 0,
				'Registered'   => 0,
				'Labeled'      => 9,
				'Fabrication'  => 0,
				'Ready'        => 1,
				'Installed'    => 0,
				'Commissioned' => 6,
				'Damaged'      => 0,
				'Retired'      => 0
			),
			'is_submitted'  => 1,
			'submitted_sec' => LusiTime::parse('2011-09-30 08:41:23')->sec,
			'submitted'     => LusiTime::parse('2011-09-30 08:41:23')->toStringDay(),
			'due_sec'       => LusiTime::parse('2011-10-02 08:00:00')->sec,
			'due'           => LusiTime::parse('2011-10-02 08:00:00')->toStringDay()
		)
	);
	array_push(
		$projects,
		array(
			'id'            => $pid++,
			'created_sec'   => LusiTime::parse('2011-10-03 18:11:35')->sec,
			'created'       => LusiTime::parse('2011-10-03 18:11:35')->toStringDay(),
			'title'         => 'Ethernet 10 Gbps LAN for NEH Data Servers',
			'owner'         => 'gapon',
			'status'           => array (
				'total'        => 78,
				'Planned'      => 78,
				'Registered'   => 0,
				'Labeled'      => 0,
				'Fabrication'  => 0,
				'Ready'        => 0,
				'Installed'    => 0,
				'Commissioned' => 0,
				'Damaged'      => 0,
				'Retired'      => 0
			),
			'is_submitted'  => 0,
			'submitted_sec' => 0,
			'submitted'     => '',
			'due_sec'       => LusiTime::parse('2011-10-02 08:00:00')->sec,
			'due'           => LusiTime::parse('2011-10-02 08:00:00')->toStringDay()
		)
	);
	array_push(
		$projects,
		array(
			'id'            => $pid++,
			'created_sec'   => LusiTime::parse('2011-01-23 08:11:35')->sec,
			'created'       => LusiTime::parse('2011-11-23 08:11:35')->toStringDay(),
			'title'         => 'IPMI cables (server room)',
			'owner'         => 'perazzo',
			'status'           => array (
				'total'        => 54,
				'Planned'      => 54,
				'Registered'   => 0,
				'Labeled'      => 0,
				'Fabrication'  => 0,
				'Ready'        => 0,
				'Installed'    => 0,
				'Commissioned' => 0,
				'Damaged'      => 0,
				'Retired'      => 0
			),
			'is_submitted'  => 0,
			'submitted_sec' => 0,
			'submitted'     => '',
			'due_sec'       => LusiTime::parse('2011-10-02 08:00:00')->sec,
			'due'           => LusiTime::parse('2011-10-02 08:00:00')->toStringDay()
		)
	);
	array_push(
		$projects,
		array(
			'id'            => $pid++,
			'created_sec'   => LusiTime::parse('2010-10-03 18:11:35')->sec,
			'created'       => LusiTime::parse('2010-10-03 18:11:35')->toStringDay(),
			'title'         => 'Ethernet 1 Gbps LAN for DAQ consoles',
			'owner'         => 'perazzo',
			'status'           => array (
				'total'        => 29,
				'Planned'      => 29,
				'Registered'   => 0,
				'Labeled'      => 0,
				'Fabrication'  => 0,
				'Ready'        => 0,
				'Installed'    => 0,
				'Commissioned' => 0,
				'Damaged'      => 0,
				'Retired'      => 0
			),
			'is_submitted'  => 1,
			'submitted_sec' => LusiTime::parse('2010-11-13 08:01:35')->sec,
			'submitted'     => LusiTime::parse('2010-11-13 08:01:35')->toStringDay(),
			'due_sec'       => LusiTime::parse('2011-10-02 08:00:00')->sec,
			'due'           => LusiTime::parse('2011-10-02 08:00:00')->toStringDay()
		)
	);

	print
   		'{ "status": '.json_encode("success").
   		', "updated": '.json_encode( LusiTime::now()->toStringShort()).
   		', "projects": '.json_encode( $projects ).
   		'}';

    
} catch( LusiTimeException $e ) { report_error( $e->toHtml()); }
  
?>