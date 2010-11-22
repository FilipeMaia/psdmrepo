<?php

require_once( 'LogBook/LogBook.inc.php' );
require_once( 'FileMgr/FileMgr.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

/* This script will post a new translation request for the specified
 * instrument/experiment/run and, if successfull, return a JSON object
 * describing the request. Otherwise another JSON object with an explanation
 * of the problem will be returned.
 */

/**
 * This function is used to report errors back to the script caller applications
 *
 * @param $msg - a message to be reported
 */
function return_error( $msg ) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

	echo '{"ResultSet":{"Status":"error","Reason":"'.json_encode( $msg ).'"}}';

	exit;
}

/* Translate & analyze input parameters
 */
if( !isset( $_GET[ 'exper_id' ] ))
	return_error( 'no experiment identifier parameter found' );

$exper_id = (int)trim( $_GET[ 'exper_id' ] );
if( $exper_id <= 0 )
	return_error( 'invalid experiment identifier' );

if( !isset( $_GET[ 'runnum' ] ))
	return_error( 'no run number parameter found' );

$runnum = (int)trim( $_GET[ 'runnum' ] );
if( $runnum <= 0 )
	return_error( 'invalid run number' );

/**
 * Produce a document with JSON representation of successfully
 * submitted requests.
 *
 * @param $result - an array of requests
 */	
function return_result( $requests ) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

	echo '{"ResultSet":{"Status":"success","Result":'.json_encode( $requests ).'}}';

	exit;
}

try {

	$logbook = new LogBook();
	$logbook->begin();

	/* Find the experiment & run
	 */
	$experiment = $logbook->find_experiment_by_id( $exper_id );
	if( is_null( $experiment ))
		return_error( 'no such experiment exists' );

	$run = $experiment->find_run_by_num( $runnum );
	if( is_null( $run ))
		return_error( 'no such run exists' );

	$requests = null;
	FileMgrIfaceCtrlWs::create_request(
		$requests,
		$experiment->instrument()->name(),
		$experiment->name(),
		$runnum );

	return_result( $requests );

} catch( LogBookException $e ) {
	return_error( $e->toHtml());
} catch( FileMgrException $e ) {
	return_error( $e->toHtml());
}
?>
