<?php

require_once( 'filemgr/filemgr.inc.php' );

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

/* This script will delete a pending translation request for the specified
 * identifier and, if successfull, return a JSON object describing the deleted
 * request. Otherwise another JSON object with an explanation of the problem will be returned.
 */

/**
 * This function is used to report errors back to the script caller applications
 *
 * @param $msg - a message to be reported
 */
function report_error( $msg ) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

	echo '{"Status":"error","Message":'.json_encode( $msg ).'}';
	exit;
}

/* Translate & analyze input parameters
 */
if( !isset( $_GET[ 'id' ] )) report_error( 'no request identifier parameter found' );

$id = (int)trim( $_GET[ 'id' ] );
if( $id <= 0 ) report_error( 'invalid request identifier' );

/**
 * Produce a document with JSON representation of successfully
 * deleted requests.
 *
 * @param $result - an array of requests
 */	
function return_result( $request ) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

	echo '{"Status":"success","Result":'.json_encode( $request ).'}';
	exit;
}

try {
	$request = null;
	FileMgrIfaceCtrlWs::delete_request(
		$request,
		$id );

	return_result( $request );

} catch( FileMgrException $e ) { report_error( $e->toHtml()); }

?>
