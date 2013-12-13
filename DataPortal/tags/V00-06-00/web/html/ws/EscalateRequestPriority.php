<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

/* This script will escalate the priority of a pending translation request for the specified
 * identifier and, if successfull, return a JSON object describing the updated status of
 * the request. Otherwise another JSON object with an explanation of the problem will be returned.
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

	echo
'{'.
'  "Status": "error", "Reason" : ' .json_encode( $msg ).', '.
'  "status": "error", "message": '.json_encode( $msg ).
'}';
	exit;
}

/* Translate & analyze input parameters
 */
if( !isset( $_GET[ 'exper_id' ] )) report_error( 'no experiment identifier parameter found' );
$exper_id = (int)trim( $_GET[ 'exper_id' ] );
if( $exper_id <= 0 ) report_error( 'invalid experiment identifier' );

if( !isset( $_GET[ 'id' ] )) report_error( 'no request identifier parameter found' );
$id = (int)trim( $_GET[ 'id' ] );
if( $id <= 0 ) report_error( 'invalid request identifier' );

/**
 * Return a JSON document with the new priority of the request.
 *
 * @param $priority - the new priority
 */	
function return_result( $priority ) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

	echo
'{'.
'  "Status": "success", '.
'  "status": "success", '.
'  "Priority": '.json_encode( $priority ).
'}';
	exit;
}

try {
	LogBook::instance()->begin();

	/* Find the experiment & run
	 */
	$experiment = LogBook::instance()->find_experiment_by_id( $exper_id );
	if( is_null( $experiment )) report_error( 'no such experiment exists' );
	
	/* Find pending requests for the experiment in order to calculate
	 * the desired priority range for the modified request.
	 */
	$pending_requests = FileMgrIfaceCtrlWs::experiment_requests (
       	$experiment->instrument()->name(),
        $experiment->name()
    );
    
    /* Find the highest priority of the pending requests (if any).
     */
    $priority = 0;
    foreach( $pending_requests as $req ) {
    	if(( $req->status == 'Initial_Entry' ) || ( $req->status == 'Waiting_Translation' )) {
    		if( $req->priority > $priority ) {
    			$priority = $req->priority;
    		}
    	}
    }
	$priority++;

	$request = null;
	FileMgrIfaceCtrlWs::set_request_priority(
		$request,
		$id,
		$priority );

	return_result( $priority );

} catch( LogBookException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException $e ) { report_error( $e->toHtml()); }

?>
