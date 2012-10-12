<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookUtils;
use LogBook\LogBookException;

/*
 * This script will process a request for changing a scope of teh specified
 * free-form entry.
 *
 * A JSON object with the result of the operation will be returned.
 * If the operation was successfull then the reply will also contain
 * an additional information about a new scope of the message.
 */
function report_error($msg) {
	return_result(
        array(
            'status' => 'error',
            'message' => $msg
        )
    );
}
function report_success($result) {
    $result['status'] = 'success';
  	return_result($result);
}
function return_result($result) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

    echo json_encode($result);
	exit;
}

if( !isset( $_GET['id'] )) report_error( "no valid message identifier found" );
$id = trim( $_GET['id'] );
if( $id == '' ) report_error( "message identifier can't be empty" );

if( !isset( $_GET['scope'] )) report_error( "no valid message scope found" );
$scope = trim( $_GET['scope'] );
if( $scope == '' ) report_error( "scope can't be empty" );

$run_num = null;
if( $scope == 'run' ) {
    if( isset( $_GET['run_num'] )) {
        $run_num = intval(trim($_GET['run_num']));
        if( !$run_num ) report_error( "run number is not valid" );
    }
} else {
    report_error( "unsupported message scope found among parameters" );
}

try {

    LogBook::instance()->begin();

    $entry = LogBook::instance()->find_entry_by_id( $id ) or report_error( "no such message entry" );

    LogBookAuth::instance()->canEditMessages( $entry->parent()->id()) or
        report_error( 'You are not authorized to modify messages for the experiment' );

    if( $scope == 'run' ) {
        if( $entry->parent_entry_id()) report_error('operation is not permitted on child messages');
        if( $entry->shift_id()) report_error('operation is not permitted on messages associated with shifts');
        if( is_null( $run_num )) {
            $entry = LogBook::instance()->dettach_entry_from_run( $entry );
        } else {
        	$run = $entry->parent()->find_run_by_num( $run_num );
            if( is_null($run)) report_error( 'no such round found in the experiemnt' );
            $entry = LogBook::instance()->attach_entry_to_run($entry, $run);
        }
    }
    $result = array(
        'run_id'  => $entry->run_id() ? $entry->run_id()     : 0,
        'run_num' => $entry->run_id() ? $entry->run()->num() : 0
    );

    LogBook::instance()->commit();

    report_success($result);

} catch( LogBookException $e ) { report_error( $e->toHtml()); }

?>
