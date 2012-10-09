<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;
/*
 * This script will process a request for setting/updating a value
 * of the 'DATA_PATH' parameter for an experiment.
 */
/*
 * NOTE: Can not return JSON MIME type because of the following issue:
 *       http://jquery.malsup.com/form/#file-upload
 *
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
*/

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error_and_exit( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( $msg );
    print <<< HERE
<textarea>
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
</textarea>
HERE;
    exit;
}

if( isset( $_POST['exper_id'] )) {
    $exper_id = trim( $_POST['exper_id'] );
    if( $exper_id == '' ) report_error_and_exit( "experiment identifier can't be empty" );
} else report_error_and_exit( "no valid experiment identifier" );

if( isset( $_POST['data_path'] )) $data_path = trim( $_POST['data_path'] );
else report_error_and_exit( "no valid data path" );

try {
	$regdb = RegDB::instance();
	$regdb->begin();

	if( !RegDBAuth::instance()->canEdit())
		report_error_and_exit( 'not autorized to modify Data Path for the experiment' );

	$experiment = $regdb->find_experiment_by_id( $exper_id );
	if( !$experiment ) report_error_and_exit( 'no such experiment' );

	$param = $experiment->set_param( 'DATA_PATH', $data_path );
	if( is_null( $param )) report_error_and_exit( 'failed to modify the parameter' );

	$regdb->commit();

    $status_encoded = json_encode( "success" );
    print <<< HERE
<textarea>
{
  "Status": {$status_encoded}
}
</textarea>
HERE;

} catch( RegDBException $e ) { report_error_and_exit( $e->toHtml()); }

?>