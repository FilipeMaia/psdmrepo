<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDBException;

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request for deleting an existing free-form entry.
 */

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error_and_exit( $msg ) {

	header( "Content-type: application/json" );
	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

	$status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( $msg );
    print <<< HERE
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
HERE;
    exit;
}

if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' ) report_error_and_exit( "message id can't be empty" );
} else report_error_and_exit( "no valid message id" );

try {

    LogBook::instance()->begin();

    $entry      = LogBook::instance()->find_entry_by_id( $id ) or report_error_and_exit( "no such message" );
    $experiment = $entry->parent();
    $instrument = $experiment->instrument();

    LogBookAuth::instance()->canPostNewMessages( $experiment->id()) or
        report_error_and_exit( 'You are not authorized to delete messages of the experiment' );

    $deleted_time = LusiTime::now();
    $deleted_by   = LogBookAuth::instance()->authName();

    $experiment->delete_entry( $id, $deleted_time, $deleted_by );

    /* Return a JSON object describing extended attachments and tags back to the caller.
     */
	header( "Content-type: application/json" );
	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

	$status_encoded       = json_encode( "success" );
    $deleted_time_encoded = json_encode( $deleted_time->toStringShort());
    $deleted_by_encoded   = json_encode( $deleted_by );
    
	print <<< HERE
{
  "Status": {$status_encoded},
  "deleted_time": {$deleted_time_encoded},
  "deleted_by": {$deleted_by_encoded}
}
HERE;

    LogBook::instance()->commit();

} catch( AuthDBException   $e ) { report_error_and_exit( $e->toHtml()); }
  catch( LogBookException  $e ) { report_error_and_exit( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error_and_exit( $e->toHtml()); }

?>
