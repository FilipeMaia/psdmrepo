<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    print <<< HERE
{
  "Status": {$status_encoded}, "Message": {$msg_encoded} ,
  "status": {$status_encoded}, "message": {$msg_encoded}
}
HERE;
    exit;
}

/*
 * This script will process a request for tags and authors used in a context of
 * the specified experiment or an instrument.
 */
if( !isset( $_GET['id'] )) report_error( "no experiemnt identifier found among script parameters" );
$exper_id = trim( $_GET['id'] );
if( $exper_id == '' )  report_error( "experiment identifier can't be empty" );

$accross_instrument = isset( $_GET['accross_instrument'] );

/*
 * Return JSON objects with two lists: one for tags and the other one for authors.
 */
try {

    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $exper_id ) or report_error( "no such experiment" );
    if( !LogBookAuth::instance()->canRead( $experiment->id()))
    	report_error( 'No authorization to access any information about the experiment' );

    /* If 'accross_instrument' mode was requested then harvest tags and author
     * names accross all experiments (of the current instrument) the user is
     * authorized for
     */
    $all_tags = array();
    $all_authors = array();

    $experiments = $accross_instrument ? LogBook::instance()->experiments_for_instrument( $experiment->instrument()->name()) : array( $experiment );
    foreach( $experiments as $e ) {

    	if( !LogBookAuth::instance()->canRead( $e->id())) {

    		/* Silently skip this experiemnt if browsing accross the whole instrument.
    		 * The only exception would be the main experient from which we started
    		 * things.
    		 */
    		if( $accross_instrument && ( $e->id() != $exper_id )) continue;

	        report_error( 'not authorized to read messages for the experiment' );
    	}
    	
    	/* Get names of all tags and authors which have ever been used in free-form
     	 * entries of the experiment (including its shifts and runs).
     	 */
    	foreach( $e->used_tags   () as $tag    ) $all_tags   [$tag   ] = true;
    	foreach( $e->used_authors() as $author ) $all_authors[$author] = true;
    }
    $tags = array_keys( $all_tags );
    sort( $tags );

    $authors = array_keys( $all_authors );
    sort( $authors );
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $success_encoded = json_encode("success");
    print <<< HERE
{ "Status": {$success_encoded},
  "status": {$success_encoded},
  "Tags": [
HERE;
    $first = true;
    foreach( $tags as $t ) {
    	if($first) {
    		$first = false;
    		echo json_encode( $t );
    	} else {
    		echo ','.json_encode( $t );
    	}
	}
    print <<< HERE
  ],
  "Authors": [
HERE;
    $first = true;
    foreach( $authors as $a ) {
    	if($first) {
    		$first = false;
    		echo json_encode( $a );
    	} else {
    		echo ','.json_encode( $a );
    	}
    }
    echo " ] }";

    LogBook::instance()->commit();

} catch( LogBookException $e ) { report_error( $e->toHtml()); }

?>
