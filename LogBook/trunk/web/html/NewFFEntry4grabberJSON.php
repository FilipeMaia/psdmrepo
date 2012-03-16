<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookUtils;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request from e-log Grabber for creating new free-form
 * entry in the specified scope.
 *
 * A JSON object with teh result of the operation will be returned.
 * If the operation was successfull then the reply will also contain
 * a message identifier of the newely create message.
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

if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' ) report_error( "experiment identifier can't be empty" );
} else report_error( "no valid experiment identifier" );

if( isset( $_POST['message_text'] )) $message = trim( $_POST['message_text'] );
else report_error( "no valid message text" );

/* The author's name (if provided) would take a precedence
 * over the author's account which is mandatory.
 */
if( isset( $_POST['author_account'] )) {
    $author = trim( $_POST['author_account'] );
    if( isset( $_POST['author_name'] )) {
        $str = trim( $_POST['author_name'] );
        if( $str != '' ) $author = $str;
    }
} else report_error( "no valid author text" );

$shift_id = null;
$run_id   = null;
$run_num  = null;

if( isset( $_POST['scope'] )) {

	$scope = trim( $_POST['scope'] );
    if( $scope == '' ) report_error( "scope can't be empty" );

    else if( $scope == 'shift' ) {
        if( isset( $_POST['shift_id'] )) {
            $shift_id = trim( $_POST['shift_id'] );
            if( $shift_id == '' ) report_error( "shift id can't be empty" );
        } else report_error( "no valid shift id" );

    } else if( $scope == 'run' ) {
        if( isset( $_POST['run_id'] )) {
            $run_id = trim( $_POST['run_id'] );
            if( $run_id == '' ) report_error( "run id can't be empty" );
        } else if( isset( $_POST['run_num'] )) {
            $run_num = trim( $_POST['run_num'] );
            if( $run_num == '' ) report_error( "run number can't be empty" );
        } else report_error( "no valid run id or number" );

    } else if( $scope == 'message' ) {
        if( isset( $_POST['message_id'] )) {
            $message_id = trim( $_POST['message_id'] );
            if( $message_id == '' ) report_error( "parent message id can't be empty" );
        } else report_error( "no valid parent message id" );
    }

} else report_error( "no valid scope" );

$relevance_time = LusiTime::now();
if( isset( $_POST['relevance_time'] )) {
	$str = trim( $_POST['relevance_time'] );
	if( $str != '' ) {
		if( $str == 'now' ) {
			;
		} else {
			$relevance_time = LusiTime::parse( $str );
			if( is_null( $relevance_time )) report_error( "incorrect format of the relevance time" );
		}
	}
}

/* Process optional tags
 */
$tags = array();
if( isset( $_POST['num_tags'] )) {
    sscanf( trim( $_POST['num_tags'] ), "%d", $num_tags )
        or report_error( "not a number where a number of tags was expected" );
    for( $i=0; $i < $num_tags; $i++ ) {
        $tag_name_key  = 'tag_name_'.$i;
        if( isset( $_POST[$tag_name_key] )) {

            $tag = trim( $_POST[$tag_name_key] );
            if( $tag != '' ) {

                $tag_value_key = 'tag_value_'.$i;
                if( !isset( $_POST[$tag_value_key] )) {
                    report_error( "No valid value for tag {$tag_name_key}" );
                }
                $value = trim( $_POST[$tag_value_key] );

                array_push(
                    $tags,
                    array(
                        'tag' => $tag,
                        'value' => $value ));
            }
        }
    }
}
$text4child = '';
if( isset( $_POST['text4child'] )) $text4child = trim($_POST['text4child']);

/* Process optional attachments
 */
$files = array();
foreach( array_keys( $_FILES ) as $file_key ) {

    $name  = $_FILES[$file_key]['name'];
    $error = $_FILES[$file_key]['error'];

    if( $error != UPLOAD_ERR_OK ) {
    	if( $error == UPLOAD_ERR_NO_FILE ) continue;
   		report_error(
   			"Attachment '{$name}' couldn't be uploaded because of the following problem: '".
   			LogBookUtils::upload_err2string( $error )."'."
   		);
    }
    if( $name ) {

        // Read file contents into a local variable
        //
        $location = $_FILES[$file_key]['tmp_name'];
        $fd = fopen( $location, 'r' )
            or report_error( "failed to open file: {$location}" );
        $contents = fread( $fd, filesize( $location ) );
        fclose( $fd );

        // Get its description. If none is present then use the original
        // name of the file at client's side.
        //
        $description = $name;
        if( isset( $_POST[$file_key] )) {
            $str = trim( $_POST[$file_key] );
            if( $str != '' ) $description = $str;
        }
        array_push(
            $files,
            array(
                'type'        => $_FILES[$file_key]['type'],
                'description' => $description,
                'contents'    => $contents ));
    }
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id ) or	report_error( "no such experiment" );
    $instrument = $experiment->instrument();

    if(( $scope == 'run' ) && is_null( $run_id )) {
    	$run = $experiment->find_run_by_num( $run_num );
    	if( is_null($run)) {
    		$first_run = $experiment->find_first_run();
    		$last_run  = $experiment->find_last_run();
    		report_error(
    			(is_null($first_run) || is_null($last_run)) ?
    				"No runs have been taken by this experiment yet." :
    				"Run number {$run_num} has not been found. Allowed range of runs is: {$first_run->num()}..{$last_run->num()}."
    		);
    	}
    	$run_id = $run->id();
    }

    LogBookAuth::instance()->canPostNewMessages( $experiment->id()) or
        report_error( 'You are not authorized to post messages for the experiment' );

    $content_type = "TEXT";

    /* If the request has been made in a scope of some parent entry then
     * one the one and create the new one in its scope.
     *
     * NOTE: Remember that child entries have no tags, but they
     *       are allowed to have attachments.
	 */
    if( $scope == 'message' ) {
        $parent = $experiment->find_entry_by_id( $message_id ) or report_error( "no such parent message exists" );
        $entry = $parent->create_child( $author, $content_type, $message );
    } else {
        $entry = $experiment->create_entry( $author, $content_type, $message, $shift_id, $run_id, $relevance_time );
        foreach( $tags as $t )
            $tag = $entry->add_tag( $t['tag'], $t['value'] );
    }
    foreach( $files as $f )
        $attachment = $entry->attach_document( $f['contents'], $f['type'], $f['description'] );

    if( $text4child != '' )
        $child = $entry->create_child( $author, $content_type, $text4child );

    $experiment->notify_subscribers( $entry );

    $message_id = $entry->id();

	$logbook->commit();

    report_success(array('message_id' => $message_id));

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }

?>
