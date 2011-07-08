<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/*
 * This script will process a request for creating new free-form entry
 * in the specified scope.
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
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
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

if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' ) report_error_and_exit( "experiment identifier can't be empty" );
} else report_error_and_exit( "no valid experiment identifier" );

if( isset( $_POST['message_text'] )) $message = trim( $_POST['message_text'] );
else report_error_and_exit( "no valid message text" );

/* The author's name (if provided) would take a precedence
 * over the author's account which is mandatory.
 */
if( isset( $_POST['author_account'] )) {
    $author = trim( $_POST['author_account'] );
    if( isset( $_POST['author_name'] )) {
        $str = trim( $_POST['author_name'] );
        if( $str != '' ) $author = $str;
    }
} else report_error_and_exit( "no valid author text" );

$shift_id = null;
$run_id   = null;
$run_num  = null;

if( isset( $_POST['scope'] )) {

	$scope = trim( $_POST['scope'] );
    if( $scope == '' ) report_error_and_exit( "scope can't be empty" );

    else if( $scope == 'shift' ) {
        if( isset( $_POST['shift_id'] )) {
            $shift_id = trim( $_POST['shift_id'] );
            if( $shift_id == '' ) report_error_and_exit( "shift id can't be empty" );
        } else report_error_and_exit( "no valid shift id" );

    } else if( $scope == 'run' ) {
        if( isset( $_POST['run_id'] )) {
            $run_id = trim( $_POST['run_id'] );
            if( $run_id == '' ) report_error_and_exit( "run id can't be empty" );
        } else if( isset( $_POST['run_num'] )) {
            $run_num = trim( $_POST['run_num'] );
            if( $run_num == '' ) report_error_and_exit( "run number can't be empty" );
        } else report_error_and_exit( "no valid run id or number" );

    } else if( $scope == 'message' ) {
        if( isset( $_POST['message_id'] )) {
            $message_id = trim( $_POST['message_id'] );
            if( $message_id == '' ) report_error_and_exit( "parent message id can't be empty" );
        } else report_error_and_exit( "no valid parent message id" );
    }

} else report_error_and_exit( "no valid scope" );

$relevance_time_enforced = false;
$relevance_time          = LusiTime::now();
if( isset( $_POST['relevance_time'] )) {
	$str = trim( $_POST['relevance_time'] );
	if( $str != '' ) {
		$relevance_time_enforced = true;
		if( $str == 'now' ) {
			;
		} else {
			$relevance_time = LusiTime::parse( $str );
			if( is_null( $relevance_time )) report_error_and_exit( "incorrect format of the relevance time" );
		}
	}
}

/* Process optional tags
 */
$tags = array();
if( isset( $_POST['num_tags'] )) {
    sscanf( trim( $_POST['num_tags'] ), "%d", $num_tags )
        or report_error_and_exit( "not a number where a number of tags was expected" );
    for( $i=0; $i < $num_tags; $i++ ) {
        $tag_name_key  = 'tag_name_'.$i;
        if( isset( $_POST[$tag_name_key] )) {

            $tag = trim( $_POST[$tag_name_key] );
            if( $tag != '' ) {

                $tag_value_key = 'tag_value_'.$i;
                if( !isset( $_POST[$tag_value_key] )) {
                    report_error_and_exit( "No valid value for tag {$tag_name_key}" );
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

/* Process optional attachments
 */
function upload_err2string( $errcode ) {

	switch( $errcode ) {
		case UPLOAD_ERR_OK:
			return "There is no error, the file uploaded with success.";
		case UPLOAD_ERR_INI_SIZE:
			return "The uploaded file exceeds the maximum of ".get_ini("upload_max_filesize")." in this Web server configuration.";
		case UPLOAD_ERR_FORM_SIZE:
			return "The uploaded file exceeds the maximum of ".$_POST["MAX_FILE_SIZE"]." that was specified in the sender's HTML form.";
		case UPLOAD_ERR_PARTIAL:
			return "The uploaded file was only partially uploaded.";
		case UPLOAD_ERR_NO_FILE:
			return "No file was uploaded.";
		case UPLOAD_ERR_NO_TMP_DIR:
			return "Missing a temporary folder in this Web server installation.";
		case UPLOAD_ERR_CANT_WRITE:
			return "Failed to write file to disk at this Web server installation.";
		case UPLOAD_ERR_EXTENSION:
			return "A PHP extension stopped the file upload.";
	}
	return "Unknown error code: ".$errorcode;
}

$files = array();
foreach( array_keys( $_FILES ) as $file_key ) {

    $name  = $_FILES[$file_key]['name'];
    $error = $_FILES[$file_key]['error'];

    if( $error != UPLOAD_ERR_OK ) {
    	if( $error == UPLOAD_ERR_NO_FILE ) continue;
   		report_error_and_exit(
   			"Attachment '{$name}' couldn't be uploaded because of the following problem: '".
   			upload_err2string( $error )."'."
   		);
    }
    if( $name ) {

        // Read file contents into a local variable
        //
        $location = $_FILES[$file_key]['tmp_name'];
        $fd = fopen( $location, 'r' )
            or report_error_and_exit( "failed to open file: {$location}" );
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

/* Translate an entry into a JSON object. Return the serialized object.
 * 
 * TODO: Attention please!!! This code has been cut-and-paste from 'Search.php'.
 *       Potentially this may lead to the a situation when two version of the code
 *       will co-exist due to inconsistent editing. Possible solutions would 
 *       include putting the code into a library or refactoring 'Search.php' to return
 *       a single object (like 'SearchOne.php'.
 */
function child2json( $entry, $posted_at_instrument ) {

    $timestamp = $entry->insert_time();

    $tag_ids = array();

    $attachment_ids = array();
    foreach( $entry->attachments() as $attachment )
        array_push(
            $attachment_ids,
            array(
                "id"          => $attachment->id(),
                "type"        => $attachment->document_type(),
                "size"        => $attachment->document_size(),
                "description" => $attachment->description(),
                "url"         => '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>'
            )
        );

    $children_ids = array();
    foreach( $entry->children() as $child )
        array_push( $children_ids, child2json( $child, $posted_at_instrument ));

    $content = wordwrap( $entry->content(), 128 );
    return json_encode(
        array (
            "event_timestamp" => $timestamp->to64(),
            "event_time"      => $entry->insert_time()->toStringShort(),
            "relevance_time"  => is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort(),
            "run"             => '',
            "shift"           => '',
            "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).$entry->author(),
            "id"              => $entry->id(),
            "subject"         => htmlspecialchars( substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' )),
            "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".htmlspecialchars($content)."</pre>",
            "html1"           => "<pre>".htmlspecialchars($content)."</pre>",
        	"content"         => htmlspecialchars( $entry->content()),
            "attachments"     => $attachment_ids,
            "tags"            => $tag_ids,
            "children"        => $children_ids,
            "is_run"          => 0,
            "run_id"          => 0,
            "run_num"         => 0,
        	"ymd"             => $timestamp->toStringDay(),
        	"hms"             => $timestamp->toStringHMS()
        )
    );
}

function entry2json( $entry, $posted_at_instrument ) {

    $timestamp = $entry->insert_time();
    $shift     = is_null( $entry->shift_id()) ? null : $entry->shift();
    $run       = is_null( $entry->run_id())   ? null : $entry->run();

    $tag_ids = array();
    foreach( $entry->tags() as $tag )
        array_push(
            $tag_ids,
            array(
                "tag"   => $tag->tag(),
                "value" => $tag->value()
            )
        );

    $attachment_ids = array();
    foreach( $entry->attachments() as $attachment )
        array_push(
            $attachment_ids,
            array(
                "id"          => $attachment->id(),
                "type"        => $attachment->document_type(),
                "size"        => $attachment->document_size(),
                "description" => $attachment->description(),
                "url"         => '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>'
            )
        );

    $children_ids = array();
    foreach( $entry->children() as $child )
        array_push( $children_ids, child2json( $child, $posted_at_instrument ));

    $content = wordwrap( $entry->content(), 128 );
    return json_encode(
        array (
            "event_timestamp" => $timestamp->to64(),
            "event_time"      => "<a href=\"index.php?action=select_message&id={$entry->id()}\"  target=\"_blank\" class=\"lb_link\">{$timestamp->toStringShort()}</a>",
            "relevance_time"  => is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort(),
            "run"             => is_null( $run   ) ? '' : "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>",
            "shift"           => is_null( $shift ) ? '' : "<a href=\"javascript:select_shift(".$shift->id().")\" class=\"lb_link\">".$shift->begin_time()->toStringShort().'</a>',
            "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).$entry->author(),
            "id"              => $entry->id(),
            "subject"         => htmlspecialchars( substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' )),
            "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".htmlspecialchars($content)."</pre>",
            "html1"           => "<pre>".htmlspecialchars($content)."</pre>",
        	"content"         => htmlspecialchars( $entry->content()),
            "attachments"     => $attachment_ids,
            "tags"            => $tag_ids,
            "children"        => $children_ids,
            "is_run"          => 0,
            "run_id"          => is_null( $run ) ? 0 : $run->id(),
            "run_num"         => is_null( $run ) ? 0 : $run->num(),
        	"ymd"             => $timestamp->toStringDay(),
        	"hms"             => $timestamp->toStringHMS()
        )
    );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id ) or	report_error_and_exit( "no such experiment" );
    $instrument = $experiment->instrument();

    if(( $scope == 'run' ) && is_null( $run_id )) {
    	$run = $experiment->find_run_by_num( $run_num ) or die( "no such run" );
    	$run_id = $run->id();
    }

    LogBookAuth::instance()->canPostNewMessages( $experiment->id()) or
        report_error_and_exit( 'You are not authorized to post messages for the experiment' );

    /* For messages to be posted for runs we may also make an optional (if
     * no relevance time is requested explicitly) adjustment to the relevance time
     * by incrementing the timestamp by one nanosecond from the run end time
     * if the run is closed. Otherwise do the same for the begin time of the run.
     */
    if( !is_null( $run_id ) && !$relevance_time_enforced ) {
    	$run  = $experiment->find_run_by_id( $run_id ) or report_error_and_exit( "no such run" );
    	$time = $run->end_time();
    	if( is_null( $time )) $time = $run->begin_time();
    	$sec  = $time->sec;
    	$nsec = $time->nsec + 1;
    	if( $nsec >= 1000*1000*1000 ) {
    		$sec += 1;
    		$nsec = 0;
    	}
    	$relevance_time = new LusiTime( $sec, $nsec );
    }

    $content_type = "TEXT";

    /* If the request has been made in a scope of some parent entry then
     * one the one and create the new one in its scope.
     *
     * NOTE: Remember that child entries have no tags, but they
     *       are allowed to have attachments.
	 */
    if( $scope == 'message' ) {
        $parent = $experiment->find_entry_by_id( $message_id ) or report_error_and_exit( "no such parent message exists" );
        $entry = $parent->create_child( $author, $content_type, $message );
    } else {
        $entry = $experiment->create_entry( $author, $content_type, $message, $shift_id, $run_id, $relevance_time );
        foreach( $tags as $t )
            $tag = $entry->add_tag( $t['tag'], $t['value'] );
    }
    foreach( $files as $f )
        $attachment = $entry->attach_document( $f['contents'], $f['type'], $f['description'] );

    $experiment->notify_subscribers( $entry );

    /* Return a JSON object describing the newly created entry back to the caller.
     */
    $status_encoded = json_encode( "success" );
    $entry_encoded  = $scope == 'message' ? child2json( $entry, false ) : entry2json( $entry, false );

	print <<< HERE
<textarea>
{
  "Status": {$status_encoded},
  "Entry": {$entry_encoded}
}
</textarea>
HERE;

	$logbook->commit();

} catch( LogBookException  $e ) { print $e->toHtml(); }
  catch( LusiTimeException $e ) { print $e->toHtml(); }

?>
