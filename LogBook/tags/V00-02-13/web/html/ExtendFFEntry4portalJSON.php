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
 * This script will process a request for extending an existing free-form entry.
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

if( isset( $_POST['message_id'] )) {
    $message_id = trim( $_POST['message_id'] );
    if( $message_id == '' ) report_error_and_exit( "parent message id can't be empty" );
} else report_error_and_exit( "no valid parent message id" );

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
$files = array();
foreach( array_keys( $_FILES ) as $file_key ) {

    $name  = $_FILES[$file_key]['name'];
    $error = $_FILES[$file_key]['error'];

    if( $error != UPLOAD_ERR_OK ) {
    	if( $error == UPLOAD_ERR_NO_FILE ) continue;
   		report_error_and_exit(
   			"Attachment '{$name}' couldn't be uploaded because of the following problem: '".
   			LogBookUtils::upload_err2string( $error )."'."
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
 */
function extended2json( $tags, $attachments ) {

    $tag_ids = array();
    foreach( $tags as $tag )
        array_push(
            $tag_ids,
            array(
                "tag"   => $tag->tag(),
                "value" => $tag->value()
            )
        );

    $attachment_ids = array();
    foreach( $attachments as $attachment )
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

    return json_encode(
        array (
            "attachments" => $attachment_ids,
            "tags"        => $tag_ids
        )
    );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $entry      = $logbook->find_entry_by_id( $message_id ) or	report_error_and_exit( "no such message" );
    $experiment = $entry->parent();
    $instrument = $experiment->instrument();

    LogBookAuth::instance()->canPostNewMessages( $experiment->id()) or
        report_error_and_exit( 'You are not authorized to extend messages for the experiment' );

    $extended_tags = array();
    foreach( $tags as $t )
        array_push(
        	$extended_tags,
        	$entry->add_tag( $t['tag'], $t['value'] ));

    $extended_attachments = array();
    foreach( $files as $f )
        array_push(
        	$extended_attachments,
        	$entry->attach_document( $f['contents'], $f['type'], $f['description'] ));

    /* Return a JSON object describing extended attachments and tags back to the caller.
     */
    $status_encoded = json_encode( "success" );
    $extended_encoded  = extended2json( $extended_tags, $extended_attachments );

	print <<< HERE
<textarea>
{
  "Status": {$status_encoded},
  "Extended": {$extended_encoded}
}
</textarea>
HERE;

	$logbook->commit();

} catch( LogBookException  $e ) { print $e->toHtml(); }
  catch( LusiTimeException $e ) { print $e->toHtml(); }

?>
