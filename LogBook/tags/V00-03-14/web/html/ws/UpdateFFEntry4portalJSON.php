<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;
use LogBook\LogBookUtils;

/*
 * This script will process a request for updating existing message.
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

$onsuccess = "../index.php";
if( isset( $_POST['onsuccess'] )) {
    $onsuccess = trim( $_POST['onsuccess'] );
}
$onfailure = $onsuccess;
if( isset( $_POST['onfailure'] )) {
    $onfailure = trim( $_POST['onfailure'] );
}


if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' ) {
        report_error_and_exit( "message identifier can't be empty" );
    }
} else {
    report_error_and_exit( "no valid experiment identifier" );
}
if( isset( $_POST['content_type'] )) {
    $content_type = trim( $_POST['content_type'] );
    if( $content_type == '' )
        report_error_and_exit( "the content type of the free-form entry can't be empty" );
} else {
    report_error_and_exit( "no valid content type provided for the entry" );
}
if( isset( $_POST['content'] )) {
    $content = trim( $_POST['content'] );
} else {
    report_error_and_exit( "no valid content provided for the entry" );
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

try {

    LogBook::instance()->begin();

    $entry      = LogBook::instance()->find_entry_by_id( $id ) or report_error_and_exit( "no such free-form entry" );
    $experiment = $entry->parent();

    if( !LogBookAuth::instance()->canEditMessages( $experiment->id()))
        report_error_and_exit( 'You are not authorized to edit messages for the experiment' );

    $entry->update_content( $content_type, $content );
    foreach( $files as $f )
        $attachment = $entry->attach_document( $f['contents'], $f['type'], $f['description'] );

    $experiment->notify_subscribers( $entry, /* new_vs_modified = */ false );

    /* Return a JSON object describing the newly created entry back to the caller.
     */
    $status_encoded = json_encode( "success" );
    $entry_encoded  = $entry->parent_entry_id() ? LogBookUtils::child2json( $entry, false ) : LogBookUtils::entry2json( $entry, false );

	print <<< HERE
<textarea>
{
  "Status": {$status_encoded},
  "Entry": {$entry_encoded}
}
</textarea>
HERE;
    
    LogBook::instance()->commit();

} catch( LogBookException $e ) { report_error_and_exit($e->toHtml()); }

?>
