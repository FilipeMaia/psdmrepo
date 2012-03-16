<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

/*
 * This script will process a request for updating existing message.
 */
$onsuccess = "index.php";
if( isset( $_POST['onsuccess'] )) {
    $onsuccess = trim( $_POST['onsuccess'] );
}
$onfailure = $onsuccess;
if( isset( $_POST['onfailure'] )) {
    $onfailure = trim( $_POST['onfailure'] );
}
function report_error_and_exit( $message ) {
	global $onfailure;
	echo <<<HERE
<center>
  <br>
  <br>
  <div style="background-color:#f0f0f0; border:solid 2px red; max-width:640px;">
    <h1 style="color:red;">Error</h1>
    <div style="height:2px; background-color:red;"></div>
    <p>{$message}</p>
    <p>Click <a href="{$onfailure}">here</a> to return to the previous context</p>
  </div>
</center>
HERE;
	exit;
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

    $logbook = new LogBook();
    $logbook->begin();

    $entry = $logbook->find_entry_by_id( $id )
        or report_error_and_exit( "no such free-form entry" );

    $experiment = $entry->parent();

    if( !LogBookAuth::instance()->canEditMessages( $experiment->id()))
        report_error_and_exit( 'You are not authorized to edit messages for the experiment' );

    $entry->update_content( $content_type, $content );
    foreach( $files as $f )
        $attachment = $entry->attach_document( $f['contents'], $f['type'], $f['description'] );

    $experiment->notify_subscribers( $entry, /* new_vs_modified = */ false );

    /* Return back to the caller
     */
    header( "Location: {$onsuccess}" );
    
    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>
