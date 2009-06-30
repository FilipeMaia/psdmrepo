<?php

// Define a maximum size for the uploaded files in Kb
//
define( "MAX_SIZE", "1000" );

// Check if the form has been submitted
//
if( !isset( $_POST['upload_file'] ))
    die( 'This is not a file uploading request' );

// Read the name of the file submitted for uploading
//
$file = $_FILES['file']['name'];
if( !$file )
    die( 'No file in the upload request' );

$filename = $_FILES['file']['tmp_name'];
$filesize = filesize( $filename );
if( $filesize > MAX_SIZE*1024 )
    die( 'allowed server-side file size exceeded' );

$fd = fopen( $filename, 'r' )
    or die( "failed to open file: {$filename}" );

$contents = fread( $fd, $filesize );
$filetype = $_FILES['file']['type'];

// Mirror the uploaded file back
//
header( "Content-type: {$filetype}" );
echo( $contents );
?>