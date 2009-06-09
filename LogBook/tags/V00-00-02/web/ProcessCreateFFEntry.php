<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new free-form
 * entry in the database.
 */
$parent_entry_id = null;
if( isset( $_POST['parent_entry_id'] )) {
    $str = trim( $_POST['parent_entry_id'] );
    if( $str != '' && ( 1 != sscanf( $str, "%ud", $parent_entry_id )))
        die( "not a number at parent entry identifier" );
} else
    die( "no parent entry id" );

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['relevance_time'])) {
    $relevance_time = LogBookTime::parse( trim( $_POST['relevance_time'] ));
    if( is_null( $relevance_time ))
        die("relevance time has invalid format");
} else
    die( "no relevance time for shift" );

if( isset( $_POST['author'] )) {
    $author = trim( $_POST['author'] );
    if( $author == '' )
        die( "author account name can't be empty" );
} else
    die( "no valid author account" );

if( isset( $_POST['content_type'] )) {
    $content_type = trim( $_POST['content_type'] );
    if( $content_type == '' )
        die( "content type can't be empty" );
} else
    die( "no valid content type" );

if( isset( $_POST['content'] ))
    $content = trim( $_POST['content'] );
else
    die( "no valid content" );

// Read optional tags submitted with the entry
//
define( MAX_TAGS, 3 );          // max number of tags to attach

$tags = array();
for( $i = 0; $i < MAX_TAGS; $i++ ) {

    $tag_name_key  = 'tag_name'.$i;
    if( isset( $_POST[$tag_name_key] )) {

        $tag = trim( $_POST[$tag_name_key] );
        if( $tag != '' ) {

            $tag_value_key = 'tag_value'.$i;
            if( !isset( $_POST[$tag_value_key] ))
                die( "No valid value for tag {$tag_name_key}" );

            $value = trim( $_POST[$tag_value_key] );

            array_push(
                $tags,
                array(
                    'tag' => $tag,
                    'value' => $value ));
        }
    }
}

// Read the names of optional files submitted for uploading
//
define( MAX_SIZE,  1024*1024 );  // max size for each uploaded file (Bytes)
define( MAX_FILES, 3 );          // max number of files to attach

$files = array();
for( $i = 0; $i < MAX_FILES; $i++ ) {

    $file_key = 'file'.$i;

    $file = $_FILES[$file_key]['name'];
    if( $file ) {

        $filetype = $_FILES[$file_key]['type'];

        // Read file contents into alocal variable
        //
        $filename = $_FILES[$file_key]['tmp_name'];
        $filesize = filesize( $filename );
        if( $filesize > MAX_SIZE )
            die( 'allowed server-side file size exceeded' );
        $fd = fopen( $filename, 'r' )
            or die( "failed to open file: {$filename}" );
        $contents = fread( $fd, $filesize );
        fclose( $fd );

        // Get its description. If none is present then use the original
        // name of the file at client's side.
        //
        $description_key = 'description'.$i;
        if( isset( $_POST[$description_key] )) {
            $description = trim( $_POST[$description_key] );
            if( $description == '' )
                $description = $file;
        } else
            die( "no valid description posted for file number {$i}" );

        array_push(
            $files,
            array(
                'type' => $filetype,
                'description' => $description,
                'contents' => $contents ));
    }
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    if( is_null( $parent_entry_id )) {

        $experiment = $logbook->find_experiment_by_name( $experiment_name )
            or die("no such experiment" );

        $entry = $experiment->create_entry(
            $relevance_time, $author, $content_type, $content );

    } else {

        $parent_entry = $logbook->find_entry_by_id( $parent_entry_id )
            or die( "no such entry" );

        $entry = $parent_entry->create_child(
            $author, $content_type, $content );
    }

    // Add tags (if any)
    //
    foreach( $tags as $t ) {
        $tag = $entry->add_tag( $t['tag'], $t['value'] );
    }

    // Attach files (if any)
    //
    foreach( $files as $f ) {
        $attachment = $entry->attach_document( $f['contents'], $f['type'], $f['description'] );
    }
?>
<!--
The page for reporting the information on the new entry.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Newely created entry</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <h1>Free-form entry which just has been created</h1>
        <h2><?php echo $entry->parent()->name(); ?></h2>
        <?php
        LogBookTestTable::Entry( "table_4" )->show( array( $entry ));
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>