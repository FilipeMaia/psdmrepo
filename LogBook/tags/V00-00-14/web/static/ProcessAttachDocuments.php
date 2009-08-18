<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for attaching documents to
 * an existingfree-form entry.
 */
$entry_id = null;
if( isset( $_POST['entry_id'] )) {
    $str = trim( $_POST['entry_id'] );
    if( $str != '' && ( 1 != sscanf( $str, "%ud", $entry_id )))
        die( "not a number at an entry identifier" );
} else
    die( "no valid entry id" );

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

// Read the names of files submitted for uploading
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

    if( is_null( $entry_id )) {
        $experiment = $logbook->find_experiment_by_name( $experiment_name )
            or die("no such experiment" );

        $entry = $experiment->find_last_entry()
            or die( "the experiment has no single entry yet - create the one first" );
    } else {
        $entry = $logbook->find_entry_by_id( $entry_id )
            or die( "no such entry" );
    }

    // Attach files
    //
    foreach( $files as $f ) {
        $attachment = $entry->attach_document( $f['contents'], $f['type'], $f['description'] );
    }
?>
<!--
The page for reporting the modified entry.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Modified Entry Report</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <h1>Free-form entry which just has been modified</h1>
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