<?php

require_once('LogBook/LogBook.inc.php');


/*
 * This script will process a request for creating new free-form entry
 * in the specified scope.
 */
if( !LogBookAuth::isAuthenticated()) return;

if( isset( $_POST['id'] )) {
    $id = trim( $_POST['id'] );
    if( $id == '' ) {
        die( "experiment identifier can't be empty" );
    }
} else {
    die( "no valid experiment identifier" );
}
if( isset( $_POST['message_text'] )) {
    $message = trim( $_POST['message_text'] );
} else {
    die( "no valid message text" );
}
// The author's name (if provided) would take a precedence
// over the author's account which is mandatory.
//
if( isset( $_POST['author_account'] )) {
    $author = trim( $_POST['author_account'] );
    if( isset( $_POST['author_name'] )) {
        $str = trim( $_POST['author_name'] );
        if( $str != '' ) $author = $str;
    }
} else {
    die( "no valid author text" );
}

$shift_id = null;
$run_id = null;
$relevance_time = LusiTime::now();

if( isset( $_POST['scope'] )) {
    $scope = trim( $_POST['scope'] );
    if( $scope == '' ) {
        die( "scope can't be empty" );
    } else if( $scope == 'shift' ) {
        if( isset( $_POST['shift_id'] )) {
            $shift_id = trim( $_POST['shift_id'] );
            if( $shift_id == '' ) {
                die( "shift id can't be empty" );
            }
        } else {
            die( "no valid shift id" );
        }
    } else if( $scope == 'run' ) {
        if( isset( $_POST['run_id'] )) {
            $run_id = trim( $_POST['run_id'] );
            if( $run_id == '' )
                die( "run id can't be empty" );
        } else {
            die( "no valid run id" );
        }
    } else if( $scope == 'message' ) {
        if( isset( $_POST['message_id'] )) {
            $message_id = trim( $_POST['message_id'] );
            if( $message_id == '' )
                die( "parent message id can't be empty" );
        } else {
            die( "no valid parent message id" );
        }
    }
} else {
    die( "no valid scope" );
}

// Read optional tags submitted with the entry
//
$tags = array();
if( isset( $_POST['num_tags'] )) {
    sscanf( trim( $_POST['num_tags'] ), "%d", $num_tags )
        or die( "not a number where a number of tags was expected" );
    for( $i=0; $i < $num_tags; $i++ ) {
        $tag_name_key  = 'tag_name_'.$i;
        if( isset( $_POST[$tag_name_key] )) {

            $tag = trim( $_POST[$tag_name_key] );
            if( $tag != '' ) {

                $tag_value_key = 'tag_value_'.$i;
                if( !isset( $_POST[$tag_value_key] )) {
                    die( "No valid value for tag {$tag_name_key}" );
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

// Read the names of optional files submitted for uploading
//
define( MAX_SIZE,  1024*1024 );  // max size for each uploaded file (Bytes)
define( MAX_FILES, 3 );          // max number of files to attach

$files = array();
$file_keys = array_keys( $_FILES );
foreach( $file_keys as $file_key ) {

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
        $description_key = $file_key;
        if( isset( $_POST[$description_key] )) {
            $description = trim( $_POST[$description_key] );
            if( $description == '' )
                $description = $file;
        } else
            die( "no valid description posted for file number {$file}" );

        array_push(
            $files,
            array(
                'type' => $filetype,
                'description' => $description,
                'contents' => $contents ));
    }
}

if( isset( $_POST['actionSuccess'] )) {
    $actionSuccess = trim( $_POST['actionSuccess'] );
}

try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $content_type = "TEXT";

    // If the request has been made in a scope of some parent entry then
    // one the one and create the new one in its scope.
    //
    // NOTE: Remember that child entries have no tags!

    if( $scope == 'message' ) {
        $parent = $experiment->find_entry_by_id( $message_id )
            or die( "no such parent message exists" );
        $entry = $parent->create_child(
            $author, $content_type, $message );
    } else {
        $entry = $experiment->create_entry(
            $author, $content_type, $message,
            $shift_id, $run_id, $relevance_time );

        // Add tags (if any)
        //
        foreach( $tags as $t ) {
            $tag = $entry->add_tag( $t['tag'], $t['value'] );
        }
    }

    // Attach files (if any)
    //
    foreach( $files as $f ) {
        $attachment = $entry->attach_document(
            $f['contents'], $f['type'], $f['description'] );
    }
    $logbook->commit();

    // Return back to the caller
    //
    if( isset( $actionSuccess )) {
        if( $actionSuccess == 'select_experiment' ) {
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name());
        } else if( $actionSuccess == 'select_experiment_and_shift' ) {
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$shift_id);
        } else if( $actionSuccess == 'select_experiment_and_run' ) {
            $run = $experiment->find_run_by_id( $run_id )
                or die( "no such run" );
            header( "Location: index.php?action={$actionSuccess}".
                '&instr_id='.$experiment->instrument()->id().
                '&instr_name='.$experiment->instrument()->name().
                '&exper_id='.$experiment->id().
                '&exper_name='.$experiment->name().
                '&shift_id='.$run->shift()->id().
                '&run_id='.$run_id);
        } else {
            ;
        }
    }

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
