<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for displaying attachments of a free-form entry.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "free-form entry identifier can't be empty" );
} else
    die( "no valid free-form entry identifier" );

function attachment2json( $a ) {
    /*
    $attachment_url =
        '<a href="javascript:show_attachment('.$a->id().
        ')" title="download and see in a separate browser window">'.
        $a->description().'</a>';
     */
    $title = "download and see in a separate browser window";
    $attachment_url = <<<HERE
<a href="javascript:window.open('ShowAttachment.php?id={$a->id()}')" title="{$title}">{$a->description()}</a>
HERE;
    return json_encode(
        array (
            "attachment" => $attachment_url,
            "size" => $a->document_size(),
        )
    );
}

/*
 * Return JSON objects with a list of attachments.
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $entry = $logbook->find_entry_by_id( $id )
        or die( "no such free-form entry" );
    $attachments = $entry->attachments();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $attachments as $a ) {
      if( $first ) {
          $first = false;
          echo "\n".attachment2json( $a );
      } else {
          echo ",\n".attachment2json( $a );
      }
    }
    print <<< HERE
 ] } }
HERE;

    $logbook->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( LusiTimeException $e ) {
    print $e->toHtml();
}
?>
