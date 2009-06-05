<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for listing shifts of an
 * experiment.
 */
if( isset( $_GET['id'] )) {
    if( 1 != sscanf( trim( $_GET['id'] ), "%d", $id ))
        die( "invalid format of the attachment identifier" );
} else
    die( "no valid attachment identifier" );

/* Proceed to the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $attachment = $logbook->find_attachment_by_id( $id )
        or die("no such attachment" );

    header( "Content-type: {$attachment->document_type()}" );
    echo( $attachment->document());

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>