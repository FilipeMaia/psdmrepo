<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

/*
 * This script will process a request for displaying tags of a free-form entry.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "free-form entry identifier can't be empty" );
} else
    die( "no valid free-form entry identifier" );

function tag2json( $t ) {
    return json_encode(
        array (
            "tag" => $t->tag(),
            "value" => $t->value()
        )
    );
}

/*
 * Return JSON objects with a list of tags.
 */
try {

    LogBook::instance()->begin();

    $entry      = LogBook::instance()->find_entry_by_id( $id ) or die( "no such free-form entry" );
    $experiment = $entry->parent();
    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Proceed to the operation
    //
    $tags = $entry->tags();

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach( $tags as $t ) {
      if( $first ) {
          $first = false;
          echo "\n".tag2json( $t );
      } else {
          echo ",\n".tag2json( $t );
      }
    }
    print <<< HERE
 ] } }
HERE;

    LogBook::instance()->commit();

} catch( LogBookException $e ) { print $e->toHtml(); }
?>
