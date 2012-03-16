<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying a free-form entry.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "entry identifier can't be empty" );
} else
    die( "no valid expentryeriment identifier" );

try {

    $logbook = new LogBook();
    $logbook->begin();

    $entry = $logbook->find_entry_by_id( $id )
        or die( "no such entry" );

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
    $relevance_time_str   = is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort();
    $shift_begin_time_str = is_null( $entry->shift_id())       ? 'n/a' : $entry->shift()->begin_time()->toStringShort();
    $run_number_str       = is_null( $entry->run_id())         ? 'n/a' : $entry->run()->num();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 550, 325 );
    echo $con
        ->label    (   0,   0, 'Experiment:' )
        ->value    ( 100,   0, $instrument->name().' / '.$experiment->name())
        ->label    ( 250,  25, 'By:'        )->value( 300,  25, $entry->author())
        ->label    (   0,  25, 'Posted:'    )->value( 100,  25, $entry->insert_time()->toStringShort())
        ->label    (   0,  50, 'Relevance:' )->value( 100,  50, $relevance_time_str )
        ->label    ( 250,  50, 'Run:'       )->value( 300,  50, $run_number_str )
        ->label    ( 350,  50, 'Shift:'     )->value( 400,  50, $shift_begin_time_str )
        ->textarea (   0,  75, $entry->content(), 550, 200 )
        ->label    (   0, 300, 'Tags' )
        // ->container(   0, 325, 'entry_tags_table_container' )
        ->label    ( 250, 300, 'Attachments' )
        //->container( 250, 325, 'entry_attachments_table_container' )
        ->html();

    $logbook->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>
