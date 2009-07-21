<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will generate a module with input elements for the search form
 * in a context of the specified experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' ) {
        die( "experiment identifier can't be empty" );
    }
} else {
    die( "no valid experiment identifier" );
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or die( "no such experiment" );

    /* Get names of all tags and authors which have ever been used in free-form
     * entries of the experiment (including its shifts and runs).
     */
    $tags = $experiment->used_tags();
    array_unshift( $tags, '' );

    $authors = $experiment->used_authors();
    array_unshift( $authors, '' );

    $begin_time_title =
        "Valid format:\n".
        "\t".LusiTime::now()->toStringShort()."\n".
        "Also the (case neutral) shortcuts are allowed:\n".
        "\t'm' - month (-31 days) ago\n".
        "\t'w' - week (-7 days) ago\n".
        "\t'd' - day (-24 hours) ago\n".
        "\t'y' - since yesterday (at 00:00:00)\n".
        "\t't' - today (at 00:00:00)\n".
        "\t'h' - an hour (-60 minutes) ago";

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 200, 600 );
    echo $con
        ->label       (   0,   0, 'Text to search:' )
        ->value_input (   0,  20, 'text2search' )

        ->label         (   0,  50, 'Search in:' )
        ->checkbox_input(   0,  70, 'search_in_messages','Message'   , true )->label(  20,  70, 'message body',  false )
        ->checkbox_input(   0,  90, 'search_in_tags',    'Tag'       , true )->label(  20,  90, 'tags',       false )
        ->checkbox_input(   0, 110, 'search_in_values',  'Value'     , true )->label(  20, 110, 'tag values', false )
        ->label         ( 120,  50, 'Posted at:' )
        ->checkbox_input( 120,  70, 'posted_at_experiment', 'Experiment', true )->label( 140,  70, 'experiment', false )
        ->checkbox_input( 120,  90, 'posted_at_shifts',     'Shifts'    , true )->label( 140,  90, 'shifts',     false )
        ->checkbox_input( 120, 110, 'posted_at_runs',       'Runs'      , true )->label( 140, 110, 'runs',       false )

        ->label       (   0, 140, 'Begin Time:' )
        ->value_input (   0, 160, 'begin', '', $begin_time_title )
        ->label       (   0, 190, 'End Time:' )
        ->value_input (   0, 210, 'end',   '', LusiTime::now()->toStringShort() )
        ->label       (   0, 240, 'Tag:' )
        ->select_input(   0, 260, 'tag', $tags, '' )
        ->label       (   0, 290, 'Posted by:' )
        ->select_input(   0, 310, 'author', $authors, '' )
      //->button      (   0, 360, 'reset_form_button',  'Reset', 'reset form to its initial state' )
      //->button      (  75, 360, 'submit_search_button', 'Search', 'initiate the search operation' )
        ->button      (   0, 350, 'reset_form_button',  'Reset', 'reset form to its initial state' )

        ->label         (   0, 410, 'Presentation format:' )
        ->radio_input   (   0, 430, 'presentation_format', 'compact',  true  )->label( 20, 430, 'compact', false )
        ->radio_input   (   0, 450, 'presentation_format', 'detailed', false )->label( 20, 450, 'detailed', false )
        ->checkbox_input(  80, 450, 'preview_attachments', 'Preview', false )->label( 100, 450, 'preview attachments',  false )

        ->label         (   0, 490, 'Show on page:' )
        ->radio_input   (   0, 510, 'show_on_page', 'all',   false )->label( 20, 510, 'all',  false )
        ->radio_input   (   0, 530, 'show_on_page', 'limit', true  )->label( 20, 530, 'limit to:', false )
        ->select_input  (  80, 525, 'limit_per_page', Array( 5, 10, 20, 50, 100 ))

        ->button        (   0, 570, 'submit_search_button', 'Search', 'initiate the search operation' )

        ->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>