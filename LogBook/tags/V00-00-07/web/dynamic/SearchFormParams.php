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

    $con = new RegDBHtml( 0, 0, 180, 400 );
    echo $con
        ->label       (   0,   0, 'Text to search' )
        ->value_input (   0,  20, 'text2search' )

        ->label         (   0,  50, 'Search in' )
        ->checkbox_input(   0,  70, 'search_experiment', 'Experiment', true )->label(  20,  70, 'experiment', false )
        ->checkbox_input(   0,  90, 'search_shifts',     'Shifts'    , true )->label(  20,  90, 'shifts', false )
        ->checkbox_input(   0, 110, 'search_runs',       'Runs'      , true )->label(  20, 110, 'runs', false )
        ->checkbox_input( 100,  70, 'search_tags',       'Tag'       , true )->label( 120,  70, 'tags', false )
        ->checkbox_input( 100,  90, 'search_values',     'Value'     , true )->label( 120,  90, 'tag values', false )

        ->label       (   0, 140, 'Begin Time' )
        ->value_input (   0, 160, 'begin', '', $begin_time_title )
        ->label       (   0, 190, 'End Time' )
        ->value_input (   0, 210, 'end',   '', LusiTime::now()->toStringShort() )
        ->label       (   0, 240, 'Tag' )
        ->select_input(   0, 260, 'tag', $tags, '' )
        ->label       (   0, 290, 'Posted by' )
        ->select_input(   0, 310, 'author', $authors, '' )
        ->button      (   0, 360, 'reset_form_button',  'Reset', 'reset form to its initial state' )
        ->button      (  75, 360, 'submit_search_button', 'Search', 'initiate the search operation' )


        ->html();

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>