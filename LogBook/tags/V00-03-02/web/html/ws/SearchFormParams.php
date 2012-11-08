<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDBHtml;
use RegDB\RegDBException;

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

$accross_instrument = isset( $_GET['accross_instrument'] );

/* Proceed with the operation
 */
try {
    LogBook::instance()->begin();

    /* Primary experiment of the request.
     */
    $experiment = LogBook::instance()->find_experiment_by_id( $id ) or die( "no such experiment" );

    /* If 'accross_instrument' mode was requested then harvest tags and author
     * names accross all experiments (of the current instrument) the user is
     * authorized for
     */
    $all_tags = array();
    $all_authors = array();

    $experiments = null;
    if( $accross_instrument ) {
    	$experiments = LogBook::instance()->experiments_for_instrument( $experiment->instrument()->name());
    } else {
    	$experiments = array( $experiment );
    }
    foreach( $experiments as $e ) {

    	/* Check for the authorization
    	 */
    	if( !LogBookAuth::instance()->canRead( $e->id())) {

    		/* Silently skip this experiemnt if browsing accross the whole instrument.
    		 * The only exception would be the main experient from which we started
    		 * things.
    		 */
    		if( $accross_instrument && ( $e->id() != $id )) continue;

	        report_error( 'not authorized to read messages for the experiment' );
    	}
    	
    	/* Get names of all tags and authors which have ever been used in free-form
     	 * entries of the experiment (including its shifts and runs).
     	 */
    	foreach( $e->used_tags   () as $tag    ) $all_tags   [$tag   ] = true;
    	foreach( $e->used_authors() as $author ) $all_authors[$author] = true;
    }
    $tags = array_keys( $all_tags );
    array_unshift( $tags, '' );

    $authors = array_keys( $all_authors );
    array_unshift( $authors, '' );

    $time_title =
        "Valid format:\n".
        "\t".LusiTime::now()->toStringShort()."\n".
        "Also the (case neutral) shortcuts are allowed:\n".
        "\t'b' - the begin time of the experiment\n".
        "\t'e' - the end time of the experiment\n".
        "\t'm' - month (-31 days) ago\n".
        "\t'w' - week (-7 days) ago\n".
        "\t'd' - day (-24 hours) ago\n".
        "\t'y' - since yesterday (at 00:00:00)\n".
        "\t't' - today (at 00:00:00)\n".
        "\t'h' - an hour (-60 minutes) ago";

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 230, 400 );
    echo $con
        ->label       (   0,   0, 'Text to search:' )
        ->value_input (   0,  20, 'text2search', '', '', 32 )

        ->label         (   0,  50, 'Search in:' )
        ->checkbox_input(   0,  70, 'search_in_messages','Message'   , true  )->label(  20,  70, 'message body',  false )
        ->checkbox_input(   0,  90, 'search_in_tags',    'Tag'       , false )->label(  20,  90, 'tags',          false )  // TODO: temporarily disabled
        ->checkbox_input(   0, 110, 'search_in_values',  'Value'     , false )->label(  20, 110, 'tag values',    false )  // TODO: temporarily disabled
        ->label         ( 120,  50, 'Posted at:' )
        ->checkbox_input( 120,  70, 'posted_at_instrument', 'Instrument', $accross_instrument, false, 'load_search_form(this.checked)' )->label( 140,  70, 'instrument', false )
        ->checkbox_input( 120,  90, 'posted_at_experiment', 'Experiment', true  )->label( 140,  90, 'experiment', false )
        ->checkbox_input( 120, 110, 'posted_at_shifts',     'Shifts'    , true  )->label( 140, 110, 'shifts',     false )
        ->checkbox_input( 120, 130, 'posted_at_runs',       'Runs'      , true  )->label( 140, 130, 'runs',       false )

        ->label       (   0, 160, 'Begin Time:' )
        ->value_input (   0, 180, 'begin', '', $time_title )
        ->label       (   0, 210, 'End Time:' )
        ->value_input (   0, 230, 'end',   '', $time_title )
        ->label       (   0, 260, 'Tag:' )
        ->select_input(   0, 280, 'tag', $tags, '' )
        ->label       (   0, 310, 'Posted by:' )
        ->select_input(   0, 330, 'author', $authors, '' )
        ->button      (   0, 380, 'reset_form_button',  'Reset', 'reset form to its initial state' )
        ->button      (  75, 380, 'submit_search_button', 'Search', 'initiate the search operation' )

        ->html();

    LogBook::instance()->commit();

} catch( LogBookException  $e ) { print $e->toHtml(); }
  catch( LusiTimeException $e ) { print $e->toHtml(); }
  catch( RegDBException    $e ) { print $e->toHtml(); }

?>