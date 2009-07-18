<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will perform the search for free-form entries in a scope
 * of an experiment using values of specified parameter. The result is returned
 * as a JASON obejct which in case of success will have the following format:
 *
 *   "ResultSet": {
 *     "Status": "success",
 *     "Result": [
 *       { "event_time": <timestamp>, "html": <free-form entry markup> }
 *       { .. }
 *     ]
 *   }
 *
 * And in case of any error it will be:
 *
 *   "ResultSet": {
 *     "Status": "error",
 *     "Message": <markup with the explanation>
 *   }
 *
 * Errors are reported via function report_error().
 */
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

if( !isset( $_GET['id'] )) report_error( "no valid experiment id parameter" );
$id = trim( $_GET['id'] );
if( $id == '' ) report_error( "experiment id can't be empty" );

if( !isset( $_GET['text2search'] )) report_error( "no text2search parameter" );
$text2search = trim( $_GET['text2search'] );

if( !isset( $_GET['search_experiment'] )) report_error( "no search_experiment parameter" );
$search_experiment = '0' != trim( $_GET['search_experiment'] );

if( !isset( $_GET['search_shifts'] )) report_error( "no search_shifts parameter" );
$search_shifts = '0' != trim( $_GET['search_shifts'] );

if( !isset( $_GET['search_runs'] )) report_error( "no search_runs parameter" );
$search_runs = '0' != trim( $_GET['search_runs'] );

if( !$search_experiment && !$search_shifts && !$search_runs )
    report_error( "at least one of (search_experiment, search_shifts, search_runs) parameters must be set" );

if( !isset( $_GET['search_tags'] )) report_error( "no search_tags parameter" );
$search_tags = '0' != trim( $_GET['search_tags'] );

if( !isset( $_GET['search_values'] )) report_error( "no search_values parameter" );
$search_values = '0' != trim( $_GET['search_values'] );

$begin = null;
if( !isset( $_GET['begin'] )) report_error( "no begin parameter" );
$begin_str = trim( $_GET['begin'] );
if( $begin_str != '' ) {
    // Check for shortcuts first
    //
    switch( $begin_str[0] ) {
        case 'm':
        case 'M':
            $begin = LusiTime::minus_month();
            break;
        case 'w':
        case 'W':
            $begin = LusiTime::minus_week();
            break;
        case 'd':
        case 'D':
            $begin = LusiTime::minus_day();
            break;
        case 'y':
        case 'Y':
            $begin = LusiTime::yesterday();
            break;
        case 't':
        case 'T':
            $begin = LusiTime::today();
            break;
        case 'h':
        case 'H':
            $begin = LusiTime::minus_hour();
            break;
    }
    if( is_null( $begin )) {
        $begin = LusiTime::parse( trim( $begin_str ))
            or report_error( "begin time has invalid format" );
    }
}

$end = null;
if( !isset( $_GET['end'] )) report_error( "no end parameter" );
$end_str = trim( $_GET['end'] );
if( $end_str != '' ) {
    $end = LusiTime::parse( trim( $end_str ))
        or report_error( "end time has invalid format" );
}
if( !is_null( $begin ) && !is_null( $end ) && !$begin->less( $end ))
    report_error( "invalid interval - begin time isn't strictly less than the end one" );

if( !isset( $_GET['tag'] )) report_error( "no tag parameter" );
$tag = trim( $_GET['tag'] );

if( !isset( $_GET['author'] )) report_error( "no author parameter" );
$author = trim( $_GET['author'] );

/* Package the error message into a JAON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    print <<< HERE
{
  "ResultSet": {
    "Status": {$status_encoded},
    "Message": {$msg_encoded}
  }
}
HERE;
    exit;
}

/* Translate an entry into a JASON object. Return the serialized object.
 */
function entry2json( $entry ) {

    $relevance_time_str   = is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort();
    $shift_begin_time_str = is_null( $entry->shift_id())       ? 'n/a' : $entry->shift()->begin_time()->toStringShort();
    $run_number_str       = is_null( $entry->run_id())         ? 'n/a' : $entry->run()->num();

    $tags = $entry->tags();
    $attachments = $entry->attachments();

    // Estimate a number of lines for the message text by counting
    // new lines.
    //
    $message_lines = count( explode( "\n", $entry->content()));
    $message_height = min( 200, 8 + 16*$message_lines );
    $base = 10 + $message_height;

    $extra_lines = max( count( $tags ), count( $attachments ));
    $extra_vspace = $extra_lines == 0 ? 0 :  20 + 20 * $extra_lines;

    $con = new RegDBHtml( 0, 0, 750, 75 + $message_height + $extra_vspace );
    $con->container_1 (   0,   0, "<pre style=\"padding:4px; font-size:14px; background-color:#cfecec;\">{$entry->content()}</pre>", 750, $message_height )
        ->label   ( 250,  $base,    'By:'        )->value( 300,  $base,    $entry->author())
        ->label   (  20,  $base,    'Posted:'    )->value( 100,  $base,    $entry->insert_time()->toStringShort())
        ->label   (  20,  $base+20, 'Relevance:' )->value( 100,  $base+20, $relevance_time_str )
        ->label   ( 250,  $base+20, 'Run:'       )->value( 300,  $base+20, $run_number_str )
        ->label   ( 350,  $base+20, 'Shift:'     )->value( 400,  $base+20, $shift_begin_time_str );

    if( count( $tags ) != 0 ) {
        $con->label_1(  20, $base+50, 'Tag', 80 )->label_1( 115, $base+50, 'Value', 100 );
    }
    if( count( $attachments ) != 0 ) {
        $con->label_1  ( 250, $base+50, 'Attachment', 200 )->label_1( 465, $base+50, 'Size', 50 )
            ->container( 520, $base+50, 'viewarea' );
    }
    $base4tags = $base+75;
    foreach( $tags as $tag ) {
        $con->value_1(  20, $base4tags, $tag->tag())
            ->value_1( 115, $base4tags, $tag->value());
        $base4tags = $base4tags + 20;
    }
    $base4attch = $base+75;
    foreach( $attachments as $attachment ) {
        $attachment_url = '<a href="ShowAttachment.php?id='.$attachment->id().'" target="_blank">'.$attachment->description().'</a>';
/*
        $attachment_url = '<a href="javascript:preview_atatchment('.$attachment->id().')">'.$attachment->description().'</a>';
        $attachment_url =
            '<a href="ShowAttachment.php?id='.$attachment->id().'" target="_blank">'.$attachment->description().'</a>'.
            '&nbsp;'.
            '<a href="#viewarea">View</a>';
 *
 */
        $con->value_1( 250, $base4attch, $attachment_url )
            ->value_1( 465, $base4attch, $attachment->document_size());
        $base4attch = $base4attch + 20;
    }
    return json_encode(
        array (
            "event_time" => $entry->insert_time()->toStringShort(),
            "html" => $con->html()
        )
    );
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or report_error( "no such experiment" );

    $entries = $experiment->search(
        $text2search,
        $search_experiment,
        $search_shifts,
        $search_runs,
        $search_tags,
        $search_values,
        $begin,
        $end,
        $tag,
        $author );

    $status_encoded = json_encode( "success" );
    $result =<<< HERE
{
  "ResultSet": {
    "Status": {$status_encoded},
    "Result": [
HERE;
    $first = true;
    foreach( $entries as $e ) {
        if( $first ) {
            $first = false;
            $result .= "\n".entry2json( $e );
        } else {
            $result .= ",\n".entry2json( $e );
        }
    }
    $result .=<<< HERE
 ] } }
HERE;

    print $result;

    $logbook->commit();

} catch( LogBookException $e ) {
    report_error( $e->toHtml());
} catch( RegDBException $e ) {
    report_error( $e->toHtml());
}
?>