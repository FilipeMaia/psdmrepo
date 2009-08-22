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

$shift_id = null;
if( isset( $_GET['shift_id'] )) {
    $shift_id = trim( $_GET['shift_id'] );
    if( $shift_id == '' ) report_error( "shift identifier parameter can't be empty" );
}

$run_id = null;
if( isset( $_GET['run_id'] )) {
    $run_id = trim( $_GET['run_id'] );
    if( $run_id == '' ) report_error( "run identifier parameter can't be empty" );
}

if( !is_null( $shift_id ) && !is_null( $run_id ))
    report_error( "conflicting parameters found in the request: <b>shift_id</b> and <b>run_id</b>" );

if( !isset( $_GET['format'] )) report_error( "no valid presentation format parameter" );
$format = trim( $_GET['format'] );
if( $format == '' ) report_error( "presentation format parameter can't be empty" );

$text2search = '';
if( isset( $_GET['text2search'] ))
    $text2search = trim( $_GET['text2search'] );


$search_in_messages = false;
if( isset( $_GET['search_in_messages'] ))
    $search_in_messages = '0' != trim( $_GET['search_in_messages'] );

$search_in_tags = false;
if( isset( $_GET['search_in_tags'] ))
    $search_in_tags = '0' != trim( $_GET['search_in_tags'] );

$search_in_values = false;
if( isset( $_GET['search_in_values'] ))
    $search_in_values = '0' != trim( $_GET['search_in_values'] );

if( !$search_in_messages && !$search_in_tags && !$search_in_values )
    report_error( "at least one of (<b>search_in_messages</b>, <b>search_in_tags</b>, <b>search_in_values</b>) parameters must be set" );


$posted_at_experiment = false;
if( isset( $_GET['posted_at_experiment'] ))
    $posted_at_experiment = '0' != trim( $_GET['posted_at_experiment'] );

$posted_at_shifts = false;
if( isset( $_GET['posted_at_shifts'] ))
    $posted_at_shifts = '0' != trim( $_GET['posted_at_shifts'] );

$posted_at_runs = false;
if( isset( $_GET['posted_at_runs'] ))
    $posted_at_runs = '0' != trim( $_GET['posted_at_runs'] );

if( !$posted_at_experiment && !$posted_at_shifts && !$posted_at_runs )
    report_error( "at least one of (<b>posted_at_experiment</b>, <b>posted_at_shifts</b>, <b>posted_at_runs</b>) parameters must be set" );

$begin_str = '';
if( isset( $_GET['begin'] ))
    $begin_str = trim( $_GET['begin'] );

$end_str = '';
if( isset( $_GET['end'] ))
    $end_str = trim( $_GET['end'] );

$tag = '';
if( isset( $_GET['tag'] ))
    $tag = trim( $_GET['tag'] );

$author = '';
if( isset( $_GET['author'] ))
    $author = trim( $_GET['author'] );

/* This is a special modifier which (if present) is used to return an updated list
 * of messages since (strictly newer than) the specified time.
 * 
 * ATTENTION: This parameter will only be respected if it strictly falls into
 * the [begin,end) interval of the request!
 */
$since_str = '';
if( isset( $_GET['since'] ))
    $since_str = trim( $_GET['since'] );


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

/* Translate timestamps which may also contain shortcuts
 */
function translate_time( $experiment, $str ) {
    $str_trimmed = trim( $str );
    if( $str_trimmed == '' ) return null;
    switch( $str_trimmed[0] ) {
        case 'b':
        case 'B': return $experiment->begin_time();
        case 'e':
        case 'E': return $experiment->end_time();
        case 'm':
        case 'M': return LusiTime::minus_month();
        case 'w':
        case 'W': return LusiTime::minus_week();
        case 'd':
        case 'D': return LusiTime::minus_day();
        case 'y':
        case 'Y': return LusiTime::yesterday();
        case 't':
        case 'T': return LusiTime::today();
        case 'h':
        case 'H': return LusiTime::minus_hour();
    }
    $result = LusiTime::parse( $str_trimmed );
    if( is_null( $result )) $result = LusiTime::from64( $str_trimmed );
    return $result;
}

/* Translate an entry into a JASON object. Return the serialized object.
 */
function entry2json( $entry, $format ) {

    $relevance_time_str = is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort();
    $tags = $entry->tags();
    $attachments = $entry->attachments();

    // Produce different output depending on the requested format.
    //
    if( $format == 'detailed' ) {

        $shift_begin_time_str = is_null( $entry->shift_id()) ? '' : "<a href=\"javascript:select_shift(".$entry->shift()->id().")\" class=\"lb_link\">".$entry->shift()->begin_time()->toStringShort().'</a>';
        $run_number_str = '';
        if( !is_null( $entry->run_id())) {
            $run = $entry->run();
            $run_number_str = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>";
        }

        // Estimate a number of lines for the message text by counting
        // new lines.
        //
        $message_lines = count( explode( "\n", $entry->content()));
        $message_height = min( 200, 14 + 14*$message_lines );

        $con = new RegDBHtml( 0, 0, 800, $message_height );

        $highlight = false;
        $con->container_1 (   0,   0, "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">{$entry->content()}</pre>", 800, $message_height, $highlight );

        $tag_ids = array();
        if( count( $tags ) != 0 ) {
            foreach( $tags as $tag ) {
                array_push(
                    $tag_ids,
                    array(
                        "tag" => $tag->tag(),
                        "value" => $tag->value()
                    )
                );
            }
        }

        $attachment_ids = array();
        if( count( $attachments ) != 0 ) {
            foreach( $attachments as $attachment ) {
                $attachment_url = '<a href="ShowAttachment.php?id='.$attachment->id().'" target="_blank" class="lb_link">'.$attachment->description().'</a>';
                array_push(
                    $attachment_ids,
                    array(
                        "id" => $attachment->id(),
                        "type" => $attachment->document_type(),
                        "size" => $attachment->document_size(),
                        "url" => $attachment_url
                    )
                );
            }
        }
        return json_encode(
            array (
                "event_time" => $entry->insert_time()->toStringShort(),
                "relevance_time" => $relevance_time_str,
                "run" => $run_number_str,
                "shift" => $shift_begin_time_str,
                "author" => $entry->author(),
                "id" => $entry->id(),
                "subject" => substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' ),
                //"html" => $con->html(),
                "html" => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">{$entry->content()}</pre>",
                "content" => $entry->content(),
                "attachments" => $attachment_ids,
                "tags" => $tag_ids
            )
        );
    }
    return null;
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or report_error( "no such experiment" );

    // Timestamps are translated here because of possible shoftcuts which
    // may reffer to the experiment's validity limits.
    //
    $begin = null;
    if( $begin_str != '' ) {
        $begin = translate_time( $experiment, $begin_str );
        if( is_null( $begin ))
            report_error( "begin time has invalid format" );
    }
    $end = null;
    if( $end_str != '' ) {
        $end = translate_time( $experiment, $end_str );
        if( is_null( $end ))
            report_error( "end time has invalid format" );
    }
    if( !is_null( $begin ) && !is_null( $end ) && !$begin->less( $end ))
        report_error( "invalid interval - begin time isn't strictly less than the end one" );

    $since = null;
    if( $since_str != '' ) {
        $since = translate_time( $experiment, $since_str );
        if( is_null( $since ))
            report_error( "'since' time has invalid format" );
    }

    $entries = $experiment->search(
        $shift_id, $run_id,
        $text2search,
        $search_in_messages,
        $search_in_tags,
        $search_in_values,
        $posted_at_experiment,
        $posted_at_shifts,
        $posted_at_runs,
        $begin,
        $end,
        $tag,
        $author,
        $since );

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
            $result .= "\n".entry2json( $e, $format );
        } else {
            $result .= ",\n".entry2json( $e, $format );
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