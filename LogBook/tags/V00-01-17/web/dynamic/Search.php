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

$inject_runs = false;
if( isset( $_GET['inject_runs'] ))
    $inject_runs = '0' != trim( $_GET['inject_runs'] );

/* This is a special modifier which (if present) is used to return an updated list
 * of messages since (strictly newer than) the specified time.
 * 
 * NOTES:
 * - this parameter will only be respected if it strictly falls into
 *   the [begin,end) interval of the request!
 * - unlike outher time related parameters of the service this one is expected
 *   to be a full precision 64-bit numeric representation of time.
 */
$since_str = '';
if( isset( $_GET['since'] )) {
    $since_str = trim( $_GET['since'] );
}

/* This is a special modifier which (if present) is used to return a shortened list
 * of messages.
 */
$limit = null;  // no limit
if( isset( $_GET['limit'] )) {
    $limit = trim( $_GET['limit'] );
    if( $limit == 'all' ) $limit = null;
}

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
function child2json( $entry ) {

    $timestamp = $entry->insert_time();

    $relevance_time_str = is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort();
    $attachments = $entry->attachments();
    $children = $entry->children();

    $shift_begin_time_str = is_null( $entry->shift_id()) ? '' : "<a href=\"javascript:select_shift(".$entry->shift()->id().")\" class=\"lb_link\">".$entry->shift()->begin_time()->toStringShort().'</a>';
    $run_number_str = '';
    if( !is_null( $entry->run_id())) {
        $run = $entry->run();
        $run_number_str = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>";
    }
    $tag_ids = array();
    $attachment_ids = array();
    if( count( $attachments ) != 0 ) {
        foreach( $attachments as $attachment ) {
            //$attachment_url = '<a href="ShowAttachment.php?id='.$attachment->id().'" target="_blank" class="lb_link">'.$attachment->description().'</a>';
            $attachment_url = '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>';
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
    $children_ids = array();
    foreach( $children as $child )
        array_push( $children_ids, child2json( $child ));

    $content = wordwrap( $entry->content(), 128 );
    return json_encode(
        array (
            "event_timestamp" => $timestamp->to64(),
            "event_time" => $entry->insert_time()->toStringShort(),
            "relevance_time" => $relevance_time_str,
            "run" => $run_number_str,
            "shift" => $shift_begin_time_str,
            "author" => $entry->author(),
            "id" => $entry->id(),
            "subject" => substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' ),
            "html" => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">{$content}</pre>",
            "content" => $entry->content(),
            "attachments" => $attachment_ids,
            "tags" => $tag_ids,
            "children" => $children_ids,
            "is_run" => 0
        )
    );
}

function entry2json( $entry ) {

    $timestamp = $entry->insert_time();
    $event_time_url =  "<a href=\"javascript:display_message({$entry->id()})\" class=\"lb_link\">{$timestamp->toStringShort()}</a>";
    $relevance_time_str = is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort();
    $tags = $entry->tags();
    $attachments = $entry->attachments();
    $children = $entry->children();

    $shift_begin_time_str = is_null( $entry->shift_id()) ? '' : "<a href=\"javascript:select_shift(".$entry->shift()->id().")\" class=\"lb_link\">".$entry->shift()->begin_time()->toStringShort().'</a>';
    $run_number_str = '';
    if( !is_null( $entry->run_id())) {
        $run = $entry->run();
        $run_number_str = "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>";
    }
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
            //$attachment_url = '<a href="ShowAttachment.php?id='.$attachment->id().'" target="_blank" class="lb_link">'.$attachment->description().'</a>';
            $attachment_url = '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>';
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
    $children_ids = array();
    foreach( $children as $child )
        array_push( $children_ids, child2json( $child ));

    $content = wordwrap( $entry->content(), 128 );
    return json_encode(
        array (
            "event_timestamp" => $timestamp->to64(),
            "event_time" => $event_time_url,
            "relevance_time" => $relevance_time_str,
            "run" => $run_number_str,
            "shift" => $shift_begin_time_str,
            "author" => $entry->author(),
            "id" => $entry->id(),
            "subject" => substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' ),
            "html" => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">{$content}</pre>",
            "content" => $entry->content(),
            "attachments" => $attachment_ids,
            "tags" => $tag_ids,
            "children" => $children_ids,
            "is_run" => 0
        )
    );
}

function run2json( $run, $type ) {

    /* TODO: WARNING! Pay attention to the artificial message identifier
     * for runs. an assumption is that normal message entries will
     * outnumber 512 million records.
     */
    $timestamp = $type == 'begin_run' ? $run->begin_time() : $run->end_time();
    $msg       = '<b>'.( $type == 'begin_run' ? 'begin run ' : 'end run ' ).$run->num().'</b>';
    $id        = $type == 'begin_run' ? 512*1024*1024 + $run->id() : 2*512*1024*1024 + $run->id();

    $event_time_url =  "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$timestamp->toStringShort()}</a>";
    $relevance_time_str = $timestamp->toStringShort();

    $shift_begin_time_str = '';
    $run_number_str = '';

    $tag_ids = array();
    $attachment_ids = array();
    $children_ids = array();

    $content = wordwrap( $msg, 128 );
    return json_encode(
        array (
            "event_timestamp" => $timestamp->to64(),
            "event_time" => $event_time_url, //$entry->insert_time()->toStringShort(),
            "relevance_time" => $relevance_time_str,
            "run" => $run_number_str,
            "shift" => $shift_begin_time_str,
            "author" => 'DAQ/RC',
            "id" => $id,
            "subject" => substr( $msg, 0, 72).(strlen( $msg ) > 72 ? '...' : '' ),
            "html" => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">{$content}</pre>",
            "content" => $msg,
            "attachments" => $attachment_ids,
            "tags" => $tag_ids,
            "children" => $children_ids,
            "is_run" => 1
        )
    );
}

/* Truncate the input array of timestamps if the limit has been requested.
 * Note that the extra entries will be removed from the _HEAD_ of the input
 * array. The function will not modify the input array. The truncated array
 * will be returned instead.
 */
function sort_and_truncate_from_head( $timestamps, $limit ) {

    sort( $timestamps );

    /* Return the input array if no limit specified or if the array is smaller
     * than the limit.
     */
    if( !$limit ) return $timestamps;

    $limit_num = (int)$limit;
    if( count( $timestamps ) <= $limit_num ) return $timestamps;

    /* Do need to truncate.
     */
    $idx = 0;
    $first2copy_idx =  count( $timestamps ) - $limit_num;

    $result = array();
    foreach( $timestamps as $t ) {
        if( $idx >= $first2copy_idx ) array_push( $result, $t );
        $idx = $idx + 1; 
    }
    return $result;
}


/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $id )
        or report_error( "no such experiment" );

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        report_error( 'not authorized to read messages for the experiment' );
    }

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

    $since = !$since_str ? null : LusiTime::from64( $since_str );

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
        $since,
        $limit );

    /* Also check for runs.
     */
    $runs = array();

    /* Verify parameters
     */
    if( !is_null( $shift_id ) && !is_null( $run_id ))
        report_error( "conflicting parameters: shift_id=".$shift_id." and run_id=".$run_id );

    /* For explicitly specified shifts and runs force the search limits not
     * to exceed their intervals (if the one is specified).
     */
    $begin4runs = $begin;
    $end4runs   = $end;
    if( !is_null( $shift_id )) {
        $shift = $experiment->find_shift_by_id( $shift_id );
        if( is_null( $shift ))
            report_error( "no shift with shift_id=".$shift_id." found" );

        $begin4runs = $shift->begin_time();
        $end4runs   = $shift->end_time();
    }
    if( !is_null( $run_id )) {
        $run = $experiment->find_run_by_id( $run_id );
        if( is_null( $run ))
            report_error( "no run with run_id=".$run_id." found" );

        $begin4runs = $run->begin_time();
        $end4runs   = $run->end_time();
    }
    
    /* Readjust 'begin' parameter for runs if 'since' is present.
     * Completelly ignore 'since' if it doesn't fall into an interval of
     * the requst.
     */    
    if( !is_null( $since )) {
        $since4runs = $since;
    	if( !is_null( $begin4runs ) && $since->less( $begin4runs )) {
            $since4runs = null;
        }
        if( !is_null( $end4runs ) && $since->greaterOrEqual( $end4runs )) {
            $since4runs = null;
        }
        if( !is_null( $since4runs )) $begin4runs = $since4runs;
    }
    $runs = !$inject_runs ? array() : $experiment->runs_in_interval( $begin4runs, $end4runs, $limit );
    
    /* Mix entries and run records in the right order
     */
    $entries_by_timestamps = array();

    foreach( $entries as $e ) {
        $entries_by_timestamps[$e->insert_time()->to64()] = array( 'type' => 'entry', 'object' => $e );
    }
    foreach( $runs as $r ) {

        /* The following fix helps to avoid duplicating "begin_run" entries because
         * the way we are getting runs (see before) would yeld runs in the interval:
         *
         *   [begin4runs,end4runs)
         */
        if( is_null( $begin4runs ) || $begin4runs->less( $r->begin_time())) {
            $entries_by_timestamps[$r->begin_time()->to64()] = array( 'type' => 'begin_run', 'object' => $r );
        }

        /* This check would prevent start of run entry to be replaced by
         * the end of the previous run wich was automatically closed
         * when starting the next run.
         */
        if( !is_null( $r->end_time())) {
            if( !array_key_exists( $r->end_time()->to64(), $entries_by_timestamps )) {
                $entries_by_timestamps[$r->end_time()->to64()] = array( 'type' => 'end_run', 'object' => $r );
            }
        }
    }
    $timestamps = sort_and_truncate_from_head( array_keys( $entries_by_timestamps ), $limit );

    $status_encoded = json_encode( "success" );
    $result =<<< HERE
{
  "ResultSet": {
    "Status": {$status_encoded},
    "Result": [
HERE;
    $first = true;
    foreach( $timestamps as $t ) {
    	$type  = $entries_by_timestamps[$t]['type'];
    	$entry = $entries_by_timestamps[$t]['object'];
    	if( $type == 'entry' ) {
    		if( $first ) {
                $first = false;
                $result .= "\n".entry2json( $entry );
            } else {
                $result .= ",\n".entry2json( $entry );
            }
    	} else {
    		if( $first ) {
                $first = false;
                $result .= "\n".run2json( $entry, $type );
            } else {
                $result .= ",\n".run2json( $entry, $type );
            }
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
} catch( LusiTimeException $e ) {
    report_error( $e->toHtml());
}
?>

