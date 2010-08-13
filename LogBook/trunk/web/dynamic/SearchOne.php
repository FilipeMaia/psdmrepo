<?php

require_once('LogBook/LogBook.inc.php');


/*
 * This script will perform the search for a single free-form entry in a scope
 * of an experiment using a numer identifier of the entry. The result is returned
 * as a JSON obejct which in case of success will have the following format:
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

if( !isset( $_GET['id'] )) report_error( "no valid message id parameter" );
$id = trim( $_GET['id'] );
if( $id == '' ) report_error( "message id can't be empty" );


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


/* Translate an entry into a JSON object. Return the serialized object.
 */
function child2json( $entry ) {

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
            "children" => $children_ids
        )
    );
}

function entry2json( $entry ) {

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
            "children" => $children_ids
        )
    );
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $entry = $logbook->find_entry_by_id( $id )
        or report_error( "no such message entry" );

    $experiment = $entry->parent();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        report_error( 'not authorized to read messages for the experiment' );
    }


    $status_encoded = json_encode( "success" );
    $result =<<< HERE
{
  "ResultSet": {
    "Status": {$status_encoded},
    "Result": [
HERE;
    $result .= "\n".entry2json( $entry );
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