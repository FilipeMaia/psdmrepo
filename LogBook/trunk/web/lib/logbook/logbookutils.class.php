<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LusiTime\LusiTime;

/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookUtils provides common utilities for e-log Web services
 *
 * @author gapon
 */
class LogBookUtils {

    /**
     * The method accepts a numerioc error code on its input and returns back
     * a text with human readable interpretation (if available) of the code.
     *
     * @param number $errcode
     * @return string
     */
    public static function upload_err2string( $errcode ) {
    
        switch( $errcode ) {
            case UPLOAD_ERR_OK:
                return "There is no error, the file uploaded with success.";
            case UPLOAD_ERR_INI_SIZE:
                return "The uploaded file exceeds the maximum of ".get_ini("upload_max_filesize")." in this Web server configuration.";
            case UPLOAD_ERR_FORM_SIZE:
                return "The uploaded file exceeds the maximum of ".$_POST["MAX_FILE_SIZE"]." that was specified in the sender's HTML form.";
            case UPLOAD_ERR_PARTIAL:
                return "The uploaded file was only partially uploaded.";
            case UPLOAD_ERR_NO_FILE:
                return "No file was uploaded.";
            case UPLOAD_ERR_NO_TMP_DIR:
                return "Missing a temporary folder in this Web server installation.";
            case UPLOAD_ERR_CANT_WRITE:
                return "Failed to write file to disk at this Web server installation.";
            case UPLOAD_ERR_EXTENSION:
                return "A PHP extension stopped the file upload.";
        }
        return "Unknown error code: ".$errorcode;
    }

    /**
     * Translate a child entry into a JSON object. Return the serialized object.
     *
     * @param LogBookFFEntry $entry
     * @param boolean $posted_at_instrument
     * @return string
     */
    public static function child2json( $entry, $posted_at_instrument=false, $inject_deleted_messages=false ) {
    
        $timestamp = $entry->insert_time();
    
        $tag_ids = array();
    
        $attachment_ids = array();
        foreach( $entry->attachments() as $attachment )
            array_push(
                $attachment_ids,
                array(
                    "id"          => $attachment->id(),
                    "type"        => $attachment->document_type(),
                    "size"        => $attachment->document_size(),
                    "description" => $attachment->description(),
                    "url"         => '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>'
                )
            );
    
        $children_ids = array();
        foreach( $entry->children() as $child ) {
            if( $child->deleted() && !$inject_deleted_messages ) continue;
            array_push( $children_ids, LogBookUtils::child2json( $child, $posted_at_instrument, $inject_deleted_messages ));
        }
        $content = wordwrap( $entry->content(), 128 );
        return json_encode(
            array (
                "event_timestamp" => $timestamp->to64(),
                "event_time"      => $entry->insert_time()->toStringShort(),
                "relevance_time"  => is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort(),
                "run"             => '',
                "shift"           => '',
                "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).$entry->author(),
                "id"              => $entry->id(),
                "subject"         => htmlspecialchars( substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' )),
                "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".htmlspecialchars($content)."</pre>",
                "html1"           => "<pre>".htmlspecialchars($content)."</pre>",
                "content"         => htmlspecialchars( $entry->content()),
                "attachments"     => $attachment_ids,
                "tags"            => $tag_ids,
                "children"        => $children_ids,
                "is_run"          => 0,
                "run_id"          => 0,
                "run_num"         => 0,
                "ymd"             => $timestamp->toStringDay(),
                "hms"             => $timestamp->toStringHMS(),
                "deleted"          => $entry->deleted() ? 1 : 0,
                "deleted_by"      => $entry->deleted() ? $entry->deleted_by() : '',
                "deleted_time"      => $entry->deleted() ? $entry->deleted_time()->toStringShort() : ''
            )
        );
    }

    /**
     * Translate an entry into a JSON object. Return the serialized object.
     *
     * @param LogBookFFEntry $entry
     * @param boolean $posted_at_instrument
     * @return unknown_type
     */
    public static function entry2json( $entry, $posted_at_instrument=false, $inject_deleted_messages=false ) {
    
        $timestamp = $entry->insert_time();
        $shift     = is_null( $entry->shift_id()) ? null : $entry->shift();
        $run       = is_null( $entry->run_id())   ? null : $entry->run();
    
        $tag_ids = array();
        foreach( $entry->tags() as $tag )
            array_push(
                $tag_ids,
                array(
                    "tag"   => $tag->tag(),
                    "value" => $tag->value()
                )
            );
    
        $attachment_ids = array();
        foreach( $entry->attachments() as $attachment )
            array_push(
                $attachment_ids,
                array(
                    "id"          => $attachment->id(),
                    "type"        => $attachment->document_type(),
                    "size"        => $attachment->document_size(),
                    "description" => $attachment->description(),
                    "url"         => '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>'
                )
            );
    
        $children_ids = array();
        foreach( $entry->children() as $child ) {
            if( $child->deleted() && !$inject_deleted_messages ) continue;
            array_push( $children_ids, LogBookUtils::child2json( $child, $posted_at_instrument, $inject_deleted_messages ));
        }
    
        $content = wordwrap( $entry->content(), 128 );
        return json_encode(
            array (
                "event_timestamp" => $timestamp->to64(),
                "event_time"      => "<a href=\"index.php?action=select_message&id={$entry->id()}\"  target=\"_blank\" class=\"lb_link\">{$timestamp->toStringShort()}</a>",
                "relevance_time"  => is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort(),
                "run"             => is_null( $run   ) ? '' : "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>",
                "shift"           => is_null( $shift ) ? '' : "<a href=\"javascript:select_shift(".$shift->id().")\" class=\"lb_link\">".$shift->begin_time()->toStringShort().'</a>',
                "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).$entry->author(),
                "id"              => $entry->id(),
                "subject"         => htmlspecialchars( substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' )),
                "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".htmlspecialchars($content)."</pre>",
                "html1"           => "<pre>".htmlspecialchars($content)."</pre>",
                "content"         => htmlspecialchars( $entry->content()),
                "attachments"     => $attachment_ids,
                "tags"            => $tag_ids,
                "children"        => $children_ids,
                "is_run"          => 0,
                "run_id"          => is_null( $run ) ? 0 : $run->id(),
                "run_num"         => is_null( $run ) ? 0 : $run->num(),
                "ymd"             => $timestamp->toStringDay(),
                "hms"             => $timestamp->toStringHMS(),
                "deleted"          => $entry->deleted() ? 1 : 0,
                "deleted_by"      => $entry->deleted() ? $entry->deleted_by() : '',
                "deleted_time"      => $entry->deleted() ? $entry->deleted_time()->toStringShort() : ''
            )
        );
    }

    /**
     * Translate a run entry into a JSON object
     *
     * @param LogBookRun $run
     * @param string $type
     * @param boolean $posted_at_instrument
     * @return unknown_type
     */
    public static function run2json( $run, $type, $posted_at_instrument=false ) {
    
        /* TODO: WARNING! Pay attention to the artificial message identifier
         * for runs. an assumption is that normal message entries will
         * outnumber 512 million records.
         */
        $timestamp = '';
        $msg       = '';
        $id        = '';

        switch( $type ) {
        case 'begin_run':
            $timestamp = $run->begin_time();
            $msg       = '<b>begin run '.$run->num().'</b>';
            $id        = 512*1024*1024 + $run->id();
            break;
        case 'end_run':
            $timestamp = $run->end_time();
            $msg       = '<b>end run '.$run->num().'</b> ( '.LogBookUtils::format_seconds( $run->end_time()->sec - $run->begin_time()->sec ).' )';
            $id        = 2*512*1024*1024 + $run->id();
            break;
        case 'run':
            $timestamp = $run->end_time();
            $msg       = '<b>run '.$run->num().'</b> ( '.LogBookUtils::format_seconds( $run->end_time()->sec - $run->begin_time()->sec ).' )';
            $id        = 3*512*1024*1024 + $run->id();
            break;
        }
    
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
                "author" => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).'DAQ/RC',
                "id" => $id,
                "subject" => substr( $msg, 0, 72).(strlen( $msg ) > 72 ? '...' : '' ),
                "html" => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".$content."</pre>",
                "html1" => $content,
                "content" => $msg,
                "attachments" => $attachment_ids,
                "tags" => $tag_ids,
                "children" => $children_ids,
                "is_run" => 1,
                "run_id" => $run->id(),
                "begin_run" => $run->begin_time()->toStringShort(),
                "end_run" => is_null($run->end_time()) ? '' : $run->end_time()->toStringShort(),
                "run_num" => $run->num(),
                "ymd" => $timestamp->toStringDay(),
                "hms" => $timestamp->toStringHMS()
            )
        );
    }

    /**
     * Translate the number of seconds into a string of the following format:
     *
     *    'M min S sec'
     *
     * where:
     * 
     *    M - a number from 0 and higher
     *    S - a number from 0 through 59
     *
     * @param number $sec
     * @return unknown_type
     */
    private static function format_seconds( $sec ) {
        if( $sec < 60 ) return $sec.' sec';
        return floor( $sec / 60 ).' min '.( $sec % 60 ).' sec ';
    }
        

    public static function search_around($message_id, $report_error) {

        $entry = LogBook::instance()->find_entry_by_id( $message_id ) or $report_error( "no such message entry" );
        $id = $entry->exper_id();
        $shift_id = null;
        $run_id = null;
        if( !is_null( $shift_id ) && !is_null( $run_id )) $report_error( "conflicting parameters found in the request: <b>shift_id</b> and <b>run_id</b>" );
        $text2search = '';
        $search_in_messages = true;
        $search_in_tags = false;
        $search_in_values = false;
        if( !$search_in_messages && !$search_in_tags && !$search_in_values ) $report_error( "at least one of (<b>search_in_messages</b>, <b>search_in_tags</b>, <b>search_in_values</b>) parameters must be set" );
        $posted_at_instrument = false;
        $posted_at_experiment = true;
        $posted_at_shifts = true;
        $posted_at_runs = true;
        if( !$posted_at_experiment && !$posted_at_shifts && !$posted_at_runs ) $report_error( "at least one of (<b>posted_at_experiment</b>, <b>posted_at_shifts</b>, <b>posted_at_runs</b>) parameters must be set" );
        $begin = null;
        $end = null;
        $tag = '';
        $author = '';
        $inject_runs = true;
        $inject_deleted_messages = true;
        $since = null;
        $limit = null;

        /* The functon will produce a sorted list of timestamps based on keys of
         * the input dictionary. If two consequitive begin/end run records are found
         * for the same run then the records will be collapsed into a single record 'run'
         * with the begin run timestamp. The list may also be truncated if the limit has
         * been requested. In that case excessive entries will be removed from the _HEAD_
         * of the input array.
         * 
         * NOTE: The contents of input array will be modified for collapsed runs
         *       by replacing types for 'begin_run' / 'end_run' with just 'run'.
         */
        function sort_and_truncate_from_head( &$entries_by_timestamps, $limit, $report_error ) {

            $all_timestamps = array_keys( $entries_by_timestamps );
            sort( $all_timestamps );

            /* First check if we need to collapse here anything.
             * 
             * TODO: !!!
             */
            $timestamps = array();
            $prev_begin_run = null;
            foreach( $all_timestamps as $t ) {
                foreach( $entries_by_timestamps[$t] as $pair ) {
                       $entry = $pair['object'];
                    switch( $pair['type'] ) {
                    case 'entry':
                        $prev_begin_run = null;
                        array_push( $timestamps, $t );
                        break;
                    case 'begin_run':
                        $prev_begin_run = $t;
                        array_push( $timestamps, $t );
                        break;
                    case 'end_run':
                        if( is_null( $prev_begin_run )) {
                            array_push( $timestamps, $t );
                        } else {
                            foreach( array_keys( $entries_by_timestamps[$prev_begin_run] ) as $k ) {
                                if( $entries_by_timestamps[$prev_begin_run][$k]['type'] == 'begin_run' ) {
                                    $entries_by_timestamps[$prev_begin_run][$k]['type'] = 'run';
                                    $prev_begin_run = null;
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            // Remove duplicates (if any). They may show up if an element of
            // $entries_by_timestamps will have more than one entry.
            //
            $timestamps = array_unique( $timestamps );

            /* Do need to truncate. Apply different limiting techniques depending
             * on a value of the parameter.
             */
            if( !$limit ) return $timestamps;

            $result = array();

            $limit_num = null;
            $unit = null;
            if( 2 == sscanf( $limit, "%d%s", $limit_num, $unit )) {

                $nsec_ago = 1000000000 * $limit_num;
                switch( $unit ) {
                    case 's': break;
                    case 'm': $nsec_ago *=            60; break;
                    case 'h': $nsec_ago *=          3600; break;
                    case 'd': $nsec_ago *=     24 * 3600; break;
                    case 'w': $nsec_ago *= 7 * 24 * 3600; break;
                    default:
                        $report_error( "illegal format of the limit parameter" );
                }
                $now_nsec = LusiTime::now()->to64();
                foreach( $timestamps as $t ) {
                    if( $t >= ( $now_nsec - $nsec_ago )) array_push( $result, $t );
                }

            } else {

                $limit_num = (int)$limit;

                /* Return the input array if no limit specified or if the array is smaller
                 * than the limit.
                 */
                if( count( $timestamps ) <= $limit_num ) return $timestamps;

                $idx = 0;
                $first2copy_idx =  count( $timestamps ) - $limit_num;

                foreach( $timestamps as $t ) {
                    if( $idx >= $first2copy_idx ) array_push( $result, $t );
                    $idx = $idx + 1; 
                }
            }
            return $result;
        }

        /* Make adjustments relative to the primary experiment of the search.
         */
        $experiment = LogBook::instance()->find_experiment_by_id( $id );
        if( is_null( $experiment)) $report_error( "no such experiment" );

        /* Mix entries and run records in the right order. Results will be merged
         * into this dictionary before returning to the client.
         */
        $entries_by_timestamps = array();

        /* Scan all relevant experiments. Normally it would be just one. However, if
         * the instrument is selected then all experiments of the given instrument will
         * be taken into consideration.
         */
        $experiments = array();
        if( $posted_at_instrument ) {
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
                if( $posted_at_instrument && ( $e->id() != $id )) continue;

                $report_error( 'not authorized to read messages for the experiment' );
            }

            /* Get the info for entries and (if requested) for runs.
             * 
             * NOTE: If the full text search is involved then the search will
             * propagate down to children subtrees as well. However, the resulting
             * list of entries will only contain the top-level ("thread") messages.
             * To ensure so we're going to pre-scan the result of the query to identify
             * children and finding their top-level parents. The parents will be put into
             * the result array. Also note that we're not bothering about having duplicate
             * entries in the array becase this will be sorted out on the next step.
             */
            $entries = array();
            foreach(
                $e->search(
                    $e->id() == $id ? $shift_id : null,    // the parameter makes sense for the main experiment only
                    $e->id() == $id ? $run_id   : null,    // ditto
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
                    null, /* $limit */
                    $inject_deleted_messages,
                    $search_in_messages && ( $text2search != '' )   // include children into the search for
                                                                    // the full-text search in message bodies.
                )
                as $entry ) {
                    $parent = $entry->parent_entry();
                    if( is_null($parent)) {
                        array_push ($entries, $entry);
                    } else {
                        while(true ) {
                            $parent_of_parent = $parent->parent_entry();
                            if( is_null($parent_of_parent)) break;
                            $parent = $parent_of_parent;
                        }
                        array_push ($entries, $parent);
                    }
            }

            $runs = !$inject_runs ? array() : $e->runs_in_interval( $begin, $end );

            /* Merge both results into the dictionary for further processing.
             */
            foreach( $entries as $e ) {
                $t = $e->insert_time()->to64();
                if( !array_key_exists( $t, $entries_by_timestamps )) $entries_by_timestamps[$t] = array();
                array_push(
                    $entries_by_timestamps[$t],
                    array(
                        'type'   => 'entry',
                        'object' => $e
                    )
                );
            }
            foreach( $runs as $r ) {

                /* The following fix helps to avoid duplicating "begin_run" entries because
                 * the way we are getting runs (see before) would yeld runs in the interval:
                 *
                 *   [begin4runs,end4runs)
                 */
                if( is_null( $begin ) || $begin->less( $r->begin_time())) {
                    $t = $r->begin_time()->to64();
                    if( !array_key_exists( $t, $entries_by_timestamps )) $entries_by_timestamps[$t] = array();
                    array_push(
                        $entries_by_timestamps[$t],
                        array(
                            'type'   => 'begin_run',
                            'object' => $r
                        )
                    );
                }

                if( !is_null( $r->end_time())) {
                    $t = $r->end_time()->to64();
                    if( !array_key_exists( $t, $entries_by_timestamps )) $entries_by_timestamps[$t] = array();
                    array_push(
                        $entries_by_timestamps[$t],
                        array(
                            'type'   => 'end_run',
                            'object' => $r
                        )
                    );
                }
            }
        }

        $timestamps = sort_and_truncate_from_head( $entries_by_timestamps, $limit, $report_error );

        $status_encoded  = json_encode( "success" );
        $updated_encoded = json_encode( LusiTime::now()->toStringShort());

        $result =<<< HERE
{
  "ResultSet": {
    "Status": {$status_encoded},
    "Updated": {$updated_encoded},
    "Result": [
HERE;
        $first = true;
        foreach( $timestamps as $t ) {
            foreach( $entries_by_timestamps[$t] as $pair ) {
                $type  = $pair['type'];
                $entry = $pair['object'];
                if( $type == 'entry' ) {
                    if( $first ) {
                        $first = false;
                        $result .= "\n".LogBookUtils::entry2json( $entry, $posted_at_instrument, $inject_deleted_messages );
                    } else {
                        $result .= ",\n".LogBookUtils::entry2json( $entry, $posted_at_instrument, $inject_deleted_messages );
                    }
                } else {
                    if( $first ) {
                        $first = false;
                        $result .= "\n".LogBookUtils::run2json( $entry, $type, $posted_at_instrument );
                    } else {
                        $result .= ",\n".LogBookUtils::run2json( $entry, $type, $posted_at_instrument );
                    }
                }
            }
        }
        $result .=<<< HERE
 ] } }
HERE;
        return $result;
    }
}
?>
