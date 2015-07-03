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
     * @brief Translate a child entry into an array
     *
     * @param LogBookFFEntry $entry
     * @param boolean $posted_at_instrument
     * @return string
     */
    public static function child2array( $entry, $posted_at_instrument=false, $inject_deleted_messages=false ) {
    
        $timestamp = $entry->insert_time();
    
        $tags_num = 0;
        $tag_ids  = array();
    
        $attachments_num = 0 ;
        $attachment_ids  = array();
        foreach( $entry->attachments() as $attachment ) {
            $attachments_num++;
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
        }
        $children_num = 0;
        $children_ids = array();
        foreach( $entry->children() as $child ) {
            if( $child->deleted() && !$inject_deleted_messages ) continue;
            $children_num++;
            array_push( $children_ids, LogBookUtils::child2array( $child, $posted_at_instrument, $inject_deleted_messages ));
        }
        $content = wordwrap( $entry->content(), 128 );
        return array (
            "event_timestamp" => $timestamp->to64(),
            "event_time"      => $entry->insert_time()->toStringShort(),
            "relevance_time"  => is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort(),
            "run"             => '',
            "shift"           => '',
            "shift_id"        => 0,
            "shift_begin"     => 
                array (
                    'ymd'  => '',
                    'hms'  => '',
                    'time' => ''),
            "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).$entry->author(),
            "id"              => $entry->id(),
            "parent_id"       => $entry->parent_entry_id() ,
            "subject"         => htmlspecialchars( substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' )),
            "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".htmlspecialchars($content)."</pre>",
            "html1"           => "<pre>".htmlspecialchars($content)."</pre>",
            "content"         => htmlspecialchars( $entry->content()),
            "attachments_num" => $attachments_num,
            "attachments"     => $attachment_ids,
            "tags_num"        => $tags_num,
            "tags"            => $tag_ids,
            "children_num"    => $children_num,
            "children"        => $children_ids,
            "is_run"          => 0,
            "run_id"          => 0,
            "run_num"         => 0,
            "ymd"             => $timestamp->toStringDay(),
            "hms"             => $timestamp->toStringHMS(),
            "deleted"         => $entry->deleted() ? 1 : 0,
            "deleted_by"      => $entry->deleted() ? $entry->deleted_by() : '',
            "deleted_time"    => $entry->deleted() ? $entry->deleted_time()->toStringShort() : ''
        );
    }
    public static function child2json( $entry, $posted_at_instrument=false, $inject_deleted_messages=false ) {
        return json_encode(LogBookUtils::child2array($entry, $posted_at_instrument, $inject_deleted_messages)) ;
    }

    /**
     * @brief Translate an entry into an array
     *
     * @param LogBookFFEntry $entry
     * @param boolean $posted_at_instrument
     * @return unknown_type
     */
    public static function entry2array( $entry, $posted_at_instrument=false, $inject_deleted_messages=false ) {
    
        $timestamp = $entry->insert_time();
        $shift     = is_null( $entry->shift_id()) ? null : $entry->shift();
        $run       = is_null( $entry->run_id())   ? null : $entry->run();
    
        $tags_num = 0;
        $tag_ids  = array();
        foreach( $entry->tags() as $tag ) {
            $tags_num++;
            array_push(
                $tag_ids,
                array(
                    "tag"   => $tag->tag(),
                    "value" => $tag->value()
                )
            );
        }
    
        $attachments_num = 0 ;
        $attachment_ids  = array();
        foreach( $entry->attachments() as $attachment ) {
            $attachments_num++;
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
        }
        $children_num = 0;
        $children_ids = array();
        foreach( $entry->children() as $child ) {
            if( $child->deleted() && !$inject_deleted_messages ) continue;
            $children_num++;
            array_push( $children_ids, LogBookUtils::child2array( $child, $posted_at_instrument, $inject_deleted_messages ));
        }
    
        $content = wordwrap( $entry->content(), 128 );
        return array (
            "event_timestamp" => $timestamp->to64(),
            "event_time"      => "<a href=\"index.php?action=select_message&id={$entry->id()}\"  target=\"_blank\" class=\"lb_link\">{$timestamp->toStringShort()}</a>",
            "relevance_time"  => is_null( $entry->relevance_time()) ? 'n/a' : $entry->relevance_time()->toStringShort(),
            "run"             => is_null( $run   ) ? '' : "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$run->num()}</a>",
            "shift"           => is_null( $shift ) ? '' : "<a href=\"javascript:select_shift(".$shift->id().")\" class=\"lb_link\">".$shift->begin_time()->toStringShort().'</a>',
            "shift_id"        => is_null( $shift ) ? 0  : $shift->id() ,
            "shift_begin"     => is_null( $shift ) ?
                array (
                    'ymd'  => '',
                    'hms'  => '',
                    'time' => '') :
                array (
                    'ymd'  => $shift->begin_time()->toStringDay(),
                    'hms'  => $shift->begin_time()->toStringHMS(),
                    'time' => $shift->begin_time()->toStringShort()),
            "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).$entry->author(),
            "id"              => $entry->id(),
            "parent_id"       => 0 ,
            "subject"         => htmlspecialchars( substr( $entry->content(), 0, 72).(strlen( $entry->content()) > 72 ? '...' : '' )),
            "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".htmlspecialchars($content)."</pre>",
            "html1"           => "<pre>".htmlspecialchars($content)."</pre>",
            "content"         => htmlspecialchars( $entry->content()),
            "attachments_num" => $attachments_num,
            "attachments"     => $attachment_ids,
            "tags_num"        => $tags_num,
            "tags"            => $tag_ids,
            "children_num"    => $children_num,
            "children"        => $children_ids,
            "is_run"          => 0,
            "run_id"          => is_null( $run ) ? 0 : $run->id(),
            "run_num"         => is_null( $run ) ? 0 : $run->num(),
            "ymd"             => $timestamp->toStringDay(),
            "hms"             => $timestamp->toStringHMS(),
            "deleted"         => $entry->deleted() ? 1 : 0,
            "deleted_by"      => $entry->deleted() ? $entry->deleted_by() : '',
            "deleted_time"    => $entry->deleted() ? $entry->deleted_time()->toStringShort() : ''
        );
    }
    public static function entry2json( $entry, $posted_at_instrument=false, $inject_deleted_messages=false ) {
        return json_encode(LogBookUtils::entry2array($entry, $posted_at_instrument, $inject_deleted_messages)) ;
    }

    /**
     * @brief Translate a run entry into an array
     *
     * @param LogBookRun $run
     * @param string $type
     * @param boolean $posted_at_instrument
     * @return unknown_type
     */
    public static function run2array( $run, $type, $posted_at_instrument=false ) {
    
        /* TODO: WARNING! Pay attention to the artificial message identifier
         * for runs. an assumption is that normal message entries will
         * outnumber 512 million records.
         */
        $duration  = $type === 'begin_run' ? '' : LogBookUtils::format_seconds  ( $run->end_time()->sec - $run->begin_time()->sec );
        $duration1 = $type === 'begin_run' ? '' : LogBookUtils::format_seconds_1( $run->end_time()->sec - $run->begin_time()->sec );
        $timestamp = '';
        $msg       = '';
        $id        = '';

        switch( $type ) {
        case 'begin_run':
            $timestamp = $run->begin_time();
            $msg       = "<b>begin run {$run->num()}</b>";
            $id        = 512*1024*1024 + $run->id();
            break;
        case 'end_run':
            $timestamp = $run->end_time();
            $msg       = "<b>end run {$run->num()}</b> ( {$duration} )";
            $id        = 2*512*1024*1024 + $run->id();
            break;
        case 'run':
            $timestamp = $run->end_time();
            $msg       = "<b>run {$run->num()}</b> ( {$duration} )";
            $id        = 3*512*1024*1024 + $run->id();
            break;
        }
    
        $event_time_url =  "<a href=\"javascript:select_run({$run->shift()->id()},{$run->id()})\" class=\"lb_link\">{$timestamp->toStringShort()}</a>";
        $relevance_time_str = $timestamp->toStringShort();
    
        $shift = $run->shift();
        $run_number_str = '';
    
        $tag_ids = array();
        $attachment_ids = array();
        $children_ids = array();
    
        $content = wordwrap( $msg, 128 );
        return json_encode(
            array (
                "event_timestamp" => $timestamp->to64(),
                "event_time"      => $event_time_url, //$entry->insert_time()->toStringShort(),
                "relevance_time"  => $relevance_time_str,
                "run"             => $run_number_str,
                "shift"           => is_null( $shift ) ? '' : "<a href=\"javascript:select_shift(".$shift->id().")\" class=\"lb_link\">".$shift->begin_time()->toStringShort().'</a>',
                "shift_id"        => is_null( $shift ) ? 0  : $shift->id() ,
                "shift_begin"     =>
                    array (
                        'ymd'  => $shift->begin_time()->toStringDay(),
                        'hms'  => $shift->begin_time()->toStringHMS(),
                        'time' => $shift->begin_time()->toStringShort()),
                "author"          => ( $posted_at_instrument ? $entry->parent()->name().'&nbsp;-&nbsp;' : '' ).'DAQ/RC',
                "id"              => $id,
                "parent_id"       => 0 ,
                "subject"         => substr( $msg, 0, 72).(strlen( $msg ) > 72 ? '...' : '' ),
                "html"            => "<pre style=\"padding:4px; padding-left:8px; font-size:14px; border: solid 2px #efefef;\">".$content."</pre>",
                "html1"           => $content,
                "content"         => $msg,
                "attachments"     => $attachment_ids,
                "tags"            => $tag_ids,
                "children"        => $children_ids,
                "is_run"          => 1,
                "run_id"          => $run->id(),
                "type"            => $type,
                "begin_run"       => $run->begin_time()->toStringShort(),
                "end_run"         => is_null($run->end_time()) ? '' : $run->end_time()->toStringShort(),
                "run_num"         => $run->num(),
                "duration"        => $duration,
                "duration1"       => $duration1,
                "ymd"             => $timestamp->toStringDay(),
                "hms"             => $timestamp->toStringHMS()
            )
        );
    }
    public static function run2json( $run, $type, $posted_at_instrument=false ) {
        return json_encode(LogBookUtils::run2array($run, $type, $posted_at_instrument)) ;
    }

    public static function shift2array ($s, $extended_format=true) {

        if ($extended_format) {
    
            $total_seconds = $s->end_time() ?
                $s->end_time()->sec  - $s->begin_time()->sec :
                LusiTime::now()->sec - $s->begin_time()->sec ;

            // Exclude the current shift from consideration when calculating the maximum
            // duration of a shift

            $durat = '' ;
            $durat_days = '' ;
            $durat_hms = '' ;

            if ($total_seconds) {
                $seconds_left = $total_seconds ;

                $day          = floor($seconds_left / (24 * 3600)) ;
                $seconds_left = $seconds_left % (24 * 3600) ;

                $hour         = floor($seconds_left / 3600) ;
                $seconds_left = $seconds_left % 3600 ;

                $min          = floor($seconds_left / 60) ;
                $seconds_left = $seconds_left % 60 ;

                $sec          = $seconds_left ;
                $durat = sprintf("%03d days, %02d:%02d.%02d", $day, $hour, $min, $sec) ;
                $durat_days = $day ;
                $durat_hms  = LogBookUtils::format_seconds_2(3600 * $hour + 60 * $min + $sec) ;
            }

            // See if the 'Goals' record is found among messages
            //
            $entries = $s->parent()->search (
                $s->id() ,      // $shift_id=
                null ,          // $run_id=
                '' ,            // $text2search
                false ,         // $search_in_messages=
                true ,          // $search_in_tags=
                false ,         // $search_in_values=
                false ,         // $posted_at_experiment=
                true ,          // $posted_at_shifts=
                false ,         // $posted_at_runs=
                null ,          // $begin=
                null ,          // $end=
                'SHIFT_GOALS' , // $tag=
                '' ,            // $author=
                null            // $since=
            ) ;
            
            $goals = '' ;
            foreach ($entries as $e)
                $goals .= $e->content() ;

            $runs = array() ;
            foreach ($s->runs() as $r) {
                array_push (
                       $runs ,
                       array (
                           'id'  => $r->id() ,
                           'num' => $r->num()
                       )
                   ) ;
            }
            return array (
                'id'      => $s->id() ,
                'is_open' => is_null($s->end_time()) ? 1 : 0 ,
                
                'begin'     => $s->begin_time()->toStringShort() ,
                'begin_ymd' => $s->begin_time()->toStringDay() ,
                'begin_hms' => $s->begin_time()->toStringHMS() ,
                'begin_sec' => $s->begin_time()->sec,

                'end'       => is_null($s->end_time()) ? '<span style="color:red; font-weight:bold;">on-going</span>' : $s->end_time()->toStringShort() ,
                'end_ymd'   => is_null($s->end_time()) ? '<span style="color:red; font-weight:bold;">on-going</span>' : $s->end_time()->toStringDay() ,
                'end_hms'   => is_null($s->end_time()) ? ''                                                           : $s->end_time()->toStringHMS() ,
                'end_sec'   => is_null($s->end_time()) ? 0                                                            : $s->end_time()->sec ,

                'durat'      => $durat ,
                'durat_days' => $durat_days ,
                'durat_hms'  => $durat_hms ,
                'sec'        => $total_seconds ,

                'leader' => $s->leader() ,
                'crew'   => $s->crew() ,
                'goals'  => $goals ,

                'num_runs' => $s->num_runs() ,
                'runs'     => $runs
            ) ;

        } else {

            $begin_time_url =
                "<a href=\"javascript:select_shift(".$s->id().")\" class=\"lb_link\">" .
                $s->begin_time()->toStringShort() .
                '</a>' ;

            $end_time_status =
                is_null($s->end_time()) ?
                '<b><em style="color:red;">on-going</em></b>' :
                $s->end_time()->toStringShort() ;

            return array (
                "id"      => $s->id() ,
                'is_open' => is_null($s->end_time()) ? 1 : 0 ,

                "begin_time" => $begin_time_url ,
                "end_time"   => $end_time_status ,

                "leader"   => $s->leader() ,
                "num_runs" => $s->num_runs()
            ) ;
        }
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
    public static function format_seconds ($sec) {
        if( $sec < 60 ) return $sec.' sec';
        return floor( $sec / 60 ).' min '.( $sec % 60 ).' sec ';
    }
    public static function format_seconds_1 ($sec) {

        if     ( $sec <   60 ) return sprintf("%2d", $sec);
        else if( $sec < 3600 ) return sprintf("%d:%02d", floor($sec / 60), $sec % 60);

        $h = floor($sec / 3600);
        $sec_minus_hours = $sec - $h * 3600;
        $m = floor($sec_minus_hours / 60);
        $s = $sec_minus_hours % 60;

        return $h > 9 ? "** ** **" : sprintf("%d:%02d:%02d", $h, $m, $s);
    }

    public static function format_seconds_2 ($sec) {
        if     ( $sec <   60 ) return sprintf("%2d", $sec);
        else if( $sec < 3600 ) return sprintf("%d:%02d", floor($sec / 60), $sec % 60);

        $h = floor($sec / 3600);
        $sec_minus_hours = $sec - $h * 3600;
        $m = floor($sec_minus_hours / 60);
        $s = $sec_minus_hours % 60;

        return sprintf("%d:%02d:%02d", $h, $m, $s);
    }
    public static function search_around_message ($message_id, $report_error) {
        $entry = LogBook::instance()->find_entry_by_id($message_id) or $report_error("no such message entry") ;
        return LogBookUtils::search_around($entry->exper_id(), $report_error) ;
    }
    public static function search_around_run ($run_id, $report_error) {
        $run = LogBook::instance()->find_run_by_id($run_id) or $report_error("no such run entry") ;
        return LogBookUtils::search_around($run->exper_id(), $report_error) ;
    }
    public static function search_around ($exper_id, $report_error) {

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
        $experiment = LogBook::instance()->find_experiment_by_id( $exper_id );
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
                if( $posted_at_instrument && ( $e->id() != $exper_id )) continue;

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
                    $e->id() == $exper_id ? $shift_id : null,    // the parameter makes sense for the main experiment only
                    $e->id() == $exper_id ? $run_id   : null,    // ditto
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

        $result = array() ;

        foreach( sort_and_truncate_from_head( $entries_by_timestamps, $limit, $report_error ) as $t ) {
            foreach( $entries_by_timestamps[$t] as $pair ) {
                $type  = $pair['type'];
                $entry = $pair['object'];
                array_push (
                    $result ,
                    $type == 'entry' ?
                        LogBookUtils::entry2array( $entry,        $posted_at_instrument, $inject_deleted_messages ) :
                        LogBookUtils::run2array  ( $entry, $type, $posted_at_instrument ));
            }
        }
        return $result;
    }

    /**
     * A dictionary of known per-experiment sections and parameters
     *
     * @var array
     */
    public static $sections = array(

        /* Groups of parameters which are common for all instruments
         * and experiments
         */
        'HEADER' => array(

            array(

                'SECTION' => 'BEAMS',
                'TITLE'   => 'Electron and Photon beams',
                'PARAMS'  => array(

                    array( 'name' => 'BEND:DMP1:400:BDES',       'descr' => 'electron beam energy' ),
                    array( 'name' => 'EVNT:SYS0:1:LCLSBEAMRATE', 'descr' => 'beam rep rate' ),
                    array( 'name' => 'BPMS:DMP1:199:TMIT1H',     'descr' => 'Particle N_electrons' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO289',     'descr' => 'E.Vernier' ),
                    array( 'name' => 'BEAM:LCLS:ELEC:Q',         'descr' => 'Charge' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO195',     'descr' => 'Peak current after second bunch compressor' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO820',     'descr' => 'Pulse length' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO569',     'descr' => 'ebeam energy loss converted to photon mJ' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO580',     'descr' => 'Calculated number of photons' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO541',     'descr' => 'Photon beam energy' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO627',     'descr' => 'Photon beam energy' ),
                    array( 'name' => 'SIOC:SYS0:ML00:AO192',     'descr' => 'Wavelength' )
                )
            ),

            array(

                'SECTION' => 'FEE',
                'TITLE'   => 'FEE',
                'PARAMS'  => array(

                    array( 'name' => 'VGPR:FEE1:311:PSETPOINT_DES', 'descr' => 'Gas attenuator setpoint' ),
                    array( 'name' => 'VGCP:FEE1:311:P',             'descr' => 'Gas attenuator actual pressure' ),
                    array( 'name' => 'GATT:FEE1:310:R_ACT',         'descr' => 'Gas attenuator calculated transmission' ),
                    array( 'name' => 'SATT:FEE1:321:STATE',         'descr' => 'Solid attenuator 1' ),
                    array( 'name' => 'SATT:FEE1:322:STATE',         'descr' => 'Solid attenuator 2' ),
                    array( 'name' => 'SATT:FEE1:323:STATE',         'descr' => 'Solid attenuator 3' ),
                    array( 'name' => 'SATT:FEE1:324:STATE',         'descr' => 'Solid attenuator 4' ),
                    array( 'name' => 'SATT:FEE1:320:TACT',          'descr' => 'Total attenuator length' ),
                    array( 'name' => 'LVDT:FEE1:1811:LVPOS',        'descr' => 'FEE mirror LVDT position' ),
                    array( 'name' => 'LVDT:FEE1:1812:LVPOS',        'descr' => 'FEE mirror LVDT position' ),
                    array( 'name' => 'STEP:FEE1:1811:MOTR.RBV',     'descr' => 'FEE mirror RBV position' ),
                    array( 'name' => 'STEP:FEE1:1812:MOTR.RBV',     'descr' => 'FEE mirror RBV position' )
                )
            )
        ),

        /* Instrument-specific groups of parameters.
         */
        'AMO' => array(

            array(

                'SECTION' => 'HFP',
                'TITLE'   => 'HFP',
                'PARAMS'  => array(

                    array( 'name' => 'AMO:HFP:GCC:01:PMON', 'descr' => 'pressure' ),
                    array( 'name' => 'AMO:HFP:MMS:table.Z', 'descr' => 'z-position' )
                )
            ),

            array(

                'SECTION' => 'ETOF',
                'TITLE'   => 'eTOF',
                'PARAMS'  => array(

                    array( 'name' => 'AMO:R14:IOC:10:ao0:out1',                'descr' => '1' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH0:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH1:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH2:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS0:CH3:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH0:VoltageMeasure', 'descr' => '' ),

                    array( 'name' => 'AMO:R14:IOC:10:ao0:out2',                'descr' => '2' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH0:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH1:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH2:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS1:CH3:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH1:VoltageMeasure', 'descr' => '' ),

                    array( 'name' => 'AMO:R14:IOC:10:ao0:out3',                'descr' => '3' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH0:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH1:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH2:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS2:CH3:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH2:VoltageMeasure', 'descr' => '' ),

                    array( 'name' => 'AMO:R14:IOC:10:ao0:out4',                'descr' => '4' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH0:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH1:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH2:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS3:CH3:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS7:CH3:VoltageMeasure', 'descr' => '' ),

                    array( 'name' => 'AMO:R14:IOC:10:ao0:out5',                'descr' => '5' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH0:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH1:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH2:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS4:CH3:VoltageMeasure', 'descr' => '' ),
                    array( 'name' => 'AMO:R14:IOC:10:VHS8:CH0:VoltageMeasure', 'descr' => '' )
                )
            ),

            array(

                'SECTION' => 'ITOF',
                'TITLE'   => 'iTOF',
                'PARAMS'  => array(

                    array( 'name' => 'AMO:R14:IOC:21:VHS2:CH0:VoltageMeasure', 'descr' => 'repeller' ),
                    array( 'name' => 'AMO:R14:IOC:21:VHS0:CH0:VoltageMeasure', 'descr' => 'extractor' ),
                    array( 'name' => 'AMO:R14:IOC:21:VHS0:CH1:VoltageMeasure', 'descr' => 'acceleration' ),
                    array( 'name' => 'AMO:R14:IOC:21:VHS0:CH2:VoltageMeasure', 'descr' => 'MCP in' ),
                    array( 'name' => 'AMO:R14:IOC:21:VHS2:CH2:VoltageMeasure', 'descr' => 'MCP out' ),
                    array( 'name' => 'AMO:R14:IOC:21:VHS2:CH1:VoltageMeasure', 'descr' => 'Anode' )
                )
            ),

            array(

                'SECTION' => 'HFP_GAS',
                'TITLE'   => 'HFP Gas',
                'PARAMS'  => array(

                    array( 'name' => 'AMO:HFP:GCC:03:PMON',                    'descr' => 'pressure' ),
                    array( 'name' => 'AMO:R14:IOC:21:VHS7:CH0:VoltageMeasure', 'descr' => 'piezo voltage' ),
                    array( 'name' => 'AMO:R14:EVR:21:CTRL.DG2D',               'descr' => 'piezo timing delay' ),
                    array( 'name' => 'AMO:R14:EVR:21:CTRL.DG2W',               'descr' => 'piezo timing width' ),
                    array( 'name' => 'AMO:HFP:MMS:72.RBV',                     'descr' => 'gasjet x-position (rel. distance)' ),
                    array( 'name' => 'AMO:HFP:MMS:71.RBV',                     'descr' => 'gasjet y-position (rel. distance)' ),
                    array( 'name' => 'AMO:HFP:MMS:73.RBV',                     'descr' => 'Gas Jet motor Z axis (mm)' )
                )
            ),

            array(

                'SECTION' => 'DIA',
                'TITLE'   => 'DIA',
                'PARAMS'  => array(

                    array( 'name' => 'AMO:DIA:GCC:01:PMON', 'descr' => 'pressure' )
                )
            ),

            array(

                'SECTION' => 'MBES',
                'TITLE'   => 'MBES',
                'PARAMS'  => array(

                    array( 'name' => 'AMO:DIA:SHC:11:I', 'descr' => 'coil 1' ),
                    array( 'name' => 'AMO:DIA:SHC:12:I', 'descr' => 'coil 2' ),

                    array( 'name' => 'AMO:R15:IOC:40:VHS0:CH0:VoltageSet', 'descr' => 'AMO:R15:IOC:40:VHS0:CH0:VoltageSet' ),
                    array( 'name' => 'AMO:R15:IOC:40:VHS0:CH1:VoltageSet', 'descr' => 'AMO:R15:IOC:40:VHS0:CH1:VoltageSet' ),
                    array( 'name' => 'AMO:R15:IOC:40:VHS2:CH1:VoltageSet', 'descr' => 'AMO:R15:IOC:40:VHS2:CH1:VoltageSet' ),
                    array( 'name' => 'AMO:R15:IOC:40:VHS2:CH2:VoltageSet', 'descr' => 'AMO:R15:IOC:40:VHS2:CH2:VoltageSet' )
                )
            )
        ),

        'SXR' => array(

            array(

                'SECTION' => 'COL',
                'TITLE'   => 'Collimator',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:COL:GCC:01:PMON', 'descr' => 'SXR:COL:GCC:01:PMON' ),
                    array( 'name' => 'SXR:COL:PIP:01:PMON', 'descr' => 'SXR:COL:PIP:01:PMON' )
                )
            ),

            array(

                'SECTION' => 'EXS',
                'TITLE'   => 'Exit Slit',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:EXS:GCC:01:PMON', 'descr' => 'SXR:EXS:GCC:01:PMON' ),
                    array( 'name' => 'SXR:EXS:MMS:01.VAL',  'descr' => 'SXR:EXS:MMS:01.VAL'  ),
                    array( 'name' => 'SXR:EXS:PIP:01:PMON', 'descr' => 'SXR:EXS:PIP:01:PMON' )
                )
            ),

            array(

                'SECTION' => 'FLX',
                'TITLE'   => 'Flux Chamber',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:FLX:GCC:01:PMON', 'descr' => 'SXR:FLX:GCC:01:PMON' ),
                    array( 'name' => 'SXR:FLX:STC:01',      'descr' => 'SXR:FLX:STC:01' )
                )
            ),

            array(

                'SECTION' => 'KBO',
                'TITLE'   => 'KBO Optics',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:KBO:GCC:01:PMON', 'descr' => 'SXR:KBO:GCC:01:PMON' ),
                    array( 'name' => 'SXR:KBO:GCC:02:PMON', 'descr' => 'SXR:KBO:GCC:02:PMON' )
                )
            ),

            array(

                'SECTION' => 'LIN',
                'TITLE'   => 'Laser Incoupling',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:LIN:GCC:01:PMON', 'descr' => 'SXR:LIN:GCC:01:PMON' )
                )
            ),

            array(

                'SECTION' => 'MON',
                'TITLE'   => 'Grating',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:MON:GCC:01:PMON', 'descr' => 'SXR:MON:GCC:01:PMON' ),
                    array( 'name' => 'SXR:MON:MMS:01.VAL',  'descr' => 'SXR:MON:MMS:01.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:02.VAL',  'descr' => 'SXR:MON:MMS:02.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:03.VAL',  'descr' => 'SXR:MON:MMS:03.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:04.VAL',  'descr' => 'SXR:MON:MMS:04.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:05.VAL',  'descr' => 'SXR:MON:MMS:05.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:06.VAL',  'descr' => 'SXR:MON:MMS:06.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:07.VAL',  'descr' => 'SXR:MON:MMS:07.VAL'  ),
                    array( 'name' => 'SXR:MON:MMS:08.VAL',  'descr' => 'SXR:MON:MMS:08.VAL'  ),
                    array( 'name' => 'SXR:MON:PIP:01:PMON', 'descr' => 'SXR:MON:PIP:01:PMON' ),
                    array( 'name' => 'SXR:MON:STC:01',      'descr' => 'SXR:MON:STC:01'      ),
                    array( 'name' => 'SXR:MON:STC:02',      'descr' => 'SXR:MON:STC:02'      )
                )
            ),

            array(

                'SECTION' => 'PST',
                'TITLE'   => 'Photon Stopper',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:PST:PIP:01:PMON',  'descr' => 'SXR:PST:PIP:01:PMON' ),
                    array( 'name' => 'PPS:NEH1:2:S2STPRSUM', 'descr' => 'PPS:NEH1:2:S2STPRSUM' )
                )
            ),

            array(

                'SECTION' => 'SPS',
                'TITLE'   => 'Single Pulse Shutter',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:SPS:GCC:01:PMON', 'descr' => 'SXR:SPS:GCC:01:PMON' ),
                    array( 'name' => 'SXR:SPS:MPA:01:OUT',  'descr' => 'SXR:SPS:MPA:01:OUT'  ),
                    array( 'name' => 'SXR:SPS:PIP:01:PMON', 'descr' => 'SXR:SPS:PIP:01:PMON' ),
                    array( 'name' => 'SXR:SPS:STC:01',      'descr' => 'SXR:SPS:STC:01'      )
                )
            ),

            array(

                'SECTION' => 'TSS',
                'TITLE'   => 'Transmission Sample Stage',
                'PARAMS'  => array(

                    array( 'name' => 'SXR:TSS:GCC:01:PMON', 'descr' => 'SXR:TSS:GCC:01:PMON' ),
                    array( 'name' => 'SXR:TSS:GCC:02:PMON', 'descr' => 'SXR:TSS:GCC:02:PMON' ),
                    array( 'name' => 'SXR:TSS:MMS:01.VAL',  'descr' => 'SXR:TSS:MMS:01.VAL'  ),
                    array( 'name' => 'SXR:TSS:MMS:02.VAL',  'descr' => 'SXR:TSS:MMS:02.VAL'  ),
                    array( 'name' => 'SXR:TSS:MMS:03.VAL',  'descr' => 'SXR:TSS:MMS:03.VAL'  ),
                    array( 'name' => 'SXR:TSS:PIP:01:PMON', 'descr' => 'SXR:TSS:PIP:01:PMON' )
                )
            )
        ),

        'XPP' => array(
            array(

                'SECTION' => 'scan',
                'TITLE'   => 'Scan Info',
                'PARAMS'  => array(
                    array( 'descr' => 'Run_is_scan',              'name' => 'XPP:SCAN:ISSCAN' ),
                    array( 'descr' => 'Scanmotor_0',              'name' => 'XPP:SCAN:SCANVAR00' ),
                    array( 'descr' => 'Scanmotor_1',              'name' => 'XPP:SCAN:SCANVAR01' ),
                    array( 'descr' => 'Scanmotor_2',              'name' => 'XPP:SCAN:SCANVAR02' ),
                    array( 'descr' => 'Scanmotor_0_position_min', 'name' => 'XPP:SCAN:MIN00' ),
                    array( 'descr' => 'Scanmotor_0_position_max', 'name' => 'XPP:SCAN:MAX00' ),
                    array( 'descr' => 'Scanmotor_1_position_min', 'name' => 'XPP:SCAN:MIN01' ),
                    array( 'descr' => 'Scanmotor_1_position_max', 'name' => 'XPP:SCAN:MAX01' ),
                    array( 'descr' => 'Scanmotor_2_position_min', 'name' => 'XPP:SCAN:MIN02' ),
                    array( 'descr' => 'Scanmotor_2_position_max', 'name' => 'XPP:SCAN:MAX02' ),
                    array( 'descr' => 'Scan_Number_of_steps',     'name' => 'XPP:SCAN:NSTEPS' ),
                    array( 'descr' => 'Scan_shots_pre_step',      'name' => 'XPP:SCAN:NSHOTS' )
                )
            )
        ),

        'CXI' => array(

            array(

                'SECTION' => 'undulator',
                'TITLE'   => 'Undulator Status',
                'PARAMS'  => array(
                    array( 'descr' => 'Undulator  1 X pos', 'name' => 'USEG:UND1:150:XIN' ),
                    array( 'descr' => 'Undulator  2 X pos', 'name' => 'USEG:UND1:250:XIN' ),
                    array( 'descr' => 'Undulator  3 X pos', 'name' => 'USEG:UND1:350:XIN' ),
                    array( 'descr' => 'Undulator  4 X pos', 'name' => 'USEG:UND1:450:XIN' ),
                    array( 'descr' => 'Undulator  5 X pos', 'name' => 'USEG:UND1:550:XIN' ),
                    array( 'descr' => 'Undulator  6 X pos', 'name' => 'USEG:UND1:650:XIN' ),
                    array( 'descr' => 'Undulator  7 X pos', 'name' => 'USEG:UND1:750:XIN' ),
                    array( 'descr' => 'Undulator  8 X pos', 'name' => 'USEG:UND1:850:XIN' ),
                    array( 'descr' => 'Undulator  9 X pos', 'name' => 'USEG:UND1:950:XIN' ),
                    array( 'descr' => 'Undulator 10 X pos', 'name' => 'USEG:UND1:1050:XIN' ),
                    array( 'descr' => 'Undulator 11 X pos', 'name' => 'USEG:UND1:1150:XIN' ),
                    array( 'descr' => 'Undulator 12 X pos', 'name' => 'USEG:UND1:1250:XIN' ),
                    array( 'descr' => 'Undulator 13 X pos', 'name' => 'USEG:UND1:1350:XIN' ),
                    array( 'descr' => 'Undulator 14 X pos', 'name' => 'USEG:UND1:1450:XIN' ),
                    array( 'descr' => 'Undulator 15 X pos', 'name' => 'USEG:UND1:1550:XIN' ),
                    array( 'descr' => 'Undulator 16 X pos', 'name' => 'USEG:UND1:1650:XIN' ),
                    array( 'descr' => 'Undulator 17 X pos', 'name' => 'USEG:UND1:1750:XIN' ),
                    array( 'descr' => 'Undulator 18 X pos', 'name' => 'USEG:UND1:1850:XIN' ),
                    array( 'descr' => 'Undulator 19 X pos', 'name' => 'USEG:UND1:1950:XIN' ),
                    array( 'descr' => 'Undulator 20 X pos', 'name' => 'USEG:UND1:2050:XIN' ),
                    array( 'descr' => 'Undulator 21 X pos', 'name' => 'USEG:UND1:2150:XIN' ),
                    array( 'descr' => 'Undulator 22 X pos', 'name' => 'USEG:UND1:2250:XIN' ),
                    array( 'descr' => 'Undulator 23 X pos', 'name' => 'USEG:UND1:2350:XIN' ),
                    array( 'descr' => 'Undulator 24 X pos', 'name' => 'USEG:UND1:2450:XIN' ),
                    array( 'descr' => 'Undulator 25 X pos', 'name' => 'USEG:UND1:2550:XIN' ),
                    array( 'descr' => 'Undulator 26 X pos', 'name' => 'USEG:UND1:2650:XIN' ),
                    array( 'descr' => 'Undulator 27 X pos', 'name' => 'USEG:UND1:2750:XIN' ),
                    array( 'descr' => 'Undulator 28 X pos', 'name' => 'USEG:UND1:2850:XIN' )
                )
            ),
            array(

                'SECTION' => 'FEE_data',
                'TITLE'   => 'FEE Data',
                'PARAMS'  => array(
                    array( 'descr' => 'FEE mask slit X- blade', 'name' => 'STEP:FEE1:151:MOTR.RBV'   ),
                    array( 'descr' => 'FEE mask slit X+ blade', 'name' => 'STEP:FEE1:152:MOTR.RBV'   ),
                    array( 'descr' => 'FEE mask slit Y- blade', 'name' => 'STEP:FEE1:153:MOTR.RBV'   ),
                    array( 'descr' => 'FEE mask slit Y+ blade', 'name' => 'STEP:FEE1:154:MOTR.RBV'   ),
                    array( 'descr' => 'FEE mask slit X center', 'name' => 'SLIT:FEE1:ACTUAL_XCENTER' ),
                    array( 'descr' => 'FEE mask slit X width',  'name' => 'SLIT:FEE1:ACTUAL_XWIDTH'  ),
                    array( 'descr' => 'FEE mask slit Y center', 'name' => 'SLIT:FEE1:ACTUAL_YCENTER' ),
                    array( 'descr' => 'FEE mask slit Y width',  'name' => 'SLIT:FEE1:ACTUAL_YWIDTH'  ),
                    array( 'descr' => 'FEE M1H X pos',          'name' => 'STEP:FEE1:611:MOTR.RBV'   ),
                    array( 'descr' => 'FEE M1H dX',             'name' => 'STEP:FEE1:612:MOTR.RBV'   ),
                    array( 'descr' => 'FEE M2H X pos',          'name' => 'STEP:FEE1:861:MOTR.RBV'   ),
                    array( 'descr' => 'FEE M2H dX',             'name' => 'STEP:FEE1:862:MOTR.RBV'   ),
                    array( 'descr' => 'FEE M1H X LVDT',         'name' => 'LVDT:FEE1:611:LVPOS'      ),
                    array( 'descr' => 'FEE M1H dX LVDT',        'name' => 'LVDT:FEE1:612:LVPOS'      ),
                    array( 'descr' => 'FEE M2H X LVDT',         'name' => 'LVDT:FEE1:861:LVPOS'      ),
                    array( 'descr' => 'FEE M2H dX LVDT',        'name' => 'LVDT:FEE1:862:LVPOS'      ),
                    array( 'descr' => 'FEE attenuator 1',       'name' => 'SATT:FEE1:321:STATE'      ),
                    array( 'descr' => 'FEE attenuator 2',       'name' => 'SATT:FEE1:322:STATE'      ),
                    array( 'descr' => 'FEE attenuator 3',       'name' => 'SATT:FEE1:323:STATE'      ),
                    array( 'descr' => 'FEE attenuator 4',       'name' => 'SATT:FEE1:324:STATE'      ),
                    array( 'descr' => 'FEE attenuator 5',       'name' => 'SATT:FEE1:325:STATE'      ),
                    array( 'descr' => 'FEE attenuator 6',       'name' => 'SATT:FEE1:326:STATE'      ),
                    array( 'descr' => 'FEE attenuator 7',       'name' => 'SATT:FEE1:327:STATE'      ),
                    array( 'descr' => 'FEE attenuator 8',       'name' => 'SATT:FEE1:328:STATE'      ),
                    array( 'descr' => 'FEE attenuator 9',       'name' => 'SATT:FEE1:329:STATE'      ),
                    array( 'descr' => 'FEE total attenuator',   'name' => 'SATT:FEE1:320:TACT'       )
                )
            ),
            array(

                'SECTION' => 'CXI',
                'TITLE'   => 'CXI',
                'PARAMS'  => array(
                    array( 'descr' => 'Unfocused',        'name' => 'CXI:MPS:CFG:1_MPSC' ),
                    array( 'descr' => '10 um XRT lens',   'name' => 'CXI:MPS:CFG:2_MPSC' ),
                    array( 'descr' => '1 um DG2 lens',    'name' => 'CXI:MPS:CFG:3_MPSC' ),
                    array( 'descr' => '1 um KB mirror',   'name' => 'CXI:MPS:CFG:4_MPSC' ),
                    array( 'descr' => '100 nm KB mirror', 'name' => 'CXI:MPS:CFG:5_MPSC' )
                )
            ),
            array(

                'SECTION' => 'DIA',
                'TITLE'   => 'DIA',
                'PARAMS'  => array(
                    array( 'descr' => '20 um Si foil',    'name' => 'XRT:DIA:MMS:02.RBV' ),
                    array( 'descr' => '40 um Si foil',    'name' => 'XRT:DIA:MMS:03.RBV' ),
                    array( 'descr' => '80 um Si foil',    'name' => 'XRT:DIA:MMS:04.RBV' ),
                    array( 'descr' => '160 um Si foil',   'name' => 'XRT:DIA:MMS:05.RBV' ),
                    array( 'descr' => '320 um Si foil',   'name' => 'XRT:DIA:MMS:06.RBV' ),
                    array( 'descr' => '640 um Si foil',   'name' => 'XRT:DIA:MMS:07.RBV' ),
                    array( 'descr' => '1280 um Si foil',  'name' => 'XRT:DIA:MMS:08.RBV' ),
                    array( 'descr' => '2560 um Si foil',  'name' => 'XRT:DIA:MMS:09.RBV' ),
                    array( 'descr' => '5120 um Si foil',  'name' => 'XRT:DIA:MMS:10.RBV' ),
                    array( 'descr' => '10240 um Si foil', 'name' => 'XRT:DIA:MMS:11.RBV' ),
                    array( 'descr' => 'XRT lens out',     'name' => 'XRT:DIA:MMS:14.HLS' ),
                    array( 'descr' => 'XRT lens Y pos',   'name' => 'XRT:DIA:MMS:14.RBV' )
                   )
            ),
            array(

                'SECTION' => 'DG1',
                'TITLE'   => 'DG1',
                'PARAMS'  => array(
                    array( 'descr' => 'DG1 slit X center', 'name' => 'CXI:DG1:JAWS:XTRANS.C' ),
                    array( 'descr' => 'DG1 slit X width',  'name' => 'CXI:DG1:JAWS:YTRANS.C' ),
                    array( 'descr' => 'DG1 slit Y center', 'name' => 'CXI:DG1:JAWS:XTRANS.D' ),
                    array( 'descr' => 'DG1 slit Y width',  'name' => 'CXI:DG1:JAWS:YTRANS.D' ),
                    array( 'descr' => 'DG1 Navitar Zoom',  'name' => 'CXI:DG1:CLZ:01.RBV'    )
                )
            ),
            array(

                'SECTION' => 'DG2',
                'TITLE'   => 'DG2',
                'PARAMS'  => array(
                    array( 'descr' => 'DS2/DG2 valve open',   'name' => 'CXI:DG2:VGC:01:OPN_DI' ),
                    array( 'descr' => 'DS2/DG2 valve closed', 'name' => 'CXI:DG2:VGC:01:CLS_DI' ),
                    array( 'descr' => 'DG2 slit X center',    'name' => 'CXI:DG2:JAWS:XTRANS.C' ),
                    array( 'descr' => 'DG2 slit X width',     'name' => 'CXI:DG2:JAWS:YTRANS.C' ),
                    array( 'descr' => 'DG2 slit Y center',    'name' => 'CXI:DG2:JAWS:XTRANS.D' ),
                    array( 'descr' => 'DG2 slit Y width',     'name' => 'CXI:DG2:JAWS:YTRANS.D' ),
                    array( 'descr' => 'DG2 IPM diode Y pos',  'name' => 'CXI:DG2:MMS:08.RBV'    ),
                    array( 'descr' => 'DG2 IPM target Y pos', 'name' => 'CXI:DG2:MMS:10.RBV'    ),
                    array( 'descr' => 'DG2 lens out',         'name' => 'CXI:DG2:MMS:06.HLS'    ),
                    array( 'descr' => 'DG2 lens Y pos',       'name' => 'CXI:DG2:MMS:06.RBV'    ),
                    array( 'descr' => 'DG2 Navitar zoom',     'name' => 'CXI:DG2:CLZ:01.RBV'    ),
                    array( 'descr' => 'DG2/DSU valve open',   'name' => 'CXI:DG2:VGC:02:OPN_DI' ),
                    array( 'descr' => 'DG2/DSU valve closed', 'name' => 'CXI:DG2:VGC:02:CLS_DI' )
                )
            ),
            array(

                'SECTION' => 'KB1',
                'TITLE'   => 'KB1',
                'PARAMS'  => array(
                    array( 'descr' => 'KB1 chamber pressure',   'name' => 'CXI:KB1:GCC:02:PMON' ),
                    array( 'descr' => 'KB1 slit (US) X center', 'name' => 'CXI:KB1:JAWS:US:XTRANS.C' ),
                    array( 'descr' => 'KB1 slit (US) X width',  'name' => 'CXI:KB1:JAWS:US:YTRANS.C' ),
                    array( 'descr' => 'KB1 slit (US) Y center', 'name' => 'CXI:KB1:JAWS:US:XTRANS.D' ),
                    array( 'descr' => 'KB1 slit (US) Y width',  'name' => 'CXI:KB1:JAWS:US:YTRANS.D' ),
                    array( 'descr' => 'KB1 Horizontal X pos',   'name' => 'CXI:KB1:MMS:05.RBV' ),
                    array( 'descr' => 'KB1 Horizontal Y pos',   'name' => 'CXI:KB1:MMS:06.RBV' ),
                    array( 'descr' => 'KB1 Horizontal pitch',   'name' => 'CXI:KB1:MMS:07.RBV' ),
                    array( 'descr' => 'KB1 Horizontal roll',    'name' => 'CXI:KB1:MMS:08.RBV' ),
                    array( 'descr' => 'KB1 Vertical X pos',     'name' => 'CXI:KB1:MMS:09.RBV' ),
                    array( 'descr' => 'KB1 Vertical Y pos',     'name' => 'CXI:KB1:MMS:10.RBV' ),
                    array( 'descr' => 'KB1 Vertical pitch',     'name' => 'CXI:KB1:MMS:11.RBV' ),
                    array( 'descr' => 'KB1 slit (DS) X center', 'name' => 'CXI:KB1:JAWS:DS:XTRANS.C' ),
                    array( 'descr' => 'KB1 slit (DS) X width',  'name' => 'CXI:KB1:JAWS:DS:YTRANS.C' ),
                    array( 'descr' => 'KB1 slit (US) Y center', 'name' => 'CXI:KB1:JAWS:DS:XTRANS.D' ),
                    array( 'descr' => 'KB1 slit (US) Y width',  'name' => 'CXI:KB1:JAWS:DS:YTRANS.D' ),
                    array( 'descr' => 'KB1 Navitar Zoom',       'name' => 'CXI:KB1:CLZ:01.RBV' )
                )
            ),
            array(

                'SECTION' => 'KB2',
                'TITLE'   => 'KB2',
                'PARAMS'  => array(
                    array( 'descr' => 'DSU slit X center', 'name' => 'CXI:DSU:JAWS:XTRANS.C' ),
                    array( 'descr' => 'DSU slit X width',  'name' => 'CXI:DSU:JAWS:YTRANS.C' ),
                    array( 'descr' => 'DSU slit Y center', 'name' => 'CXI:DSU:JAWS:XTRANS.D' ),
                    array( 'descr' => 'DSU slit Y width',  'name' => 'CXI:DSU:JAWS:YTRANS.D' )
                )
            ),
            array(

                'SECTION' => 'DSU',
                'TITLE'   => 'DSU',
                'PARAMS'  => array(
                )
            ),
            array(

                'SECTION' => 'SC1',
                'TITLE'   => 'SC1',
                'PARAMS'  => array(
                    array( 'descr' => 'SC1 chamber pressure',   'name' => 'CXI:SC1:GCC:01:PMON'           ),
                    array( 'descr' => 'DSU/SC1 valve open',     'name' => 'CXI:SC1:VGC:01:OPN_DI'         ),
                    array( 'descr' => 'DSU/SC1 valve closed',   'name' => 'CXI:SC1:VGC:01:CLS_DI'         ),
                    array( 'descr' => 'SC1 MZM aperture 1 X',   'name' => 'CXI:SC1:MZM:01:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM aperture 1 Y',   'name' => 'CXI:SC1:MZM:02:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM aperture 2 X',   'name' => 'CXI:SC1:MZM:03:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM aperture 2 Y',   'name' => 'CXI:SC1:MZM:04:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM aperture 3 X',   'name' => 'CXI:SC1:MZM:05:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM aperture 3 Y',   'name' => 'CXI:SC1:MZM:06:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM aperture 3 Z',   'name' => 'CXI:SC1:MZM:07:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM sample X',       'name' => 'CXI:SC1:MZM:08:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM sample Y',       'name' => 'CXI:SC1:MZM:09:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM sample Z',       'name' => 'CXI:SC1:MZM:10:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM part aper X',    'name' => 'CXI:SC1:MZM:12:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM part aper Z',    'name' => 'CXI:SC1:MZM:13:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM view mirror X',  'name' => 'CXI:SC1:MZM:14:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 MZM view mirror Y',  'name' => 'CXI:SC1:MZM:15:ENCPOSITIONGET' ),
                    array( 'descr' => 'SC1 sample yaw',         'name' => 'CXI:SC1:PIC:01.RBV'            ),
                    array( 'descr' => 'SC1 sample pitch/yaw 1', 'name' => 'CXI:SC1:PIC:02.RBV'            ),
                    array( 'descr' => 'SC1 sample pitch/yaw 2', 'name' => 'CXI:SC1:PIC:03.RBV'            ),
                    array( 'descr' => 'SC1 sample pitch/yaw 3', 'name' => 'CXI:SC1:PIC:04.RBV'            ),
                    array( 'descr' => 'SC1 sample x (long)',    'name' => 'CXI:SC1:MMS:02.RBV'            ),
                    array( 'descr' => 'SC1/DS1 valve open',     'name' => 'CXI:SC1:VGC:02:OPN_DI'         ),
                    array( 'descr' => 'SC1/DS1 valve closed',   'name' => 'CXI:SC1:VGC:02:CLS_DI'         )
                )
            ),
            array(

                'SECTION' => 'DS1',
                'TITLE'   => 'DS1',
                'PARAMS'  => array(
                    array( 'descr' => 'DS1 chamber pressure',  'name' => 'CXI:DS1:GCC:01:PMON'              ),
                    array( 'descr' => 'DS1 detector Z pos',    'name' => 'CXI:DS1:MMS:06.RBV'               ),
                    array( 'descr' => 'DS1 stick Y pos',       'name' => 'CXI:DS1:MMS:07.RBV'               ),
                    array( 'descr' => 'DS1 chiller temp',      'name' => 'CXI:DS1:TEMPERATURE'              ),
                    array( 'descr' => 'DS1 chiller flowmeter', 'name' => 'CXI:DS1:FLOW_METER'               ),
                    array( 'descr' => 'DS1 quad 0 temp',       'name' => 'CXI:DS1:TE-TECH1:ACTUAL_TEMP'     ),
                    array( 'descr' => 'DS1 quad 1 temp',       'name' => 'CXI:DS1:TE-TECH1:ACTUAL_SEC_TEMP' ),
                    array( 'descr' => 'DS1 quad 2 temp',       'name' => 'CXI:DS1:TE-TECH2:ACTUAL_TEMP'     ),
                    array( 'descr' => 'DS1 quad 3 temp',       'name' => 'CXI:DS1:TE-TECH2:ACTUAL_SEC_TEMP' ),
                    array( 'descr' => 'DS1 bias voltage',      'name' => 'CXI:DS1:BIAS'                     )
                )
            ),
            array(

                'SECTION' => 'DSD',
                'TITLE'   => 'DSD',
                'PARAMS'  => array(
                    array( 'descr' => 'DSD chamber pressure',  'name' => 'CXI:DSD:GCC:01:PMON'   ),
                    array( 'descr' => 'DSD detector Z pos',    'name' => 'CXI:DSD:MMS:06.RBV'    ),
                    array( 'descr' => 'DSD chiller temp',      'name' => 'CXI:DS1:TEMPERATURE'   ),
                    array( 'descr' => 'DSD chiller flowmeter', 'name' => 'CXI:DS1:FLOW_METER'    ),
                    array( 'descr' => '1MS/DG3 valve open',    'name' => 'CXI:DS1:VGC:01:OPN_DI' ),
                    array( 'descr' => '1MS/DG3 valve closed',  'name' => 'CXI:DS1:VGC:01:CLS_DI' )
                )
            ),
            array(

                'SECTION' => 'DG4',
                'TITLE'   => 'DG4',
                'PARAMS'  => array(
                    array( 'descr' => 'DG4 IPM diode Y',  'name' => 'CXI:DG4:MMS:02.RBV' ),
                    array( 'descr' => 'DG4 IPM target Y', 'name' => 'CXI:DG4:MMS:03.RBV' ),
                    array( 'descr' => 'DG4 Navitar zoom', 'name' => 'CXI:DG4:CLZ:01.RBV' )
                )
            ),
            array(

                'SECTION' => 'USR',
                'TITLE'   => 'USR',
                'PARAMS'  => array(
                    array( 'descr' => 'User motor ch  1',        'name' => 'CXI:USR:MMS:01.RBV'   ),
                    array( 'descr' => 'User motor ch  2',        'name' => 'CXI:USR:MMS:02.RBV'   ),
                    array( 'descr' => 'User motor ch  3',        'name' => 'CXI:USR:MMS:03.RBV'   ),
                    array( 'descr' => 'User motor ch  4',        'name' => 'CXI:USR:MMS:04.RBV'   ),
                    array( 'descr' => 'User motor ch  5',        'name' => 'CXI:USR:MMS:05.RBV'   ),
                    array( 'descr' => 'User motor ch  6',        'name' => 'CXI:USR:MMS:06.RBV'   ),
                    array( 'descr' => 'User motor ch 17',        'name' => 'CXI:USR:MMS:17.RBV'   ),
                    array( 'descr' => 'User motor ch 18',        'name' => 'CXI:USR:MMS:18.RBV'   ),
                    array( 'descr' => 'User motor ch 19',        'name' => 'CXI:USR:MMS:19.RBV'   ),
                    array( 'descr' => 'Current gas pressure 1',  'name' => 'CXI:R52:AI:PRES_REG1' ),
                    array( 'descr' => 'Total pressure change 1', 'name' => 'CXI:R52:UPDATE1'      ),
                    array( 'descr' => 'Current gas pressure 2',  'name' => 'CXI:R52:AI:PRES_REG2' ),
                    array( 'descr' => 'Total pressure change 2', 'name' => 'CXI:R52:UPDATE2'      )
                )
            )
        ),

        'XCS' => array(
        ),
        'MEC' => array(
        )
    );

    /**
     * A dictionary of known per-run attribute sections
     *
     * @var array
     */
    public static $attribute_sections = array (
        'DAQ_Detectors' => 'DAQ Detectors'
    ) ;

    /**
     * Return definitions of the EPICS PV sections along with PVs
     *
     * The result is returned as teh following data structure:
     *
     *   { 'sections'      : { <section_name> : { 'title'      : <section-title> ,
     *                                            'parameters' : [ <pv-name>, ... ] } } ,
     *     'section_names' : [ <section_name>, ... ]
     *   }
     *
     * The 'section_names' array is used to store the order in which the sections should
     * be used in the Web UI.
     *
     * @param LogBookExperiment $experiment
     * @return array
     */
    public static function get_epics_sections ($experiment) {

        $instr_name = $experiment->instrument()->name() ;

        $section_names = array() ;  // ordered list of names
        $sections = array() ;

        // Predefined sections first

        $in_use = array() ;

        foreach (array('HEADER', $instr_name) as $area) {
            foreach (LogBookUtils::$sections[$area] as $section) {
                $parameters = array() ;
                foreach ($section['PARAMS'] as $p) {
                    $p_name = $p['name'] ;
                    $in_use[$p_name] = True ;
                    
                    // - overload the defaulr description of the parameter from the database
                    //   if it's available in there
            
                    $p_descr = $p['descr'] ;
                    $param = $experiment->find_run_param_by_name($p_name) ;
                    if ($param) {
                        $descr = $param->description() ;
                        switch ($descr) {
                            case ''  :
                            case 'PV': break ;
                            default  : $p_descr = $descr ; break ;
                        }
                    }
                    array_push($parameters, array (
                        'name'  => $p_name ,
                        'descr' => $p_descr
                    )) ;
                }
                sort($parameters) ;
                $s_name = $section['SECTION'] ;
                array_push($section_names, $s_name) ;
                $sections[$s_name] = array (
                    'title'      => $section['TITLE'] ,
                    'parameters' => $parameters) ;
            }
        }

        // The last section is for any other parameters

        $parameters = array() ;
        foreach ($experiment->run_params() as $param) {
            $p_name = $param->name() ;
            if (!array_key_exists($p_name, $in_use)) {
                $in_use[$p_name] = True ;
                $p_descr = $p_name ;
                $descr = $param->description() ;
                switch ($descr) {
                    case ''  :
                    case 'PV': break ;
                    default  : $p_descr = $descr ; break ;
                }
                array_push($parameters, array (
                    'name'  => $p_name ,
                    'descr' => $p_descr
                )) ;
            }
        }
        $s_name = 'FOOTER' ;
        array_push($section_names, $s_name) ;
        $sections[$s_name] = array (
            'title'      => 'Additional Parameters' ,
            'parameters' => $parameters) ;

        return array(
            'section_names' => $section_names ,
            'sections'      => $sections
        ) ;
    }

    public static function get_epics_sections_1 ($experiment) {

        $instr_name = $experiment->instrument()->name() ;

        $section_names = array() ;  // ordered list of names
        $sections = array() ;

        // Predefined sections first

        $in_use = array() ;

        foreach (array('HEADER', $instr_name) as $area) {
            foreach (LogBookUtils::$sections[$area] as $section) {
                $parameters = array() ;
                foreach ($section['PARAMS'] as $p) {
                    $p_name = $p['name'] ;
                    $in_use[$p_name] = True ;
                    array_push($parameters, $p_name) ;
                }
                sort($parameters) ;
                $s_name = $section['SECTION'] ;
                array_push($section_names, $s_name) ;
                $sections[$s_name] = array (
                    'title'      => $section['TITLE'] ,
                    'parameters' => $parameters) ;
            }
        }

        // The last section is for any other parameters

        $parameters = array() ;
        foreach ($experiment->run_params() as $p) {
            $p_name = $p->name() ;
            if (!array_key_exists($p_name, $in_use)) {
                $in_use[$p_name] = True ;
                array_push($parameters, $p_name) ;
            }
        }
        $s_name = 'FOOTER' ;
        array_push($section_names, $s_name) ;
        $sections[$s_name] = array (
            'title'      => 'Additional Parameters' ,
            'parameters' => $parameters) ;

        return array(
            'section_names' => $section_names ,
            'sections'      => $sections
        ) ;
    }

    /**
     * Return names of detectors used for a specific range of runs of an experiment
     *
     * The result is returned into the following data structure:
     * 
     *   {  'runs'           : { <run-num>  : { <detector> : 1 } } ,
     *      'detectors'      :                { <detector> : 1 } ,
     *      'detector_names' : [ <detector> , ... ]
     *   }
     * 
     * Note that the first dictionary will tell a caller all detectors used
     * for a particular run. MEanwhile the second one - all detectors which
     * have ever been used in any run of the specified range (of runs).
     *
     * @param LogBookExperiment $experiment
     * @param integer $from_runnum
     * @param integer $through_runnum
     * @return array()
     */
    public static function get_daq_detectors ($experiment, $from_runnum=null, $through_runnum=null) {
        $runs = array() ;
        $detectors = array() ;
        foreach ($experiment->runs() as $run) {

            $runnum = $run->num() ;
            if ($from_runnum    && ($runnum < $from_runnum))    continue ;
            if ($through_runnum && ($runnum > $through_runnum)) continue ;

            $runs[$runnum] = array() ;

            // ATTENTION: Scan the legacy class 'DAQ_Detectors' for experiments
            //            taken prior Feb 2015 when a transition of the class
            //            name to 'DAQ Detectors' happened.
            //
            foreach (array('DAQ_Detectors', 'DAQ Detectors') as $class) {
                foreach ($run->attributes($class) as $attr) {
                    $detector = $attr->name() ;
                    $runs[$runnum][$detector] = 1 ;
                    $detectors[$detector] = 1 ;
                }
            }
        }
        $detector_names = array() ;
        foreach ($detectors as $detector_name => $val) array_push($detector_names, $detector_name) ;
        sort($detector_names) ;
        return array('runs' => $runs, 'detectors' => $detectors, 'detector_names' => $detector_names) ;
    }

    /**
     * Return total counters for the DAQ data recording at a specific range
     * of runs of an experiment
     *
     * The result is returned into the following data structure:
     * 
     *   {  'runs'          : { <run-num>  : { <counter> : <value> } } ,
     *      'counters'      :                { <counter> : <descr> } ,
     *      'counter_names' : [ <counter> , ... ]
     *   }
     * 
     * Note that the first dictionary will tell a caller all counters used
     * for a particular run. Meanwhile the second one - all counters which
     * have ever been used in any run of the specified range (of runs) AS WELL AS
     * description of the counters.
     *
     * @param LogBookExperiment $experiment
     * @param integer $from_runnum
     * @param integer $through_runnum
     * @return array()
     */
    public static function get_daq_counters ($experiment, $from_runnum=null, $through_runnum=null) {
        $runs = array() ;
        $counters = array() ;
        foreach ($experiment->runs() as $run) {

            $runnum = $run->num() ;
            if ($from_runnum    && ($runnum < $from_runnum))    continue ;
            if ($through_runnum && ($runnum > $through_runnum)) continue ;

            $runs[$runnum] = array() ;

            foreach ($run->attributes('DAQ Detector Totals') as $attr) {
                $counter = $attr->name() ;
                $runs[$runnum][$counter] = $attr->val() ;
                $counters[$counter] = $attr->description() ;
            }
        }
        $counter_names = array() ;
        foreach ($counters as $counter_name => $val) array_push($counter_names, $counter_name) ;
        sort($counter_names) ;
        return array('runs' => $runs, 'counters' => $counters, 'counter_names' => $counter_names) ;
    }
    
    public static function get_daq_detectors_new ($experiment, $from_runnum=null, $through_runnum=null) {

        $runs      = array() ;
        $detectors = array () ;

        foreach ($experiment->runs() as $run) {

            $runnum = $run->num() ;
            if ($from_runnum    && ($runnum < $from_runnum))    continue ;
            if ($through_runnum && ($runnum > $through_runnum)) continue ;

            $runs[$runnum] = array() ;

            // ATTENTION: Scan the legacy class 'DAQ_Detectors' for experiments
            //            taken prior Feb 2015 when a transition of the class
            //            name to 'DAQ Detectors' happened.
            //
            foreach (array('DAQ_Detectors', 'DAQ Detectors') as $class) {
                foreach ($run->attributes($class) as $attr) {
                    $detector = $attr->name() ;
                    $runs[$runnum][$detector] = 1 ;
                    $detectors    [$detector] = 1 ;
                }
            }
        }
        $names        = array() ;
        $descriptions = array() ;
        foreach ($detectors as $detector => $val) {
            array_push($names, $detector) ;
            $descriptions[$detector] = $detector ;
        }
        return array (
            'runs'         => $runs ,
            'names'        => $names ,
            'descriptions' => $descriptions
        ) ;
    }

    public static function get_daq_detector_totals ($experiment, $from_runnum=null, $through_runnum=null) {

        $runs      = array() ;
        $detectors = array () ;

        foreach ($experiment->runs() as $run) {

            $runnum = $run->num() ;
            if ($from_runnum    && ($runnum < $from_runnum))    continue ;
            if ($through_runnum && ($runnum > $through_runnum)) continue ;

            $runs[$runnum] = array() ;

            foreach ($run->attributes('DAQ Detector Totals') as $attr) {
                $detector = $attr->name() ;
                $runs[$runnum][$detector] = $attr->val() ;
                $detectors    [$detector] = $attr->description() ;
            }
        }
        $names        = array() ;
        $descriptions = array() ;
        foreach ($detectors as $detector => $descr) {
            array_push($names, $detector) ;
            $descriptions[$detector] = $descr ;
        }
        return array (
            'runs'         => $runs ,
            'names'        => $names ,
            'descriptions' => $descriptions
        ) ;
    }    
}
?>
