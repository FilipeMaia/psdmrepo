<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

use LogBook\LogBookException;

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
	public static function child2json( $entry, $posted_at_instrument ) {
	
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
	    foreach( $entry->children() as $child )
	        array_push( $children_ids, LogBookUtils::child2json( $child, $posted_at_instrument ));
	
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
	        	"hms"             => $timestamp->toStringHMS()
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
	public static function entry2json( $entry, $posted_at_instrument ) {
	
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
	    foreach( $entry->children() as $child )
	        array_push( $children_ids, LogBookUtils::child2json( $child, $posted_at_instrument ));
	
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
	        	"hms"             => $timestamp->toStringHMS()
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
	public static function run2json( $run, $type, $posted_at_instrument ) {
	
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
}
?>
