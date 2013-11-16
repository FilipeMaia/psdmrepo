<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookUtils;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

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


$posted_at_instrument = false;
if( isset( $_GET['posted_at_instrument'] ))
    $posted_at_instrument = '0' != trim( $_GET['posted_at_instrument'] );

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

$range_of_runs = '';
if( isset( $_GET['range_of_runs'] ))
    $range_of_runs = trim( $_GET['range_of_runs'] );

$inject_runs = false;
if( isset( $_GET['inject_runs'] ))
    $inject_runs = '0' != trim( $_GET['inject_runs'] );

if( '' != $range_of_runs ) {
	$inject_runs = true;	// Force it because a request was made to search around
							// a specific run or a range of those.

	$posted_at_instrument = false;	// the range of runs isn't compatible with
									// the broader search accross instruments
}

$inject_deleted_messages = isset( $_GET['inject_deleted_messages'] );

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
  "status": {$status_encoded}, "message": {$msg_encoded} ,
  "ResultSet": {
    "Status": {$status_encoded}, "Message": {$msg_encoded}
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
function sort_and_truncate_from_head( &$entries_by_timestamps, $limit ) {

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
    if( 2 == sscanf( $limit, "%d%s", $limit_num, &$unit )) {

    	$nsec_ago = 1000000000 * $limit_num;
    	switch( $unit ) {
    		case 's': break;
    		case 'm': $nsec_ago *=            60; break;
    		case 'h': $nsec_ago *=          3600; break;
    		case 'd': $nsec_ago *=     24 * 3600; break;
    		case 'w': $nsec_ago *= 7 * 24 * 3600; break;
    		default:
    			report_error( "illegal format of the limit parameter" );
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

/* Proceed with the operation
 */
try {
    LogBook::instance()->begin();

    /* Make adjustments relative to the primary experiment of the search.
     */
   	$experiment = LogBook::instance()->find_experiment_by_id( $id );
   	if( is_null( $experiment)) report_error( "no such experiment" );

   	/* Make sure the 'range_of_runs' parameters isn't used along with
   	 * explicitly specified 'begin' or 'end' time limits. A reason for that
   	 * is that we're going to redefine these limits for the range of runs.
   	 * The new limits will include all messages posted:
   	 * 
   	 *   - after the end of the previous to the first run. If no such previous run
   	 *   exist then all messages before the begin time of the first run will be
   	 *   selected.
   	 *   - before the beginning of the next to the last run. Of no such next run
   	 *   is found then all messages till the end of the logbook will be selected.
   	 */
   	$run_specific_entries = array();
   	if( '' != $range_of_runs ) {
   		if(( '' != $begin_str ) or ( '' != $end_str ))
	   		report_error( "begin/end time limits can't be used together with the run limi" );

	   	/* Pasrse the run numbers first. If the parse succeeds and no last run
	   	 * is provided then assume the second run as the last one.
	   	 */
	   	if( strpos($range_of_runs, '-') === false ) {
	   		$r1 = $range_of_runs;
	   		$r2 = '';
	   	} else {
			list($r1,$r2) = explode( '-', $range_of_runs, 2 );
	   	}
		$r1 = trim( $r1 );
		$r2 = trim( $r2 );
		if( '' == $r1 ) report_error( "syntax error in the range of runs" );

		$first_run_num = null;
		if(( 1 != sscanf( $r1, "%d", $first_run_num )) or ( $first_run_num <= 0 ))
			report_error( "syntax error in the first run number of the range" );

		$last_run_num = $first_run_num;
		if( '' != $r2 )
			if(( 1 != sscanf( $r2, "%d", $last_run_num )) or ( $last_run_num <= 0 ))
				report_error( "syntax error in the last run number of the range" );

		if( $last_run_num < $first_run_num ) report_error( "last run in the range can't be less than the first one" );

		$first_run = $experiment->find_run_by_num( $first_run_num );
		if( is_null( $first_run )) report_error( "run {$first_run_num} can't be found" );
		$last_run = $experiment->find_run_by_num( $last_run_num );
		if( is_null( $last_run )) report_error( "run {$last_run_num} can't be found" );

		$previous_run = $experiment->find_run_by_num( $first_run_num - 1 );
		if( !is_null( $previous_run )) {

			// ATTENTION: If the previous run has never been explicitly closed
			//            then it's okay to use its begin time.
			//
			$begin_str =
				is_null($previous_run->end_time()) ?
				$previous_run->begin_time()->toStringShort() :
				$previous_run->end_time()->toStringShort();
		}
		$begin_str = $first_run->begin_time()->toStringShort();
		$next_run = $experiment->find_run_by_num( $last_run_num + 1 );
		if( is_null( $next_run )) {

			// ATTENTION: If no next run exists then, depending on the current status
			//            of the requested last run in the range we assume:
			//
			//            o  +1 hr after the end of that run if the run is still open
			//               and the run's duration doesn't exceed 24 hours (so called "run-away"
			//               run).
			//
			//            o  no end time otherwise (and assumption is that the experiment
			//               is still taking data)
			//
			//            These measures will protect us from seein the whole tail of e-log messages
			//            posted after the end of the last run.
			//
			$last_run_end_time = $last_run->end_time();
			if( !is_null( $last_run_end_time ) && ( LusiTime::now()->sec - $last_run_end_time->sec > 24*3600 )) {
				$end_str = new LusiTime( $last_run_end_time->sec + 3600, 0 );
			}
		} else {
			$end_str = $next_run->begin_time()->toStringShort();
		}

		/* Find messages which are explicitly or implicitly associated with runs, no matter
		 * when those messages were posted. The messages (if any) will be automatically
		 * mixed into the output result.
		 * 
		 * Note the meaning of the associations:
		 * 
		 * 'explicit' - these messages were posted (replied) for the specified run,
		 *              so that there will be a database association
		 *
		 * 'implicit' - these messages have (case insensitive) 'run NNN' in the message
		 *              body. So we have to use full text search to find them.
		 */
		for( $num = $first_run_num; $num <= $last_run_num; ++$num ) {
			$run = $experiment->find_run_by_num( $num);
			if( is_null( $run )) continue;

			// Explicit
			//
			foreach( $experiment->entries_of_run( $run->id(), $inject_deleted_messages ) as $e ) array_push( $run_specific_entries, $e );

			// Implicit
			//
    		foreach( $experiment->search(
        		null,			// $shift_id
        		null,			// $run_id
        		"run {$num} ",	// $text2search
        		true, 			// $search_in_messages,
        		false,			// $search_in_tags,
        		false, 			// $search_in_values,
        		true,			// $posted_at_experiment,
        		false,			// $posted_at_shifts,
        		false, 			// $posted_at_runs,
        		null,			// $begin,
        		null, 			// $end,
        		null,			// $tag,
        		null,			// $author,
        		null,			// $since
        		null,			// $limit
        		$inject_deleted_messages
        		) as $e ) array_push( $run_specific_entries, $e	);
		}
   	}

	/* Timestamps are translated here because of possible shoftcuts which
	 * may reffer to the experiment's validity limits.
	 */
    $begin = null;
   	if( $begin_str != '' ) {
       	$begin = translate_time( $experiment, $begin_str );
       	if( is_null( $begin )) report_error( "begin time has invalid format" );
    }
    $end = null;
    if( $end_str != '' ) {
       	$end = translate_time( $experiment, $end_str );
       	if( is_null( $end )) report_error( "end time has invalid format" );
	}
    if( !is_null( $begin ) && !is_null( $end ) && !$begin->less( $end ))
       	report_error( "invalid interval - begin time isn't strictly less than the end one" );        
        	
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
    $since = !$since_str ? null : LusiTime::from64( $since_str );

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

	        report_error( 'not authorized to read messages for the experiment' );
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
                $e->id() == $id ? $shift_id : null,	// the parameter makes sense for the main experiment only
                $e->id() == $id ? $run_id   : null,	// ditto
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

		$runs = !$inject_runs ? array() : $e->runs_in_interval( $begin4runs, $end4runs/*, $limit*/ );

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
        	if( is_null( $begin4runs ) || $begin4runs->less( $r->begin_time())) {
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
    /* Merge in the run-specific entries. Make sure there won't be any duplicates.
     * The duplicates can be easily identified becouase they would be within
     * teh same timestamps.
     */
    foreach( $run_specific_entries as $e ) {
    	$skip = false;
    	$t = $e->insert_time()->to64();
    	if( array_key_exists( $t, $entries_by_timestamps )) {
    		foreach( $entries_by_timestamps[$t] as $pair )
    			if(( $pair['type'] == 'entry' ) && ( $pair['object']->id() == $e->id())) {
    				$skip = true;
    				break;
    			}
    	} else {
    		$entries_by_timestamps[$t] = array();
    	}
    	if( $skip ) continue;
    	array_push(
   			$entries_by_timestamps[$t],
   			array(
   				'type'   => 'entry',
   				'object' => $e
    		)
   		);
    }

    /* Now produce the desired output.
     */
    $timestamps = sort_and_truncate_from_head( $entries_by_timestamps, $limit );

    $status_encoded  = json_encode( "success" );
   	$updated_encoded = json_encode( LusiTime::now()->toStringShort());

   	$result =<<< HERE
{
  "status": {$status_encoded},
  "updated": {$updated_encoded},
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

    print $result;

    LogBook::instance()->commit();

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }

?>

