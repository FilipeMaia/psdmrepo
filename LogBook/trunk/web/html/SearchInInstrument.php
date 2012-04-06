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
 * of all experiments of the specified instrument using values of specified
 * search criterias. The result is returned as a JSON obejct which in case
 * of success will have the following format:
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

if( !isset( $_GET['instr'] )) report_error( "no valid instrument name found among script parameters" );
$instr = trim( $_GET['instr'] );
if( $instr == '' ) report_error( "instrument name can't be empty" );

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

$inject_deleted_messages = isset( $_GET['inject_deleted_messages'] );

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

/* Translate timestamps which can be specified either in human-readable format
 * or as 64-bit timestamps.
 */
function translate_time( $experiment, $str ) {
    $str_trimmed = trim( $str );
    if( $str_trimmed == '' ) return null;
    $result = LusiTime::parse( $str_trimmed );
    if( is_null( $result )) $result = LusiTime::from64( $str_trimmed );
    return $result;
}

/* The functon will produce a sorted list of _unique_ timestamps based
 * on keys of the input dictionary.
 */
function sort_timestamps( $entries_by_timestamps ) {
	$timestamps = array_keys( $entries_by_timestamps );
    sort( $timestamps, SORT_NUMERIC );
    $timestamps = array_unique( $timestamps );
    return $timestamps;
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    /* Make adjustments relative to the primary experiment of the search.
     */
   	$experiment = $logbook->find_experiment_by_id( $id );
   	if( is_null( $experiment)) report_error( "no such experiment" );

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

    /* Mix entries found in all experiment of the instrument in the right order.
     * Results will be merged into this dictionary before returning to the client.
     */
    $entries_by_timestamps = array();

    /* Scan all relevant experiments.
     */
    foreach( $logbook->experiments_for_instrument( $experiment->instrument()->name()) as $e ) {

    	/* Check for the authorization. Silently skip this experiemnt if not allowed
         * to read it.
    	 */
    	if( !LogBookAuth::instance()->canRead( $e->id())) continue;
 
    	/* Get the info for entries.
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
                null,   /* $shift_id */
                null,   /* $run_id */
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
                null, /* $since */
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

		/* Merge both results into the dictionary for further processing.
		 */
    	foreach( $entries as $e ) {
    		$t = $e->insert_time()->to64();
    		if( !array_key_exists( $t, $entries_by_timestamps )) $entries_by_timestamps[$t] = array();
    		array_push(
    			$entries_by_timestamps[$t],
    			array(
    				'object' => $e
    			)
    		);
    	}
    }

    /* Now produce the desired output.
     */
    $timestamps = sort_timestamps( $entries_by_timestamps );

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
    		$entry = $pair['object'];
            if( $first ) {
                $first = false;
                $result .= "\n".LogBookUtils::entry2json( $entry, true /* $posted_at_instrument */, $inject_deleted_messages );
            } else {
                $result .= ",\n".LogBookUtils::entry2json( $entry, true /* $posted_at_instrument */, $inject_deleted_messages );
            }
    	}
    }
    $result .=<<< HERE
 ] } }
HERE;

    print $result;

    $logbook->commit();

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }

?>

