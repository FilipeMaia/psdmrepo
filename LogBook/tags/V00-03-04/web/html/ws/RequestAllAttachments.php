<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( '<b><em style="color:red;" >Error:</em></b>&nbsp;'.$msg );
    print <<< HERE
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
HERE;
    exit;
}

/*
 * This script will process a request for retreiving all attachments of all messages
 * posted in the experiments. Return a JSON object with the descriptions of attachments.
 * Othersise return another JSON object with an explanation of a problem.
 */
if( !isset( $_GET['exper_id'] )) report_error('no experiment identifier parameter found in the request');
$exper_id = trim( $_GET['exper_id'] );
if( $exper_id == '' ) report_error( 'experiment identifier found in the request is empty' );

function sort_by_time_and_merge( $runs, $attachments ) {
	$result = array();
    $title = "open the run in a separate browser tab";
	foreach( $runs as $r ) {
    	$run_url = <<<HERE
<a href="../logbook/?action=select_run_by_id&id={$r->id()}" target="_blank" title="{$title}" class="lb_link">{$r->num()}</a>
HERE;
		array_push(
			$result,
			array(
				'time64'  => $r->begin_time()->to64(),
				'type'    => 'r',
				'r_url'   => $run_url,
				'r_id'    => $r->id(),
				'r_num'   => $r->num(),
				'r_begin' => $r->begin_time()->toStringShort(),
				'r_end'   => is_null($r->end_time()) ? '' : $r->end_time()->toStringShort()
			)
		);
	}
    $title = "open the message in a separate browser tab";
	foreach( $attachments as $a ) {

    	$attachment_url = <<<HERE
<a href="../logbook/attachments/{$a->id()}/{$a->description()}" target="_blank" title="{$title}" class="lb_link">{$a->description()}</a>
HERE;

    	$entry_url = <<<HERE
<a href="../logbook/index.php?action=select_message&id={$a->parent()->id()}" target="_blank" title="{$title}" class="lb_link">{$a->parent()->id()}</a>
HERE;
		array_push(
			$result,
			array(
				'time64'    => $a->parent()->insert_time()->to64(),
				'type'      => 'a',
        		'e_url'     => $entry_url,
        		'e_time'    => $a->parent()->insert_time()->toStringShort(),
        		'e_time_64' => $a->parent()->insert_time()->to64(),
        		'e_author'  => $a->parent()->author(),
        		'a_id'      => $a->id(),
        		'a_name'    => $a->description(),
            	'a_url'     => $attachment_url,
            	'a_size'    => $a->document_size(),
            	'a_type'    => $a->document_type()
			)
		);
	}
	usort(
		$result,
		function($a,$b) {
			return $a['time64'] - $b['time64'];
		}
	);
	return $result;
}

/*
 * Return JSON objects with a list of attachments.
 */
try {

    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $exper_id ) or report_error( 'no such experiment' );

    if( !LogBookAuth::instance()->canRead( $experiment->id())) report_error( 'You are not authorized to access any information about the experiment' );

    function find_and_process_entries( &$attachments, $entries ) {
	    foreach( $entries as $e ) {
	   		foreach( $e->attachments() as $a ) array_push( $attachments, $a );
	   		find_and_process_entries( $attachments, $e->children());
	    }
    }
    $runs = $experiment->runs();
    $attachments = array();
    find_and_process_entries( $attachments, $experiment->entries());

    $result = sort_by_time_and_merge( $runs, $attachments );

    $status_encoded = json_encode( "success" );
   	$updated_encoded = json_encode( LusiTime::now()->toStringShort());
    
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

   	print <<< HERE
{
  "Status": {$status_encoded},
  "Updated": {$updated_encoded},
  "Attachments": [

HERE;


    $first = true;
    foreach( $result as $r ) {
		if( $first ) $first = false;
		else echo ',';
		echo json_encode( $r );
    }
    print <<< HERE
  ]
}
HERE;

    LogBook::instance()->commit();

} catch( LogBookException  $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException $e ) { report_error( $e->toHtml()); }

?>
