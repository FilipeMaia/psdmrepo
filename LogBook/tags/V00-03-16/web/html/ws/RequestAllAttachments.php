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
  "Status": {$status_encoded}, "Message": {$msg_encoded} ,
  "status": {$status_encoded}, "message": {$msg_encoded}
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
    $title4run = "show the run in the e-Log Search panel within the current Portal";
    foreach( $runs as $r ) {
        $run_url = <<<HERE
<a href="javascript:global_elog_search_run_by_num({$r->num()},true);" title="{$title4run}" class="lb_link"><img src="../portal/img/link2run_f0f0f0_32x32.png" /></a>
HERE;
        $run_url_1 = <<<HERE
<a href="javascript:global_elog_search_run_by_num({$r->num()},true);" title="{$title4run}" class="lb_link"><img src="../portal/img/link2run_32x32.png" /></a>
HERE;

        array_push(
            $result,
            array(
                'time64'  => $r->begin_time()->to64(),
                'type'    => 'r',
                'r_url'   => $run_url,
                'r_url_1' => $run_url_1,
                'r_id'    => $r->id(),
                'r_num'   => $r->num(),
                'r_begin' => $r->begin_time()->toStringShort(),
                'r_end'   => is_null($r->end_time()) ? '' : $r->end_time()->toStringShort()
            )
        );
    }
    $title = "show the message in the e-Log Search panel within the current Portal";
    foreach( $attachments as $a ) {

        $attachment_url = <<<HERE
<a href="../logbook/attachments/{$a->id()}/{$a->description()}" target="_blank" title="{$title}" class="lb_link">{$a->description()}</a>
HERE;
        $entry_url = <<<HERE
<a href="javascript:global_elog_search_message_by_id({$a->parent()->id()},true);" title="{$title}" class="lb_link">{$a->parent()->id()}</a>
HERE;
        $entry_link_url = <<<HERE
<a href="javascript:global_elog_search_message_by_id({$a->parent()->id()},true);" title="{$title}" class="lb_link"><img src="../portal/img/link2message_32by32.png" /></a>
HERE;
        array_push(
            $result,
            array(
                'time64'    => $a->parent()->insert_time()->to64(),
                'type'      => 'a',
                'e_id'      => $a->parent()->id(),
                'e_url'     => $entry_url,
                'e_link_url'=> $entry_link_url,
                'e_time'    => $a->parent()->insert_time()->toStringShort(),
                'e_time_64' => $a->parent()->insert_time()->to64(),
                'e_author'  => $a->parent()->author(),
                'a_id'      => $a->id(),
                'a_name'    => $a->description(),
                'a_url'     => $attachment_url,
                'a_size'    => $a->document_size(),
                'a_type'    => $a->document_type(),
                'entry_id'  => $a->parent()->id()
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
  "status": {$status_encoded},
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
