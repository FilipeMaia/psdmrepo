<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookUtils;
use LogBook\LogBookException;
use LusiTime\LusiTime;

/*
 * This script will perform the search for a single free-form entry in a scope
 * of an experiment using a numer identifier of the entry. The result is returned
 * as a JSON object which in case of success will have the following format:
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

$id = null ;
if (isset($_GET['id'])) {
    $id = intval(trim($_GET['id'])) ;
    if (!$id) report_error("the value of <id> is either empty or invalid") ;
}
$run_num = null ;
$exper_id = null ;
if (isset( $_GET['run_num'])) {
    $run_num = intval(trim($_GET['run_num'])) ;
    if (!$run_num) report_error("the value of <run_num> is either empty or invalid") ;
    if (!isset( $_GET['exper_id'])) report_error("the <exper_id> paameter is required along with <run_num>") ;
    $exper_id = intval(trim($_GET['exper_id'])) ;
    if (!$exper_id) report_error("the value of <exper_id> is either empty or invalid") ;
}
if (is_null($id) && is_null($run_num)) report_error( "no valid <id> or <run_num> parameter provided") ;

$show_in_vicinity = false;
if( isset($_GET['show_in_vicinity'])) {
    $show_in_vicinity = intval(trim($_GET['show_in_vicinity'])) ? true : false;
}

/* Package the error message into a JSON object and return the one
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

/* Proceed with the operation
 */
try {
    LogBook::instance()->begin() ;

    if ($id) {

        $entry = LogBook::instance()->find_entry_by_id($id) or report_error("no such message entry") ;
        $experiment = $entry->parent() ;
        if (!LogBookAuth::instance()->canRead($experiment->id()))
            report_error('not authorized to read messages for the experiment') ;

        if ($show_in_vicinity) {
            print LogBookUtils::search_around_message($entry->id(), 'report_error') ;
        } else {
            $now_encoded = json_encode(LusiTime::now()->toStringShort()) ;
            $status_encoded = json_encode("success") ;
            $result =<<< HERE
{
    "status": {$status_encoded}, "updated": {$now_encoded},
    "ResultSet": {
      "Status": {$status_encoded}, "Updated": {$now_encoded},
      "Result": [
HERE;
          $result .= "\n".LogBookUtils::entry2json($entry) ;
          $result .=<<< HERE
   ] } }
HERE;
            print $result ;
        }
 
    } elseif ($run_num) {

        $experiment = LogBook::instance()->find_experiment_by_id($exper_id) or report_error("no such experiment") ;
        $run = $experiment->find_run_by_num($run_num) or report_error("no such run entry") ;
        if (!LogBookAuth::instance()->canRead($experiment->id()))
            report_error('not authorized to read messages for the experiment') ;

        if ($show_in_vicinity) {
            print LogBookUtils::search_around_run($run->id(), 'report_error') ;
        } else {
            $now_encoded = json_encode(LusiTime::now()->toStringShort()) ;
            $status_encoded = json_encode("success") ;
            $result =<<< HERE
{
    "status": {$status_encoded}, "updated": {$now_encoded},
    "ResultSet": {
      "Status": {$status_encoded}, "Updated": {$now_encoded},
      "Result": [
HERE;
          $result .= "\n".LogBookUtils::run2json($run, 'run') ;
          $result .=<<< HERE
   ] } }
HERE;
            print $result ;
        }
    }
    LogBook::instance()->commit() ;

} catch (LogBookException $e) { report_error($e->toHtml()) ; }

?>