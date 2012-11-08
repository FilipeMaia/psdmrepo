<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\ExpTimeMon;
use DataPortal\DataPortalException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;


/**
 * This service will save a comment for the specified gap.
 */

header( 'Content-type: application/json' );
header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

function report_error($msg) {
    return_result(
        array(
            'status' => 'error',
            'message' => $msg
        )
    );
}
function report_success($result) {
    $result['status'] = 'success';
    return_result($result);
}
function return_result($result) {
    echo json_encode($result);
    exit;
}

try {
    $authdb = AuthDB::instance();
    $authdb->begin();

    if( !$authdb->hasRole($authdb->authName(),null,'BeamTimeMonitor','Editor')) report_error('not authorized for this this service');

    // Process input parameters first
    //
    if( !isset($_POST[ 'gap_begin_time_64'] )) report_error( 'no gap id parameter found' );
    $gap_begin_time_64 = intval(trim( $_POST['gap_begin_time_64']));
    $gap_begin_time = LusiTime::from64($gap_begin_time_64);

    if( !isset($_POST[ 'instr_name'] )) report_error( 'no instrumenet name parameter found' );
    $instr_name = trim( $_POST['instr_name']);

    if( !isset($_POST[ 'comment'] )) report_error( 'no gap comment parameter found' );
    $comment_text = trim( $_POST['comment']);

    if( !isset($_POST[ 'system'] )) report_error( 'no system comment parameter found' );
    $system_text = trim( $_POST['system']);
 
    $exptimemon = ExpTimeMon::instance();
    $exptimemon->begin();
    if( $comment_text == '' )
        $exptimemon->beamtime_clear_gap_comment($gap_begin_time, $instr_name);
    else
        $exptimemon->beamtime_set_gap_comment(
            $gap_begin_time,
            $instr_name,
            $comment_text,
            $system_text,
            LusiTime::now(),
            AuthDB::instance()->authName());

    $exptimemon->notify_allsubscribed4explanations($instr_name, $gap_begin_time);

    $comment = $exptimemon->beamtime_comment_at($gap_begin_time, $instr_name);
    $comment_info = is_null($comment) ?
        array('available'     => 0) :
        array('available'     => 1,
              'instr_name'    => $comment->instr_name(),
              'comment'       => $comment->comment(),
              'system'        => $comment->system(),
              'posted_by_uid' => $comment->posted_by_uid(),
              'post_time'     => $comment->post_time()->toStringShort());

    $authdb->commit();
    $exptimemon->commit();

    report_success(array('comment' => $comment_info));

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( DataPortalException $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( Exception           $e ) { report_error( "{$e}" );      }
  
?>
