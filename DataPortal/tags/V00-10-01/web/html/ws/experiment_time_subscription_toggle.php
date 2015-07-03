<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\ExpTimeMon;
use DataPortal\DataPortalException;

use LusiTime\LusiTimeException;


/**
 * This service will togle current user's subscription for notifications
 * posted when downtime explanation comments are posted to the database.
 */
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

    header( 'Content-type: application/json' );
    header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
    header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

    echo json_encode($result);
    exit;
}

try {
    $authdb = AuthDB::instance();
    $authdb->begin();

    $exptimemon = ExpTimeMon::instance();
    $exptimemon->begin();

    $subscriber = $authdb->authName();
    $address    = $subscriber.'@slac.stanford.edu';

    $exptimemon->subscribe4explanations_if(
        is_null( $exptimemon->check_if_subscribed4explanations( $subscriber, $address )),
        $subscriber,
        $address );

    $authdb->commit();
    $exptimemon->commit();

    report_success(array('subscribed' => 0));

} catch( AuthDBException     $e ) { report_error( $e->toHtml()); }
  catch( DataPortalException $e ) { report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { report_error( $e->toHtml()); }
  catch( Exception           $e ) { report_error( "{$e}" );      }
  
?>
