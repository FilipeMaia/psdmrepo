<?php

/**
 * This service will save requsted notification change and return
 * an updated notification lists.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTimeException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $recipient  = NeoCaptarUtils::get_param_GET('recipient');
    $policy     = NeoCaptarUtils::get_param_GET('policy', false, false);
    $uid        = null;
    $event_name = null;
    $enabled    = null;
    if( is_null($policy)) {
        $uid        = NeoCaptarUtils::get_param_GET('uid');
        $event_name = NeoCaptarUtils::get_param_GET('event_name');
        $enabled    = NeoCaptarUtils::get_param_GET('enabled');
    }
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    if( is_null($policy)) {
        $event_type = $neocaptar->find_notify_event_type($recipient, $event_name);
        if( is_null($event_type)) NeoCaptarUtils::report_error('unknown notification event');

        $notification = $neocaptar->find_notification($uid, $event_type->id());
        if( is_null($notification)) $notification = $neocaptar->add_notification   ($uid, $event_type->id(), $enabled);
                                    $notification = $neocaptar->update_notification($notification->id(), $enabled);
    } else {
        $neocaptar->update_notification_schedule($recipient,$policy);
    }
    $notificatons2return = NeoCaptarUtils::notifications2array($neocaptar);

    $authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success($notificatons2return);

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
