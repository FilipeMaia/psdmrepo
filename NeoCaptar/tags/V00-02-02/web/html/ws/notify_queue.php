<?php

/**
 * This service will perform varios operations on entries in the notification
 * queue and return an updated notification lists.
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

    $action  = strtolower(trim(NeoCaptarUtils::get_param_GET('action')));
    $id      = NeoCaptarUtils::get_param_GET('id', false, false);

    $authdb = AuthDB::instance();
    $authdb->begin();

    $neocaptar = NeoCaptar::instance();
    $neocaptar->begin();

    if(is_null($id)) {
        foreach($neocaptar->notify_queue() as $entry) {
            $id = $entry->id();
            switch($action) {
                case 'submit': $neocaptar->submit_notification_event($id); break;
                case 'delete': $neocaptar->delete_notification_event($id); break;
                default:       NeoCaptarUtils::report_error('operation is not implemented for action: '.$action.' and id: '.$id);
            }
        }
    } else {
        switch($action) {
            case 'submit': $neocaptar->submit_notification_event($id); break;
            case 'delete': $neocaptar->delete_notification_event($id); break;
            default:       NeoCaptarUtils::report_error('operation is not implemented for action: '.$action.' and id: '.$id);
        }
    }
    $notificatons2return = NeoCaptarUtils::notifications2array($neocaptar);

    $authdb->commit();
    $neocaptar->commit();

    NeoCaptarUtils::report_success($notificatons2return);

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
