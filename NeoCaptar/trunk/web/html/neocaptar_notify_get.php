<?php

/**
 * This service will return notification lists.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $event_types = array();
    foreach( $neocaptar->notify_event_types() as $e ) {
        $recipient = $e->recipient();
        if( !array_key_exists($recipient,$event_types)) $event_types[$recipient] = array();
        array_push(
            $event_types[$recipient],
            array(
                'name'        => $e->name(),
                'description' => $e->description()
            )
        );
    }
    $notifications2array = array(
        'myself' => array(
            'uid' => 'gapon',
            'on_any'              => true,
            'on_project_assign'   => true,
            'on_project_deassign' => true,
            'on_project_delete'   => true,
            'on_cable_create'     => true,
            'on_cable_delete'     => true,
            'on_cable_edit'       => true,
            'on_register'         => true,
            'on_label'            => true,
            'on_fabrication'      => true,
            'on_ready'            => true,
            'on_install'          => true,
            'on_commission'       => true,
            'on_damage'           => true,
            'on_retire'           => true
        ),
        'others' => array(
            array(
                'uid'               => 'gapon',
                'on_project_create' => true,
                'on_fabrication'    => false,
                'on_ready'          => false,
                'on_install'        => false,
                'on_commission'     => false,
                'on_damage'         => false,
                'on_retire'         => false
            ),
            array(
                'uid'               => 'salnikov',
                'on_project_create' => true,
                'on_fabrication'    => false,
                'on_ready'          => false,
                'on_installed'      => false,
                'on_commissioned'   => false,
                'on_damaged'        => false,
                'on_retired'        => false
            )
        )
    );

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(
        array(
            'event_types' => $event_types,
            'notify' => $notifications2array ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
