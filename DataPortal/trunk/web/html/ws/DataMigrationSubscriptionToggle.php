<?php

/* The script will toggle subscriptions of the authenticated user to receive
 * e-mail notifications on delayed file migration.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\Config;
use DataPortal\DataPortalException;

try {
	$config  = Config::instance();
	$config->begin();

	$authdb = AuthDB::instance();
	$authdb->begin();

	$subscriber = $authdb->authName();
	$address    = $subscriber.'@slac.stanford.edu';

	$config->subscribe4migration_if (
		is_null( $config->check_if_subscribed4migration ( $subscriber, $address )),
		$subscriber,
		$address );
	$config->commit();

} catch ( AuthDBException     $e ) { print $e->toHtml(); }
  catch ( DataPortalException $e ) { print $e->toHtml(); }

?>