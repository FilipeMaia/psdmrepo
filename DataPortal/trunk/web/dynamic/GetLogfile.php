<?php

require_once('AuthDB/AuthDB.inc.php');
require_once('FileMgr/FileMgr.inc.php');

if( !isset( $_GET[ 'id' ] )) die( 'no request identifier parameter found' );
$id = (int)trim( $_GET[ 'id' ] );
if( $id <= 0 ) die( 'invalid request identifier' );

try {
	$authdb = new AuthDB();
	$authdb->begin();

	/* Find the request.
	 */
	$req = FileMgrIfaceCtrlWs::request_by_id( $id );
	if( is_null( $req )) die( "no translation request found for the specified log file" );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
	
    echo FileMgrIfaceCtrlWs::log( $req->log_url );

} catch( AuthDBException $e ) {
	print $e->toHtml();
	exit;
} catch( FileMgrException   $e ) {
	print $e->toHtml();
	exit;
}
?>
