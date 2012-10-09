<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned as JSON object (array).
 */
if( !isset( $_GET['type'] )) die( "no valid information type in the request" );
$type = trim( $_GET['type'] );

define( 'BROWSE_ZONES',    1 );
define( 'BROWSE_CATALOGS', 2 );

if( !isset( $_GET['path'] )) die( "no valid path in the request" );
$path = trim( $_GET['path'] );

$path = '';
switch( $type ) {
    case BROWSE_ZONES:
        break;
    case BROWSE_CATALOGS:
        if( !isset( $_GET['path'] )) die( "no valid path in the request" );
        $path = trim( $_GET['path'] );
        break;
    default:
        die( 'unknown type of the request: '.$type );
}

function catalog2json( $c ) {

    /* Check if this catalog has child collections
     */
    $catalogs = null;
    FileMgrIrodsWs::collections( $catalogs, $c->name );
    
    /* Return a 'context' for the catalog to be displayed
     */
    $context = '';
    foreach( explode( '/', $c->name ) as $s ) {
        if( $s != '' ) $context .= '&nbsp;/&nbsp;'.$s;
    }
    return json_encode(
        array (
            "label"      => '<em style="color:#0071bc;"><b>'.basename( $c->name ).'</b></em>',
            "expanded"   => false,
            "isLeaf"     => count( $catalogs ) < 1,
            "type"       => BROWSE_CATALOGS,
            "title"      => "click to see nested iRODS catalogs",
            "path"       => $c->name,
            "context"    => $context
        )
    );
}

/*
 * Analyze and process the request
 */
try {

    AuthDB::instance()->begin();

    header( "Content-type: application/json" );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;

    $first = true;
    $catalogs = null;
    FileMgrIrodsWs::collections( $catalogs, $path );
    foreach( $catalogs as $c ) {
        if( $first ) {
            $first = false;
            echo "\n".catalog2json( $c );
        } else {
            echo ",\n".catalog2json( $c );
        }
    }
    print <<< HERE
 ] } }
HERE;

    AuthDB::instance()->commit();

} catch (AuthDBException  $e) { print $e->toHtml(); }
  catch (FileMgrException $e) { print $e->toHtml(); }

?>
