<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>

  <head>
    <title>iRODS Catalkog Browser</title>
    <meta http-equiv="Pragma" content="no-cache"> 
    <meta http-equiv="Pragma-directive" content="no-cache"> 
    <meta http-equiv="cache-directive" content="no-cache"> 
    <meta http-equiv="Expires" content="0">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <!--
    <link type="text/css" rel="stylesheet" href="style.css" />
    -->
    <style type="text/css">
    </style>
  </head>

  <body>

<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

$path = '';
if( isset( $_GET['path'] )) $path = $_GET['path'];

$base_url = 'https://pswww.slac.stanford.edu/ws-auth/irodsws/files';

$url = $base_url.$path;

$opts = array(
    "timeout"      => 1,
    "httpauthtype" => HTTP_AUTH_BASIC,
    "httpauth"     => "gapon:newlife2"
);
$info = null;
$response = http_get($url, $opts, $info );

if( $info['response_code'] != 200 )
    die( "failed to get the catalog from Web service" );

$response_parsed = http_parse_message( $response );

$str = ''.$response_parsed->body;  // Promote it to a string. THis is needed because the body is returned
                                   // as stdClass which can't be JSON decoded.
$a = json_decode( $str );


# -------------------------
# Show collections (if any)
# -------------------------

$num_collections = 0;
foreach( $a as $e ) {
    if( $e->type != 'collection' ) continue;
    if( $e->name == '/' ) continue;
    $num_collections++;
}
if( $num_collections ) {
    print <<<HERE
<div id="dirs">
<table class="table"><tbody>
  <tr>
    <td class="first_col_hdr">Collection</td>
    <td class="col_hdr">Collections</td>
    <td class="col_hdr">Files</td>
    <td class="col_hdr">Size [GB]</td>
  </tr>
  <tr>
    <td><div class="first_separator"></div></td>
    <td><div class="separator"></div></td>
    <td><div class="separator"></div></td>
    <td><div class="separator"></div></td>
  </tr>
HERE;
    foreach( $a as $e ) {
        if( $e->type != 'collection' ) continue;
        if( $e->name == '/' ) continue;

        $info = get_collection_info( $base_url, $e->name ) or
            die( "failed to get collection info: {$e->name}" );
        $collections = $info['collections'];
        $files = $info['files'];
        $size  = $info['size'];
        $last_slash_pos = strrpos( $e->name, '/' );
        if( $last_slash_pos === false )
            die( 'invalid value of path passed into the operation' );
        $collection = substr( $e->name, $last_slash_pos+1 );
        print <<<HERE
  <tr>
    <td class="first_col_row"><a href="BrowseFMCatalogs.php?path={$e->name}">{$collection}</a></td>
    <td class="col_row">{$collections}</td>
    <td class="col_row">{$files}</td>
    <td class="col_row" align="right">{$size}</td>
  </tr>
HERE;
    }
    print <<<HERE
</tbody></table>
</div>
HERE;

}

# -------------------
# Show files (if any)
# -------------------

define( GB, 1.0*1024*1024*1024 );

$resources = array();
$num_files = 0;
foreach( $a as $e ) {
    if( $e->type == 'collection' ) continue;
    $num_files++;
    if( !array_key_exists ( $e->resource, $resources )) {
        $resources[$e->resource] = array( 'files' => 0, 'size' => 0.0 );
    }
    $resources[$e->resource]['files'] += 1;
    $resources[$e->resource]['size'] += $e->size / GB;
}
if( $num_files ) {
    print <<<HERE
<div id="files">
  <table class="table"><tbody>
    <tr>
      <td class="first_col_hdr">Storage</td>
      <td class="col_hdr">Files</td>
      <td class="col_hdr">Size [GB]</td>
    </tr>
    <tr>
      <td><div class="first_separator"></div></td>
      <td><div class="separator"></div></td>
      <td><div class="separator"></div></td>
    </tr>
HERE;
    foreach( array_keys( $resources ) as $resource ) {
        $files = $resources[$resource]['files'];
        $size = sprintf( "%0.2f", $resources[$resource]['size'] );
        $resource_url =<<<HERE
<a href="resource.php?resource={$resource}">$resource</a>
HERE;
        print <<<HERE
    <tr>
      <td class="first_col_row">{$resource_url}</td>
      <td class="col_row">{$files}</td>
      <td class="col_row" align="right">{$size}</td>
    </tr>
HERE;
    }
	print <<<HERE
  </tbody></table>
<br>
<table class="table"><tbody>
  <tr>
    <td class="first_col_hdr">File</td>
    <td class="col_hdr">Owner</td>
    <td class="col_hdr">Size</td>
    <td class="col_hdr">Modified</td>
    <td class="col_hdr">Resource</td>
    <td class="col_hdr">Physical location</td>
  </tr>
  <tr>
    <td><div class="first_separator"></div></td>
    <td><div class="separator"></div></td>
    <td><div class="separator"></div></td>
    <td><div class="separator"></div></td>
    <td><div class="separator"></div></td>
    <td><div class="separator"></div></td>
  </tr>
HERE;
    foreach( $a as $e ) {
        if( $e->type == 'collection' ) continue;
    
        $style = $e->resource == 'hpss-resc' ?  'style="color:#c0c0c0;"' : '';
        $mtime_str = date( "Y-m-d H-i-s", $e->ctime );
        print <<<HERE
  <tr>
    <td class="first_col_row"><span {$style}>{$e->name}</span></td>
    <td class="col_row"><span {$style}>{$e->owner}</span></td>
    <td class="col_row" align="right"><span {$style}>{$e->size}</span></td>
    <td class="col_row"><span {$style}>{$mtime_str}</span></td>
    <td class="col_row"><span {$style}>{$e->resource}</span></td>
    <td class="col_row"><span {$style}>{$e->path}</span></td>
    </tr>
HERE;
    }
    print <<<HERE
</tbody></table>
</div>
HERE;
}

function get_collection_info( $base_url, $path ) {
  
    $url = $base_url.$path;
    $opts = array(
        "timeout"      => 1,
        "httpauthtype" => HTTP_AUTH_BASIC,
        "httpauth"     => "gapon:newlife2"
    );
    $info = null;
    $response = http_get($url, $opts, $info );
    
    if( $info['response_code'] != 200 )
        die( "failed to get the catalog from Web service" );
    
    $response_parsed = http_parse_message( $response );
    
    $str = ''.$response_parsed->body;  // Promote it to a string. THis is needed because the body is returned
                                       // as stdClass which can't be JSON decoded.
    $a = json_decode( $str );

    $num_collections = 0;
    $num_files = 0;
    $size_gb = 0.0;
    foreach( $a as $e ) {
        if( $e->type == 'collection' ) {
            $num_collections++;
        } else {
            $num_files++;
            $size_gb += $e->size / ( 1024.0 * 1024.0 * 1024.0 );
        }
    }
    return array(
        'collections' => $num_collections,
    	'files' => $num_files,
    	'size' => sprintf( "%.2f", $size_gb ));
}

?>

  </body>
</html>
    