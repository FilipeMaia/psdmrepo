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
    <link type="text/css" rel="stylesheet" href="style.css" /> 
  </head>

  <body>

<?php

if( !isset( $_GET['resource'] )) die( "resource name required" );
$resource = $_GET['resource'];

$base_url = 'https://pswww.slac.stanford.edu/ws-auth/irodsws/resources';

?>

<div id="context">
<?php

    print <<<HERE
&nbsp;{$resource}&nbsp;&gt;
HERE;

?>
</div>

<?php

$url = $base_url.'/'.$resource;

$opts = array(
    "timeout"      => 1,
    "httpauthtype" => HTTP_AUTH_BASIC,
    "httpauth"     => "gapon:newlife2"
);
$info = null;
$response = http_get($url, $opts, $info );

if( $info['response_code'] != 200 )
    die( "failed to get the resource information from Web service" );

$response_parsed = http_parse_message( $response );

$str = ''.$response_parsed->body;  // Promote it to a string. THis is needed because the body is returned
                                   // as stdClass which can't be JSON decoded.
$r = json_decode( $str );

# -------------
# Show resource
# -------------

print <<<HERE
<div id="resources">
<table class="table"><tbody>
  <tr>
    <td class="first_col_hdr">Attribute</td>
    <td class="col_hdr">Value</td>
  </tr>
  <tr>
    <td><div class="first_separator"></div></td>
    <td><div class="separator"></div></td>
  </tr>
  <tr>
    <td class="first_col_row">name</td>
    <td class="col_row">{$r->name}</td>
  </tr>
  <tr>
    <td class="first_col_row">resc_id </td>
    <td class="col_row">{$r->resc_id }</td>
  </tr>
  <tr>
    <td class="first_col_row">type</td>
    <td class="col_row">{$r->type}</td>
  </tr>
  <tr>
    <td class="first_col_row">class</td>
    <td class="col_row">{$r->class}</td>
  </tr>
  <tr>
    <td class="first_col_row">zone</td>
    <td class="col_row">{$r->zone}</td>
  </tr>
  <tr>
    <td class="first_col_row">location</td>
    <td class="col_row">{$r->location}</td>
  </tr>
  <tr>
    <td class="first_col_row">valult</td>
    <td class="col_row">{$r->valult}</td>
  </tr>
  <tr>
    <td class="first_col_row">free_space</td>
    <td class="col_row">{$r->free_space}</td>
  </tr>
  <tr>
    <td class="first_col_row">info</td>
    <td class="col_row">{$r->info}</td>
  </tr>
  <tr>
    <td class="first_col_row">comment</td>
    <td class="col_row">{$r->comment}</td>
  </tr>
</tbody></table>
</div>
HERE;


?>

  </body>
</html>
    