<?php
$url = "https://pswww.slac.stanford.edu/ws-auth/irodsws/files";
$opts = array(
    "timeout"      => 1,
    "httpauthtype" => HTTP_AUTH_BASIC,
    "httpauth"     => "gapon:newlife2"
);
$info = null;
$response = http_get($url, $opts, $info );
print '<br>';
print $response;
print '<br>';
print_r( $info );

if( $info['response_code'] == 200) {
    $response_parsed = http_parse_message( $response );

    print '<br>';
    print_r( $response_parsed );
    $str = ''.$response_parsed->body;  // Promote it to a string. THis is needed because the body is returned
                                       // as stdClass which can't be JSON decoded.
    $a = json_decode( $str );
    foreach( $a as $e ) {
        print '<br>';
        print_r( $e->url );
    }
}
?>
