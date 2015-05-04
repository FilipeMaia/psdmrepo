<?php

/*
 * Simple Web service to return an array of rows to test JavaScript
 * class Table.
 */
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print json_encode( array(
        'status' => 'error',
        'message' => $msg
    ));
    exit;
}
function report_success($result) {
    $result['status'] = 'success';
    print json_encode($result);
    exit;
}
function parse_parameter($param) {
    $num = $_GET[$param];
    if( !isset($num)) report_error("please, provide parameter '{$param}'");
    $num = abs(intval(trim($num)));
    if( !$num ) report_error("paameter '{$param}' can't be 0");
    return $num;
}
$cols = parse_parameter('cols');
$rows = parse_parameter('rows');

$data = array();

for( $r = 0; $r < $rows; $r++ ) {
    $row = array();
    for( $c = 0; $c < $cols; $c++ ) {
        $value = rand(0,$rows);
        array_push($row,$value);
    }
    array_push($data,$row);
}
sleep(1.0);
report_success( array( 'data' => $data, 'rows' => $rows, 'cols' => $cols ));

?>
