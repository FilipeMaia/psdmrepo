<?php

require_once( 'lusitime.inc.php' );

echo <<<HERE
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <title>LusiTime Tests</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    </head>
    <body>
        <p style="margin-left:12px;  margin-top:12px; margin-bottom:0px; font-size:24px;"><b>Unit tests for:</b> <i>LusiTime</i></p>
        <div style="margin-left:24px; margin-top:0px;">
HERE;

echo <<<HERE
<p style="width:32em;"><b>Get and print the current time and then translate it back
to see if the bi-directional translation works correctly:</b></p>
HERE;

$now = LusiTime::now();
$now_from_64 = LusiTime::from64( $now->to64());
$now_str = $now->__toString();
$now_parsed_from_str = LusiTime::parse( $now_str );
echo <<<HERE
LusiTime::now(): {$now_str}<br>
translated into 64-bit: {$now->to64()}<br>
from 64-bit back into string: {$now_from_64}<br>
LusiTime::parse{{$now_str}}: {$now_parsed_from_str}<br>
HERE;

$t64 = "1242812928000000000";
$r   = LusiTime::from64($t64);
echo <<<HERE
<br>
<br>input:       {$t64}
<br>result:      {$r}
<br>result.sec:  {$r->sec}
<br>result.nsec: {$r->nsec}
HERE;

$t1 = new LusiTime(1);
$t2 = new LusiTime(2);
$t3 = new LusiTime(3);
echo <<<HERE
<br>
<br>1 in [2,3)    : {LusiTime::in_interval($t1, $t2, $t3)}
<br>2 in [2,3)    : {LusiTime::in_interval($t2, $t2, $t3)}
<br>3 in [2,3)    : {LusiTime::in_interval($t3, $t2, $t3)}
<br>3 in [2,null) : {LusiTime::in_interval($t3, $t2, null)}
HERE;

$t_1_2 = new LusiTime(1,2);
$t_1_3 = new LusiTime(1,3);
$t_2_2 = new LusiTime(2,2);
echo <<<HERE
<br>
<br>1.2 <  1.3 : {$t_1_2->less($t_1_3)}
<br>!(1.3 <  1.2) : {!$t_1_3->less($t_1_2)}
<br>1.2 <  2.2 : {$t_1_2->less($t_2_2)}
<br>1.2 == 1.2 : {$t_1_2->equal($t_1_2)}
HERE;

$packed = '099999999';
echo <<<HERE
<br>
<br>Packed:     {$packed}
<br>Translated: {LusiTime::from64( $packed )}
HERE;

$str = "2009-05-19 17:59:49.123-0700";
$lt  = LusiTime::parse( $str) ;
echo <<<HERE
<br>
<br><b>Input time:</b>           {$str}
<br><b>LusiTime::parse():</b>    {$lt->__toString()}
<br><b>converted to 64-bit:</b>  {$lt->to64()}
<br>
HERE;

echo "<br>LusiTime::minus_month(): ".LusiTime::minus_month().
     "<br>LusiTime::minus_week(): ".LusiTime::minus_week().
     "<br>LusiTime::minus_day(): ".LusiTime::minus_day().
     "<br>LusiTime::yesterday(): ".LusiTime::yesterday().
     "<br>LusiTime::today(): ".LusiTime::today().
     "<br>LusiTime::minus_hour(): ".LusiTime::minus_hour().
     "<br>LusiTime::now(): ".LusiTime::now();

echo <<<HERE
   </body>
</html>
HERE;

?>
