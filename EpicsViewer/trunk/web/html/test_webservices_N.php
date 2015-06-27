<!DOCTYPE html>
<html>
<head>

<title>Load and display multiple PVs</title>

<meta charset="UTF-8">
<!--
<meta name="viewport" content="width=device-width, initial-scale=1.0">
-->
<meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0">

<style>
body {
    margin:         0px;
    padding:        0px;
}
div#body {
    padding:20px;
}
h2 {
    margin:         0px;
    font-family:    Lucida Grande, Lucida Sans, Arial, sans-serif;
}
button {
    background:     rgba(240, 248, 255, 0.39) !important;
    border-radius:  2px !important;
}
button > span {
    color:      black;
    font-size:  10px;
}
#getallpvs {
    margin-top:     10px;
    margin-bottom:  20px;
    padding:        10px;
    height:         100px;
    overflow:       auto;
    border:         1px solid #b0b0b0;
}
#getallpvs div.pvname {
    float:      left;
    min-width:  260px;
    font-size:  12px;
}
#getallpvs div.pvname:hover {
    cursor:           pointer;
    background-color: aliceblue;
}
#getallpvs div.pvname.selected {
    background-color: #0071bc;
    color:            #ffffff;
}
#getallpvs div.pvname_endoflist {
    clear:      both;
}
#getdata {
    height:     120px;
}
#getdata > #loadlog ,
#getdata > #info {
    width:      50%;
    height:     100%;
    overflow:   auto;
}
#getdata_control {
    margin-top: 20px;
}
#getdata_control > .title {
    float:          left;
    margin-right:   10px;
    padding:        0px;
    padding-top:    4px;
    font-family:    'Source Sans Pro',Arial,sans-serif;
    font-size:      14px;
    font-weight:    bold;
}
#getdata_control > .title.end {
    margin-right:   0px;
}

#getdata_control > .control {
    float:          left;
    margin-right:   20px;
}
#getdata_control > .control.end {
    padding-top:    2px;
    margin-right:   0px;
    padding-left:   0px;
    padding-right:  0px;
}
#getdata_control > .control#end_ymd {
    margin-right:   5px;
}
#getdata_control > .control#end_ymd > input {
    width:          65px;
    padding:        2px;
    border:         solid 1px #ffffff;    
}
#getdata_control > .control#end_hh  > input,
#getdata_control > .control#end_mm  > input,
#getdata_control > .control#end_ss  > input {
    width:          16px;
    padding:        2px;
    padding-left:   0px;
    padding-right:  0px;
    border:         solid 1px #ffffff;
}
#getdata_control > .control#end_ymd > input:hover,
#getdata_control > .control#end_hh  > input:hover,
#getdata_control > .control#end_mm  > input:hover,
#getdata_control > .control#end_ss  > input:hover {
    border:     solid 1px #d0d0d0;
}
#getdata_control > .control#end_now {
    margin-left:    10px;
}
#getdata_timeseries{
    width:      100%;
    height:     100px;
}
span.error {
    color:  maroon;
}
</style>

<script data-main="../EpicsViewer/js/test_webservices_N.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

<script>
<?php

// pack an array of PV names into a JSON array

define ('MAX_PVS', 5) ;
$pvs = '' ;
for ($i = 0; $i < MAX_PVS; $i++) {
    $key = "pv{$i}" ;
    $pv = $_GET[$key] ;
    if (isset($pv)) {
        $pvs .= ($pvs ? ',' : '[').'"'.trim($pv) .'"' ;
    }
}
$pvs .= $pvs ? ']' : '[]' ;

$from = isset($_GET['from']) ? trim($_GET['from']) : '' ;
$to   = isset($_GET['to'])   ? trim($_GET['to'])   : '' ;

print <<<HERE
window.global_options = {
   pvs:  {$pvs} ,
   from: "{$from}" ,
   to:   "{$to}"
} ;

HERE;
?>
</script>

</head>
<body>
    <div id="body" >

      <h2 id="loaded" >getAllPVs</h2>
      <div id="getallpvs" ></div>

      <h2 id="selected" >getData</h2>
      <div id="getdata" >
        <div id="loadlog" style="float:left;" ></div>
        <div id="info"    style="float:left;" ></div>
        <div              style="clear:both;" ></div>
      </div>
      <div id="getdata_control" >
        <div                class="title"   >Window:</div>
        <div id="interval"  class="control" ></div>
        <div                class="title"   >End:</div>
        <div id="end_ymd"   class="control end" ><input type="text" size="8" /></div>
        <div id="end_hh"    class="control end" ><input type="text" size="1" value="10" /></div>
        <div                class="title   end" >:</div>
        <div id="end_mm"    class="control end" ><input type="text" size="1" value="34" /></div>
        <div                class="title   end" >:</div>
        <div id="end_ss"    class="control end" ><input type="text" size="1" value="48" /></div>
        <div id="end_now"   class="control end" ><button>NOW</button></div>
        <div style="clear:both;" ></div>
      </div>
      <canvas id="getdata_timeseries" ></canvas>
    </div>
    <!-- Do not display this image. It's needed as a repository of icons for
         plots. NOte this is just a temporary solution. Eventually the icon
         loading will be the widget's responsibility.
      -->
    <img id="lock" width="20" height="20" src="../webfwk/img/lock.png" alt="The Scream" style="display:none" >
</body>
</html>