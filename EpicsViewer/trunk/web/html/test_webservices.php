<!DOCTYPE html>
<html>
<head>

<title>Testing ideas for the EPICS Archive Viewer</title>

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
#getdata_timeseries ,
#getdata_waveform {
/*    width:      50%;*/
    width:      100%;
    height:     100px;
/*
    width:      49.5%;
    border:     1px solid #b0b0b0;
*/
}
span.error {
    color:  maroon;
}
</style>

<script data-main="../EpicsViewer/js/test_webservices.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

<script>
<?php
$pv   = isset($_GET['pv'])   ? '"'.trim($_GET['pv'])  .'"' : 'null' ;
$from = isset($_GET['from']) ? '"'.trim($_GET['from']).'"' : 'null' ;
$to   = isset($_GET['to'])   ? '"'.trim($_GET['to'])  .'"' : 'null' ;

print <<<HERE
window.global_options = {
   pv:   {$pv} ,
   from: {$from} ,
   to:   {$to}
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
      <canvas id="getdata_timeseries" style="float:left;" ></canvas>
      <!--
      <canvas id="getdata_waveform"   style="float:left;" ></canvas>
      -->
      <div                            style="clear:both;" ></div>
    </div>
</body>
</html>