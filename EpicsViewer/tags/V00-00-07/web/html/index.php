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
#title {
    font-family:    'Source Sans Pro',Arial,sans-serif;
    font-size:      28px;
    font-weight:    bold;
    text-align:     left;
}

button {
    background:     rgba(240, 248, 255, 0.39) !important;
    border-radius:  2px !important;
}
.control-button-important {
    color:  red;
}
button > span {
    color:      black;
    font-size:  10px;
}


#finder {
    position:   relative;
    margin-top: 20px;
}
#finder > #logo {
    float:          left;
    margin-right:   5px;
}
#finder > #logo > img {
    width:          20px;
}
#finder > #input {
    float:  left;
    width:  40px;
}
#finder > #input[data]:hover:after {
  content: attr(data);
  padding: 8px;
  color: #000000;
  position: absolute;
  left: -20;
  top: -60;
  white-space: pre;
  max-width: 520px;
  z-index: 2;
  border-radius: 3px ;
  background-color: lemonchiffon;
  font-family: 'Segoe UI',Tahoma,Helvetica,Arial,Verdana,sans-serif;
  font-weight: normal;
  font-size: 12px;
  text-align: left;
}
#finder > #results {
    float:      left;
    position:   absolute;
    top:        0px;
    left:       210px;
    display:    none;
    padding:    10px;
    max-height: 480px;
    overflow:   auto;
    z-index:    10;
    background-color: #ffffff;
/*    opacity:    0.9;*/
/*    border:     1px solid #b0b0b0;*/
    border-top: thin solid #c0c0c0;
    box-shadow: 0 5px 10px rgba(0,0,0,0.36);
}
#finder > #results.visible {
    display:    block ;
}
#finder > #results div.pvname {
    float:      left;
    min-width:  260px;
    font-size:  12px;
}
#finder > #results div.pvname:hover {
    cursor:             pointer;
    background-color:   aliceblue;
    font-weight:        bold;
}
#finder > #results div.pvname.selected {
    background-color: #0071bc;
    color:            #ffffff;
}
#finder > #results div.pvname_endoflist {
    clear:      both;
}
#getdata {
    display:    none;
    height:     120px;
}
#getdata_control {
    margin-top: 20px;
}
#getdata_control > .title {
    float:          left;
    margin-left:    5px;
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


#getdata_control > #selected {
    margin-bottom:  20px;
    height:         120px;
    overflow-y:     auto;
}
#getdata_control > #selected > table {
    border-spacing: 0;
    font-size:      12px;
    font-family:    Lucida Grande, Lucida Sans, Arial, sans-serif;
}

#getdata_control > #selected > table > thead > tr > td {
    padding: 5px 5px 5px 5px;
    border-right:   1px solid #c0c0c0;
    border-bottom:  1px solid #b0b0b0;
    font-weight:    bold;
    white-space:    nowrap;
    background-color: #f0f0f0;
}
#getdata_control > #selected > table > thead > tr > td:last-child {
    border-right:   0;
}
#getdata_control > #selected > table > tbody > tr > td.pvname:hover {
    background-color:   aliceblue;
}
#getdata_control > #selected > table > tbody > tr > td {
    padding:        5px 5px 2px 5px;
    border-right:   1px solid #c0c0c0;
    white-space:    nowrap;
}
#getdata_control > #selected > table > tbody > tr > td:last-child {
    border-right:   0;
}

#getdata_timeseries {
    width:      100%;
    height:     100px;
}
span.error {
    color:  maroon;
}
</style>

<script data-main="../EpicsViewer/js_demo/index_main.js?bust=<?=date_create()->getTimestamp()?>" src="/require/require.js"></script>

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

print <<<HERE
window.global_options = {
   pvs:  {$pvs}
} ;

HERE;
?>
</script>

</head>
<body>
    <div id="body" >

      <div id="title" >EPICS Archive Viewer</div>
      <div id="finder" >
        <div id="logo" ><img src="../webfwk/img/View.png" /></div>
        <div id="input"
             data="Enter GLOB pattern and press ENTER to find PVs.
Press ESC to clear the input and close the search window." >
          <input type="text" />
        </div>
        <div id="results" ></div>
        <div style="clear:both;" ></div>
      </div>

      <div id="getdata" >
        <div id="loadlog" style="float:left;" ></div>
        <div id="info"    style="float:left;" ></div>
        <div              style="clear:both;" ></div>
      </div>
      <div id="getdata_control" >
        <div id="selected" >
          <table>
            <thead>
              <tr>
                <td>Delete</td>
                <td>Plot</td>
                <td>Color</td>
                <td>Name</td>
                <td>Archive Fields</d>
                <td>RTYP</td>
                <td>Units</td>
                <td>Processing</td>
                <td>Scale</td>
              </tr>
            </thead>
            <tbody>
            </tbody>
          </table>
        </div>
        <div                class="title"   >Last:</div>
        <div id="interval"  class="control" ></div>
        <div                class="title"   >Before:</div>
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