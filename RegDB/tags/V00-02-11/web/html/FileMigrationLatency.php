<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use FileMgr\FileMgrIrodsDb;
use LusiTime\LusiTime;
use RegDB\RegDB;

function report_error_end_exit ($msg) {
    print "<h2 style=\"color:red;\">Error: {$msg}</h2>";
    exit;
}
if (!isset($_GET['exper_id'])) report_error_end_exit ('Please, provide an experiment identifier!') ;
$exper_id = intval($_GET['exper_id']) ;

?>
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style  type="text/css">

div#header {
  margin-bottom: 20px;        
}

.document_title,
.document_subtitle {
  font-family: "Times", serif;
  font-size: 32px;
  font-weight: bold;
  text-align: left;
}
.document_subtitle {
  color: #0071bc;
}
a, a.link {
  text-decoration: none;
  font-weight: bold;
  color: #0071bc;
}
a:hover, a.link:hover {
  color: red;
}

div.container {
  padding: 20px;
  /*border-bottom: solid 1px black;*/
}

div.container-last {
  border-bottom: 0;
}

div.graph {
  font: 10px sans-serif;
  width: 1400;
  height: 480;
  padding-left: 10px;
}
.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.bar {
  fill: steelblue;
}

.dot {
  stroke: #000;
}

</style>

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script> 

<script type="text/javascript" src="/d3/d3.js"></script>
<script type="text/javascript">

var exper_id = <?php echo $exper_id; ?>;

function display_latency(title, element_selector, url) {

    var that = this;
    
    this.margin = {top: 20, right: 20, bottom: 30, left: 50};
    this.width  = 1400 - this.margin.left - this.margin.right;
    this.height =  480 - this.margin.top  - this.margin.bottom;

    var parseTime = d3.time.format("%Y-%m-%d %H:%M:%S").parse;

    this.x = d3.time.scale()
        .range([0, this.width]);

    this.y = d3.scale.linear()
        .range([this.height, 0]);

    this.xAxis = d3.svg.axis()
        .scale(this.x)
        .orient("bottom");

    this.yAxis = d3.svg.axis()
        .scale(this.y)
        .orient("left");

    this.svg = d3.select(element_selector).append("svg")
        .attr("width",  this.width +  this.margin.left + this.margin.right)
        .attr("height", this.height + this.margin.top  + this.margin.bottom)
      .append("g")
        .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");

    this.color_map = {
        'daq2offline'  : 'red',
        'offline2nersc': 'blue'
    };
    d3.json(url, function(error, data) {

      data.forEach(function(d) {
        d.time = parseTime(d.time);
        d.latency = +d.latency;
      });

      that.x.domain(d3.extent(data, function(d) { return d.time; }));
      that.y.domain(d3.extent(data, function(d) { return d.latency; }));

      that.svg.selectAll('.dot')
          .data(data)
        .enter().append("circle")
          .attr("class", "dot")
          .attr("r", 2.0)
          .attr("cx", function(d) { return that.x(d.time); })
          .attr("cy", function(d) { return that.y(d.latency); })
          .style("fill", function(d) { return d3.rgb(that.color_map[d.stage]); });

      that.svg.append("g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + that.height + ")")
          .call(that.xAxis)
        .append("text")
          .attr("class", "label")
          .attr("x", that.width)
          .attr("y", 28)
          .style("text-anchor", "end")
          .text("File Creation Time");

      that.svg.append("g")
          .attr("class", "y axis")
          .call(that.yAxis)
        .append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", 6)
          .attr("dy", ".71em")
          .style("text-anchor", "end")
          .text(title);
    });
}

var shift1 = null,
    shift2 = null,
    shift3 = null,
    shift4 = null,
    shift5 = null;

$(function() {

    shift1 = new display_latency (
        'Latency [sec]',
        'div#shift1',
        '../regdb/ws/GetMigrationMonitorLatency?exper_id='+exper_id+'&first_run='+5+'&last_run='+46
    );
    shift2 = new display_latency (
        'Latency [sec]',
        'div#shift2',
        '../regdb/ws/GetMigrationMonitorLatency?exper_id='+exper_id+'&first_run='+47+'&last_run='+89
    );
    shift3 = new display_latency (
        'Latency [sec]',
        'div#shift3',
        '../regdb/ws/GetMigrationMonitorLatency?exper_id='+exper_id+'&first_run='+90+'&last_run='+122
    );
    shift4 = new display_latency (
        'Latency [sec]',
        'div#shift4',
        '../regdb/ws/GetMigrationMonitorLatency?exper_id='+exper_id+'&first_run='+123+'&last_run='+190
    );
    shift5 = new display_latency (
        'Latency [sec]',
        'div#shift5',
        '../regdb/ws/GetMigrationMonitorLatency?exper_id='+exper_id+'&first_run='+197+'&last_run='+287
    );
});

</script>

</head>
<body>
    <div style="padding:10px; padding-left:20px;">

<?php
try {

    AuthDB::instance()->begin();
    RegDB::instance()->begin();
    FileMgrIrodsDb::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id($exper_id) ;
    if (!$experiment) report_error_end_exit('NO such experiment exists') ;

    print <<<HERE
<div id="header" >
  <span class="document_title">File Migration Latencies for:&nbsp;</span>
  <span class="document_subtitle">
    <a class="link" title="Open a new tab to the Web Portal of the experiment" href="../portal/?exper_id={$exper_id}" target="_blank" >
      {$experiment->instrument()->name()} / {$experiment->name()}
    </a>
  </span>
</div>

HERE;
    AuthDB::instance()->commit();
    RegDB::instance()->commit();
    FileMgrIrodsDb::instance()->commit();

} catch( Exception $e ) { report_error_end_exit($e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>'); }

?>
        <div class="container">
            <h1>Shift 1: runs 5 - 46</h1>
            <div class="graph" id="shift1"></div>
        </div>
        <div class="container">
            <h1>Shift 2: runs 47 - 89</h1>
            <div class="graph" id="shift2"></div>
        </div>
        <div class="container">
            <h1>Shift 3: runs 90 - 122</h1>
            <div class="graph" id="shift3"></div>
        </div>
        <div class="container">
            <h1>Shift 4: runs 123 - 190</h1>
            <div class="graph" id="shift4"></div>
        </div>
        <div class="container container-last">
            <h1>Shift 5: runs 197 - 287</h1>
            <div class="graph" id="shift5"></div>
        </div>
    </div>
</body>
</html>
