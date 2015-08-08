<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style  type="text/css">

body {
  font: 10px sans-serif;
}

div.graph {
    width: 1900;
    height: 680;
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

.x.axis path {
  display: none;
}

</style>

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script> 

<script type="text/javascript" src="/d3/d3.js"></script>
<script type="text/javascript">

function display_latency(title, element_selector, url, color) {

    var that = this;
    
    this.margin = {top: 20, right: 20, bottom: 30, left: 50};
    this.width  = 1900 - this.margin.left - this.margin.right;
    this.height =  680 - this.margin.top  - this.margin.bottom;

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
          .attr("r", 3.5)
          .attr("cx", function(d) { return that.x(d.time); })
          .attr("cy", function(d) { return that.y(d.latency); })
          .style("fill", function(d) { return d3.rgb(color); });

      that.svg.append("g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + that.height + ")")
          .call(that.xAxis);

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

var daq2offline = null;
var offline2nersc = null;

$(function() {

    var first_run = 47,
        last_run  = 89;

    daq2offline = new display_latency (
        'DAQ-to-OFFLINE Latency [sec]',
        'div#daq2offline',
        'NERSCMigrationMonitorLatency1?exper_id=280&selection=daq2offline&first_run='+first_run+'&last_run='+last_run,
        'red'
    );

    offline2nersc = new display_latency (
        'OFFLINE-to-NERSC Latency [sec]',
        'div#offline2nersc',
        'NERSCMigrationMonitorLatency1?exper_id=280&selection=offline2nersc&first_run='+first_run+'&last_run='+last_run,
        'blue'
    );
});

</script>

</head>
<body>
    <div id="daq2offline_status">Loading...</div>
    <div class="graph" id="daq2offline"></div>
    <div class="graph" id="offline2nersc"></div>
</body>
</html>
