<!DOCTYPE html>
<meta charset="utf-8">
<style  type="text/css">

body {
  font: 10px sans-serif;
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

.x.axis path {
  display: none;
}

</style>
<body>
<script type="text/javascript" src="/d3/d3.js"></script>
<script type="text/javascript">

var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 2500 - margin.left - margin.right,
    height = 720 - margin.top - margin.bottom;

var formatPercent = d3.format(".0");

var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1);

var y = d3.scale.linear()
    .range([height, 0]);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .tickFormat(formatPercent);

var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.json("NERSCMigrationMonitorLatency?exper_id=280", function(error, data) {

  data.forEach(function(d) {
    d.file = +d.file;
    d.latency = +d.latency;
  });
  x.domain(data.map(function(d) { return d.file; }));
  y.domain([0, d3.max(data, function(d) { return d.latency; })]);

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Latency");

  svg.selectAll(".bar")
      .data(data)
    .enter().append("rect")
      .attr("class", "bar")
      .attr("x", function(d) { return x(d.file); })
      .attr("width", x.rangeBand())
      .attr("y", function(d) { return y(d.latency); })
      .attr("height", function(d) { return height - y(d.latency); });
});

</script>
