<!DOCTYPE html>
<html>
<head>
<title>Testing Dynamic Table class</title>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="../webfwk/js/Table.js"></script>

<style type="text/css">
body {
  margin: 0;
  padding: 0;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
h2 {
    font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
.first {
  float: left;
}
.next {
  float: left;
  margin-left: 40px;
}
.last {
  clear: both;
}
</style>

<script type="text/javascript">

$(function() {

    var table1 = new Table(
        'table1',
        [
            {   name: '1'},
            {   name: '2'},
            {   name: '3', sorted: false }
        ],
        [
            ['1(1)','2(2)','3(2)'],
            ['2(1)','3(2)','4(2)'],
            ['3(1)','4(2)','1(2)'],
            ['4(1)','1(2)','2(2)']
        ]
    );
    table1.display();

    var table2 = new Table(
        'table2',
        [
            {name: 'id'},
            {   name: '2',
                coldef: [
                    {   name: '2.1'},
                    {   name: '2.2'}
                ]},
            {name: '3'},
            {name: '4'}
        ],
        [   ['1', '1(2.1)', '1(2.2)', '1(3)', '1(4)'],
            ['2', '2(2.1)', '2(2.2)', '2(3)', '2(4)'],
            ['3', '3(2.1)', '3(2.2)', '3(3)', '3(4)'],
            ['4', '4(2.1)', '4(2.2)', '4(3)', '4(4)']
        ]
    );
    table2.display();

    var table3 = new Table(

        'table3',

        [
            {   name:   'id' },
            {   name:   '2',
                coldef: [
                    {   name:   '2.1',
                        coldef: [
                            {   name:   '2.1.1' },
                            {   name:   '2.1.2' }]},
                    {   name:   '2.2' }]},
            {   name:   '3' },
            {   name:   '4' }],

        [   ['1', '1(2.1.1)', '1(2.1.2)', '1(2.2)', '1(3)', '1(4)'],
            ['2', '2(2.1.1)', '2(2.1.2)', '2(2.2)', '2(3)', '2(4)'],
            ['3', '3(2.1.1)', '3(2.1.2)', '3(2.2)', '3(3)', '3(4)'],
            ['4', '4(2.1.1)', '4(2.1.2)', '4(2.2)', '4(3)', '4(4)']]

    );
    table3.display();

    function MyCellType() { TableCellType.call(this); }
    define_class( MyCellType, TableCellType, {}, {
       to_string     : function(a)   { return '<b>'+a+'</b>'; },
       compare_values: function(a,b) { return this.compare_strings(a,b); }}
    );

    var table4 = new Table(
        'table4',
        [
            {   name: 'Text_URL',   type: Table.Types.Text_URL },
            {   name: 'Number_URL', type: Table.Types.Number_URL },
            {   name: 'MyCellType', type: new MyCellType },
            {   name: 'Customized', type: { to_string     : function(a)   { return '<button class="my_button" name="'+a.data+'" >'+a.data+'</button>'; },
                                            compare_values: function(a,b) { return a.data - b.data; },
                                            after_sort    : function()    { $('.my_button').button().click(function() { alert(this.name); }); }}}
        ],
        null,
        Table.Status.Empty
    );
    table4.display();

    var data4 = [
        [ {text: 'A',         url: 'https://www.slac.stanford.edu'}, {number: 123, url: 'https://www.slac.stanford.edu'}, '3(2)', {data:   3} ],
        [ {text: 'a',         url: 'https://www.slac.stanford.edu'}, {number: -99, url: 'https://www.slac.stanford.edu'}, '4(2)', {data:  11} ],
        [ {text: 'xYz',       url: 'https://www.slac.stanford.edu'}, {number:   3, url: 'https://www.slac.stanford.edu'}, '1(2)', {data:  12} ],
        [ {text: 'let it be', url: 'https://www.slac.stanford.edu'}, {number:   0, url: 'https://www.slac.stanford.edu'}, '2(2)', {data:   1} ]
    ];
    $('#table4_load').button().click(function() { table4.load(data4); });
    $('#table4_erase').button().click(function() { table4.erase(); });

    var table5 = new Table(
        'table5',
        [
            {   name: 'Number', type: Table.Types.Number },
            {   name: 'Text'}
        ]
    );
    table5.display();

    $('#table5_load').button().click(function() {
        table5.erase(Table.Status.Loading);
        $.ajax({
            type: 'GET',
            url: '../webfwk/ws/table_data.php',
            data: {
                rows: 12,
                cols: table5.cols()},
            success: function(result) {
                if(result.status != 'success') {
                    table5.erase(Table.Status.error(result.message));
                    return;
                }
                table5.load(result.data);
            },
            error: function() {
                table5.erase(Table.Status.error('service is not available'));
            },
            dataType: 'json'
        });
    });
});

</script>

</head>
<body>

<div style="padding:20px;">

  <div class="first">
    <h2>1 level</h2>
    <div style="margin-left:0px;" id="table1"></div>
  </div>
  <div class="next">
    <h2>2 levels</h2>
    <div style="margin-left:0px;" id="table2"></div>
  </div>
  <div class="next">
    <h2>3 levels</h2>
    <div style="margin-left:0px;" id="table3"></div>
  </div>
  <div class="last"></div>

  <br>

  <div class="first">
    <h2>Non-trivial cell types,<br>
        Loading data from static memory object</h2>
    <div style="float:left; margin-left: 0px;" id="table4"></div>
    <div style="float:left; margin-left:20px;" ><button id="table4_load">Load</button></div>
    <div style="float:left; margin-left: 5px;" ><button id="table4_erase">Erase</button></div>
    <div style="clear:both"></div>
  </div>
  <div class="next">
    <h2>Data from Web service</h2>
    <div style="float:left; margin-left: 0px;" id="table5"></div>
    <div style="float:left; margin-left:20px;" ><button id="table5_load">Load</button></div>
    <div style="clear:both"></div>
  </div>
  <div class="last"></div>

</div>

</body>
</html>

