<!DOCTYPE html>
<html>
<head>
<title>Testing the Shift Manager application</title>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>

<script type="text/javascript" src="../portal/js/config.js"></script>
<script type="text/javascript" src="../portal/js/Table.js"></script>

<style type="text/css">
body {
  margin: 0;
  padding: 0;
}
.table_container {
  margin-right: 40px;
}
</style>

<script type="text/javascript">

var config = new config_create('shift_manager') ;

function shift_service (url, params, when_done) {
    $.ajax({
        type: 'GET',
        url:  url,
        data: params,
        success: function(result) {
            if(result.status != 'success') {
                alert(result.message);
                return;
            }
            var shifts_data = [];
            for(var i in result.shifts) {
                var shift = result.shifts[i];
                shifts_data.push ([
                    shift.id,
                    shift.instrument_name,
                    shift.begin_time,
                    shift.end_time
                ]);
            }
            table_shifts.load(shifts_data);
            if(when_done) when_done();
        },
        error: function() {
            table_shifts.erase(Table.Status.error('service is not available'));
        },
        dataType: 'json'
    });
}

var table_shifts = null ;

$(function() {
    table_shifts = new Table(
        'table_shifts',
         [
            {   name: 'Id',
                type: Table.Types.Number },
            {   name: 'Instr' },
            {   name: 'Begin Time' },
            {   name: 'End Time' }
        ]
    );
    table_shifts.display();
    table_shifts.erase(Table.Status.Loading);

    shift_service('../shiftmgr/ws/shift_get.php', {instr:''});

    $('#new_shift_AMO').button().click(function() {
        $('#new_shift_AMO').button('disable');
        shift_service('../shiftmgr/ws/shift_new.php', {instr:'AMO'}, function() {
            $('#new_shift_AMO').button('enable');
        });
    });
});

</script>

</head>
<body>

<div style="padding:20px;">

  <h2>Shifts</h2>
  <div class="table_container" id="table_shifts" style="float:left;"></div>
  <div style="float:left;">
    <button id="new_shift_AMO">New Shift at AMO</button>
  </div>
  <div style="clear:both;">

</div>

</body>
</html>
