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

function zeroPad(num, len) {
  var s = String(num);
  while (s.length < len) {
    s = "0" + s;
  }
  return s;
}

// Return yyyy-mm-dd from Date object
function dateToYMD(date) {
  return zeroPad(date.getFullYear(), 4) + "-" + zeroPad(date.getMonth() + 1, 2) + "-" + zeroPad(date.getDate(), 2);
}

// Return hh:mm:ss from Date object
function dateToHMS(date) {
  return zeroPad(date.getHours(), 2) + ":" + zeroPad(date.getMinutes(), 2) + ":" + zeroPad(date.getSeconds(), 2);
}

// Convert unix_time (seconds since 1970 UTC) to Date() object.
function convert_unix_time_to_Date(unix_time) {
  if (unix_time) {
    return new Date(1000 * unix_time);
  }
}

// Convert unix_time (seconds since 1970 UTC) to string.
function convert_unix_time_to_string(unix_time) {
  var date = convert_unix_time_to_Date(unix_time);
  if (! date) {
    return "";
  }
  return dateToYMD(date) + " " + dateToHMS(date);
}

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
                    shift.username,
                    shift.hutch,
                    convert_unix_time_to_string(shift.start_time),
                    convert_unix_time_to_string(shift.end_time),
                    convert_unix_time_to_string(shift.last_modified_time),
                    shift.stopper_out,
                    shift.door_open,
                    shift.total_shots,
                    shift.other_notes
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
            {   name: 'Username' },
            {   name: 'Hutch' },
            {   name: 'Start Time' },
            {   name: 'End Time' },
            {   name: 'Last Modified Time' },
            {   name: 'Stopper Out' },
            {   name: 'Door Open' },
            {   name: 'Total Shots' },
            {   name: 'Other Notes' }
        ]
    );
    table_shifts.display();
    table_shifts.erase(Table.Status.Loading);

    shift_service('../shiftmgr/ws/get_shifts.php');
});

</script>

</head>
<body>

<div style="padding:20px;">

  <h2>Shifts</h2>
  <div class="table_container" id="table_shifts" style="float:left;"></div>
  <div style="clear:both;">

</div>

</body>
</html>
