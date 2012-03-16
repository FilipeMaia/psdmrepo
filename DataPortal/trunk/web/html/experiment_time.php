
<!------------------- Document Begins Here ------------------------->

<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php');
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\DataPortal;
use DataPortal\DataPortalException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/* Static parameters describing a geometry of the visible components
 * on a screen.
 * 
 * TODO: The height of some elements may be to low to accomodate porinted text
 * values if the minimumally allowed gap will be less than 30 minutes. Consider
 * implementive some adaptive algorithm for adjusting the vertical limits
 * according to the select font size and the gap width.
 */
$total_height   = 860;
$time_width     =  60;
$instr_width    = 100;
$beamon_width   =  80;
$comments_width = 400;
$system_width   =  80;
$uid_width      =  80;
$posted_width   = 120;

$start_day = null;
try {
    $start_day = LusiTime::today()->toStringDay();
    if( isset($_GET['start_day'] )) {
        $t = LusiTime::parse( trim($_GET['start_day']).' 00:00:00' );
        if( is_null($t)) {
            print "Failed to translate a value of the start dat passed as a parameter to the script.";
            exit;
        }
        $start_day = $t->toStringDay();
    }
} catch( LusiTimeException $e ) {
    print "Failed to translate a value of the start dat passed as a parameter to the script.";
    exit;
}

try {

	$authdb = AuthDB::instance();
	$authdb->begin();

    $can_edit = $authdb->hasRole($authdb->authName(),null,'BeamTimeMonitor','Editor');

    //DataPortal::begin( "LCLS Beam-Time Usage Monitor" );
    DataPortal::begin( "LCLS Data Taking Time Monitor" );

?>



<!------------------- Page-specific Styles ------------------------->
<style type="text/css"> 

#workarea {
  padding: 20 20 0 20;
  border-top: 1 solid #000000;
}
#controls {
  float: left;
  margin-bottom: 20;
}
#statistics {
  float: left;
  margin-bottom: 20;
  margin-left: 20;
  font-size: 28;
  font-weight: bold;
}
#title {
  float: left;
  margin-left: 40;
}
#beam {
  float: left;
  margin-left: 10;
}
#usage {
  float: left;
  margin-left: 10;
  color: green;
}
.table_column {
  float: left;
  border: 1 solid #c0c0c0;
  border-bottom: 1 solid #c0c0c0;
}
.table_column_time {
  width: <?php echo $time_width; ?>;
  border-left: 0;
  border-right: 1 solid #080808;
}
.table_column_instr {
  width: <?php echo $instr_width; ?>;
  border-left: 0;
}
.table_column_beamon {
  width: <?php echo $beamon_width; ?>;
  border-left: 0;
  border-right: 1 solid #080808;
}
.table_column_comments {
  width: <?php echo $comments_width; ?>;
  border-left: 1 solid #080808;
}
.table_column_system {
  width: <?php echo $system_width; ?>;
}
.table_column_uid {
  width: <?php echo $uid_width; ?>;
}
.table_column_posted {
  width: <?php echo $posted_width; ?>;
}
.table_column_header {
  padding: 4;
  border-bottom: 1 solid #080808;
  background-color: #c0c0c0;
  text-align: center;
  font-weight: bold;
}
.table_column_body {
  position:relative;
  height: <?php echo $total_height; ?>;
  background-color:#e0e0e0;
}
.table_column_body_time {
  background-color:#c0c0c0;
  font-size: 12;
}
.beamon {
  width: <?php echo $beamon_width; ?>;
  background-color: #000000;
}
.comment {
  width:  <?php echo $comments_width - 4; ?>;
  background-color: #ffffff;
  padding-left: 2;
  font-size: 12;
}
.comment2edit {
  cursor: pointer;
  background-color: #B9DCF5;
}
.system {
  width:  <?php echo $system_width; ?>;
  background-color: #ffffff;
  border-left: 0;
  font-size: 12;
}
.uid {
  width:  <?php echo $uid_width; ?>;
  background-color: #ffffff;
  border-left: 0;
  font-size: 12;
}
.posted {
  width:  <?php echo $posted_width; ?>;
  background-color: #ffffff;
  border-left: 0;
  font-size: 12;
}
</style>
<!----------------------------------------------------------------->






<?php
    DataPortal::scripts( "page_specific_init" );
?>


<!------------------ Page-specific JavaScript ---------------------->
<script type="text/javascript">

var total_height   = <?php echo $total_height;   ?>;
var time_width     = <?php echo $time_width;     ?>;
var instr_width    = <?php echo $instr_width;    ?>;
var beamon_width   = <?php echo $beamon_width;   ?>;
var comments_width = <?php echo $comments_width; ?>;
var system_width   = <?php echo $system_width;   ?>;
var uid_width      = <?php echo $uid_width;      ?>;
var posted_width   = <?php echo $posted_width;   ?>;
var can_edit       = <?php echo $can_edit ? 'true' : 'false'; ?>;

function page_specific_init() {
    var controls = $('#controls');

    var initial_shift = '<?php echo $start_day; ?>';
    controls.find('input[name="shift"]').
        datepicker().
        datepicker('option','dateFormat','yy-mm-dd').
        datepicker('setDate',initial_shift).
        change(function() {
            load_shift($(this).val(),0); });

    controls.find('button[name="prev_week"]').
        button().
        click(function() {
            load_shift(
                controls.find('input[name="shift"]').val(),
                -7*24*3600);
        });

    controls.find('button[name="prev_shift"]').
        button().
        click(function() {
            load_shift(
                controls.find('input[name="shift"]').val(),
                -24*3600);
        });

    controls.find('button[name="next_shift"]').
        button().
        click(function() {
            load_shift(
                controls.find('input[name="shift"]').val(),
                24*3600);
        });

    controls.find('button[name="next_week"]').
        button().
        click(function() {
            load_shift(
                controls.find('input[name="shift"]').val(),
                7*24*3600);
        });

    load_shift(initial_shift,0);
}
    
function report_error(msg) {
    alert(msg);
}
function save_comment(gap_begin_time_64,comment,system) {
    $.ajax({
        type: 'POST',
        url: '../portal/experiment_time_save_comment.php',
        data: {
            gap_begin_time_64: gap_begin_time_64,
            comment: comment,
            system : system
        },
        success: function(data) {
            if( data.status != 'success' ) { report_error(data.message); return; }
            set_comment( gap_begin_time_64, data.comment );
        },
        error: function() {	report_error('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    });
}

var shift_data = null;
function system_changed() {
    var editor = $('#comment_editor');
    var system = editor.find('select').val();
    editor.find('input').css('display', ( system ? 'none' : 'block' ));
}
function edit_comment(gap_begin_time_64) {
    var html =
'<div id="comment_editor" >'+
'  <div style="margin-top:5px;">'+
'    <b>Comment:</b>'+
'    <br>'+
'    <textarea style="margin-top:5px; padding:2;" rows=8 cols=56 >'+$('#comment_'+gap_begin_time_64).text()+'</textarea>'+
'  </div>'+
'  <div style="margin-top:5px;">'+
'    <div style="float:left; padding-top:4; ">'+
'      <b>System responsible: </b>'+
'    </div>'+
'    <div style="float:left; ">'+
'      <select onchange="system_changed()">';
    for( var i in shift_data.systems ) {
        var system = shift_data.systems[i];
        html +=
'        <option value="'+system+'">'+system+'</option>';
    }
    html +=
'        <option value="">Create New...</option>'+
'      </select>'+
'    </div>'+
'    <div style="float:left; ">'+
'      <input type="text" style="display:none;" value="" />'+
'    </div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'</div>';
    edit_dialog(
        'Edit Downtime Justification',
        html,
        function () {
            var editor = $('#comment_editor');
            var comment = editor.find('textarea').val();
            var system = editor.find('select').val();
            if( !system ) {
                system = editor.find('input').val();
                shift_data.systems.push(system);
            }
            save_comment(gap_begin_time_64, comment, system);
        },
        null,
        460,
        280
    );
}
function enable_comment_editor(elem,yes) {
    if(yes) $(elem).addClass   ('comment2edit');
    else    $(elem).removeClass('comment2edit');
}
function display_shift() {

    $('#beam').html(shift_data.beam+'%');
    $('#usage').html(shift_data.usage+'%');

    var stop_sec    = shift_data.stop_sec;
    var start_sec   = shift_data.start_sec;

    var plot =
'<div class="table_column table_column_time" >'+
'  <div class="table_column_header" >time</div>'+
'  <div class="table_column_body table_column_body_time" >';
    for( var sec = 0.; sec <= stop_sec - start_sec; sec += 3600.) {
        var h = sec / 3600.;
        if( h >= 24. ) continue;
        var h2 = Math.floor(h);
        var left = 0;
        var top1   = Math.floor( total_height * sec   / ( stop_sec - start_sec )) - 12;
        var height = Math.floor( total_height * 1800. / ( stop_sec - start_sec ));
        var top2   = top1 + height;
        if( h > 0. ) {
            plot +=
'    <div style="position:absolute; left:'+left+'; top:'+top1+'; height:'+height+'; padding-left:12; ">'+h2+':00</div>'+
'    <div style="position:absolute; left:'+left+'; top:'+top1+'; height:'+height+'; padding-left:48; font-weight:bold;">-</div>';
        }
        plot +=
'    <div style="position:absolute; left:'+left+'; top:'+top2+'; height:'+height+'; padding-left:48; font-weight:bold;">-</div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_beamon" >'+
'  <div class="table_column_header" >beam</div>'+
'  <div class="table_column_body" >';
    {
        var left   = 0;
        var top    = 0;
        var height = total_height;
        plot +=
'    <div class="beamon" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " ></div>';
    }
    plot +=
'  </div>'+
'</div>';

    var plot4instr = {};
    for( var i in shift_data.instrument_names )
        plot4instr[shift_data.instrument_names[i]] = '';

    for( var i in shift_data.runs ) {
        var run     = shift_data.runs[i];
        var left   = 0;
        var top    = Math.floor( total_height *                           run.begin_rel2start_sec   / ( stop_sec - start_sec));
        var height = Math.floor( total_height * ( run.end_rel2start_sec - run.begin_rel2start_sec ) / ( stop_sec - start_sec));
        if( !height ) height = 1;
        plot4instr[run.instr_name] +=
'    <div style="position:absolute; left:'+left+'; top:'+top+'; width:'+instr_width+'; height:'+height+'; background-color:#000000; "></div>';
    }
    for( var i in shift_data.instrument_names ) {
        var instr_name = shift_data.instrument_names[i];
        plot +=
'<div class="table_column table_column_instr" >'+
'  <div class="table_column_header" >'+instr_name+'</div>'+
'  <div class="table_column_body" >';
        for( var j in shift_data.gaps ) {
            var gap = shift_data.gaps[j];
            var left = 0;
            var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec ));
            var height = Math.floor( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
            if( !height ) height = 1;
            plot +=
'    <div style="position:absolute; left:'+left+'; top:'+top+'; width:'+instr_width+'; height:'+height+'; background-color:#ffffff; "></div>';
        }
        plot +=
plot4instr[instr_name]+
'  </div>'+
'</div>';
    }
    plot +=
'<div class="table_column table_column_comments" >'+
'  <div class="table_column_header" >downtime justification</div>'+
'  <div class="table_column_body" >';
    for( var i in shift_data.gaps ) {
        var gap = shift_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.floor( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="comment" id="comment_'+gap.begin_time_64+'" '+
    (can_edit ?
' onclick="edit_comment('+gap.begin_time_64+')" onmouseover="enable_comment_editor(this,true)" onmouseout="enable_comment_editor(this,false)"' :
''
    )+
' style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; overflow:auto;" '+(can_edit ? ' title="click to edit"' : '')+' ></div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_system" >'+
'  <div class="table_column_header" >system</div>'+
'  <div class="table_column_body" >';
    for( var i in shift_data.gaps ) {
        var gap = shift_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.floor( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="system" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " >'+
'      <div id="system_'+gap.begin_time_64+'" ></div>'+
'    </div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_uid" >'+
'  <div class="table_column_header" >posted by</div>'+
'  <div class="table_column_body" >';
    for( var i in shift_data.gaps ) {
        var gap = shift_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.floor( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="uid" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " >'+
'      <div id="uid_'+gap.begin_time_64+'" ></div>'+
'    </div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_posted" >'+
'  <div class="table_column_header" >post time</div>'+
'  <div class="table_column_body" >';
    for( var i in shift_data.gaps ) {
        var gap = shift_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.floor( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="posted" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " >'+
'      <div id="posted_'+gap.begin_time_64+'" ></div>'+
'    </div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div style="clear:both;"></div>';

    $('#current_selection').html(plot);
    for( var i in shift_data.gaps ) {
        var gap = shift_data.gaps[i];
        set_comment( gap.begin_time_64, gap.comment );
    }
}
function set_comment(gap_begin_time_64, comment) {
    if( comment.available ) {
        $('#comment_'+gap_begin_time_64).text(comment.comment);
        $('#system_' +gap_begin_time_64).text(comment.system);
        $('#uid_'    +gap_begin_time_64).text(comment.posted_by_uid);
        $('#posted_' +gap_begin_time_64).text(comment.post_time);
    } else {
        $('#comment_'+gap_begin_time_64).text('');
        $('#system_' +gap_begin_time_64).text('');
        $('#uid_'    +gap_begin_time_64).text('');
        $('#posted_' +gap_begin_time_64).text('');
    }
}
function load_shift(shift,delta) {
    $.ajax({
        type: 'GET',
        url: '../portal/experiment_time_get.php',
        data: {shift: shift, delta:delta},
        success: function(data) {
            if( data.status != 'success' ) { report_error(data.message); return; }
            shift_data = data;
            $('#controls').find('input[name="shift"]').
                datepicker('setDate',shift_data.shift);
            display_shift();
        },
        error: function() {	report_error('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    });
}

</script>
<!----------------------------------------------------------------->


<?php
    //DataPortal::body( "LCLS Beam-Time Usage Monitor" );
    DataPortal::body( "LCLS Data Taking Time Monitor" );
?>




<!------------------ Page-specific Document Body ------------------->
<?php

    echo <<<HERE

<div id="workarea">

  <div id="controls">

    <button name="prev_week"  title="go to the previous week" > &lt;&lt; </button>
    <button name="prev_shift" title="go to the previous shift" > &lt; </button>

    <input style="padding:3; font-size:16;" type="text" size=7 name="shift" />

    <button name="next_shift" title="go to the next shift" > &gt; </button>
    <button name="next_week"  title="go to the next week"  > &gt;&gt;</button>

  </div>

  <div id="statistics" >
    <div id="title" >Beam time: </div>
    <div id="beam" ></div>
    <div id="title" >Usage: </div>
    <div id="usage" ></div>
    <div style="clear:both;"></div>
  </div>

  <div style="clear:both;"></div>

  <div id="current_selection">Loading...</div>

</div>

HERE;

?>
<!----------------------------------------------------------------->




<?php
    DataPortal::end();
?>
<!--------------------- Document End Here -------------------------->

<?php

} catch( AuthDBException     $e ) { print $e->toHtml(); exit; }
  catch( LusiTimeException   $e ) { print $e->toHtml(); exit; }
  catch( DataPortalException $e ) { print $e->toHtml(); exit; }

?>
