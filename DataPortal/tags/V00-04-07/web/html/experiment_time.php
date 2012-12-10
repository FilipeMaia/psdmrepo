
<!------------------- Document Begins Here ------------------------->

<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php');
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\ExpTimeMon;
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
$total_height   = 840;
$time_width     =  60;
$lclson_width   = 140;
$beamon_width   = 140;
$run_width      = 140;
$comments_width = 400;
$system_width   =  80;
$uid_width      = 100;
$posted_width   = 150;

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

    $exptimemon = ExpTimeMon::instance();
    $exptimemon->begin();

    $can_edit = $authdb->hasRole($authdb->authName(),null,'BeamTimeMonitor','Editor');

    DataPortal::begin( "LCLS Data Taking Time Monitor" );

?>



<!------------------- Page-specific Styles ------------------------->
<style type="text/css"> 

#controls {
  margin-bottom: 20;
}
.instrument_container {
  border-top: 1 solid #c0c0c0;
  padding: 20;
}
.alerts_container {
  border-top: 1 solid #c0c0c0;
  padding: 20;
  font-size: 110%;
  font-family: Arial, sans-serif;
}
.statistics {
  margin-bottom: 20;
  font-family: "Times", serif;
  font-weight: bold;
  text-align: left;
  font-size: 24;
}
.delivered_beam_time_title {
  float: left;
  margin-right: 10;
}
.delivered_beam_time {
  float: left;
  margin-right: 30;
  font-weight: normal;
  color: green;
}
.beam_time_title {
  float: left;
  margin-right: 10;
}
.beam_time {
  float: left;
  margin-right: 30;
  font-weight: normal;
  color: green;
}
.esimated_usage_title {
  float: left;
  margin-right: 10;
}
.estimated_usage {
  float: left;
  margin-right: 30;
  font-weight: normal;
  color: green;
}
.table_column {
  float: left;
  border: 1 solid #c0c0c0;
  border-bottom: 1 solid #c0c0c0;
}
.table_column_time {
  width: <?php echo $time_width - 1; ?>;
  border-left: 0;
  border-right: 1 solid #080808;
}
.table_column_lclson {
  width: <?php echo $lclson_width - 1; ?>;
  border-left: 0;
  border-right: 1 solid #080808;
}
.table_column_beamon {
  width: <?php echo $beamon_width - 1; ?>;
  border-left: 0;
  border-right: 1 solid #080808;
}
.table_column_run {
  width: <?php echo $run_width; ?>;
  border-left: 0;
}
.table_column_comments {
  width: <?php echo $comments_width - 1; ?>;
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
  border-right: 2 solid #a0a0a0;
  border-bottom: 2 solid #000000;
  background-color: #c0c0c0;
  text-align: center;
  font-family: "Times", serif;
  font-weight: bold;
  font-size: 18px;
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
.lclson {
  width: <?php echo $lclson_width; ?>;
  background-color: #8A0829;
}
.beamon {
  width: <?php echo $beamon_width; ?>;
  background-color: green;
}
.data_taking {
  position: absolute;
  width: <?php echo $run_width; ?>;
  background-color: green;
}
.comment {
  width:  <?php echo $comments_width - 6; ?>;
  background-color: #ffffff;
  border: solid 1px red; 
  padding-left: 4;
  font-size: 12;
}
.comment2edit {
  cursor: pointer;
  background-color: #B9DCF5;
}
.system {
  width:  <?php echo $system_width - 6; ?>;
  background-color: #ffffff;
  border-left: 0;
  padding-left: 4;
  font-size: 12;
}
.uid {
  width:  <?php echo $uid_width - 6; ?>;
  background-color: #ffffff;
  border-left: 0;
  padding-left: 4;
  font-size: 12;
}
.posted {
  width:  <?php echo $posted_width - 6; ?>;
  background-color: #ffffff;
  border-left: 0;
  padding-left: 4;
  font-size: 12;
}
.visible {
  display: block;
}
.hidden {
  display: none;
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
var lclson_width   = <?php echo $lclson_width;   ?>;
var beamon_width   = <?php echo $beamon_width;   ?>;
var run_width      = <?php echo $run_width;      ?>;
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

    $('#tabs').tabs();

    $('#unsubscribe_button').button().click(function() {
        $.ajax({
            type: 'GET',
            url: '../portal/ws/experiment_time_subscription_toggle.php',
            data: {},
            success: function(data) {
                if( data.status != 'success' ) { report_error(data.message); return; }
                $('#subscribe_area'  ).removeClass('hidden').addClass('visible');
                $('#unsubscribe_area').removeClass('visible').addClass('hidden');
            },
            error: function() {    report_error('The request can not go through due a failure to contact the server.'); },
            dataType: 'json'
        });
    });
    $('#subscribe_button').button().click(function() {
        $.ajax({
            type: 'GET',
            url: '../portal/ws/experiment_time_subscription_toggle.php',
            data: {},
            success: function(data) {
                if( data.status != 'success' ) { report_error(data.message); return; }
                $('#subscribe_area'  ).removeClass('visible').addClass('hidden');
                $('#unsubscribe_area').removeClass('hidden').addClass('visible');
            },
            error: function() {    report_error('The request can not go through due a failure to contact the server.'); },
            dataType: 'json'
        });
    });
    load_shift(initial_shift,0);
}
    
function report_error (msg, on_cancel) {
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left ;"></span>'+msg+'</p>'
    ) ;
    $('#popupdialogs').dialog({
        resizable: true,
        modal: true,
        buttons: {
            Cancel: function() {
                $(this).dialog('close') ;
                if (on_cancel) on_cancel() ;
            }
        },
        title: 'Error'
    }) ;
}

function save_comment(gap_begin_time_64,instr_name,comment,system) {
    $.ajax({
        type: 'POST',
        url: '../portal/ws/experiment_time_save_comment.php',
        data: {
            gap_begin_time_64: gap_begin_time_64,
            instr_name: instr_name,
            comment: comment,
            system : system
        },
        success: function(data) {
            if( data.status != 'success' ) { report_error(data.message); return; }
            set_comment( gap_begin_time_64, instr_name, data.comment );
        },
        error: function() {    report_error('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    });
}

var shift_data = null;

function time_and_lcls2html(start_sec,stop_sec,instr_name) {
    var plot =
'<div class="table_column table_column_time" >'+
'  <div class="table_column_header" >Time</div>'+
'  <div class="table_column_body table_column_body_time" >';
    for( var sec = 0.; sec <= stop_sec - start_sec; sec += 3600.) {
        var h = sec / 3600.;
        if( h >= 24. ) continue;
        var h2 = Math.floor(h);
        var left = 0;
        var top1   = Math.floor( total_height * sec   / ( stop_sec - start_sec )) - 12;
        var height = Math.ceil ( total_height * 1800. / ( stop_sec - start_sec ));
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
'<div class="table_column table_column_lclson" >'+
'  <div class="table_column_header" >LCLS status</div>'+
'  <div class="table_column_body" >';
    for( var i in shift_data.lcls_status ) {
        var ival = shift_data.lcls_status[i];
        if( ival.status > 0 ) {
            var left   = 0;
            var top    = Math.floor( total_height *                            ival.begin_rel2start_sec   / ( stop_sec - start_sec));
            var height = Math.ceil ( total_height * ( ival.end_rel2start_sec - ival.begin_rel2start_sec ) / ( stop_sec - start_sec));
            if( !height ) height = 1;
            plot +=
'    <div class="lclson" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " ></div>';
        }
    }
    plot +=
'  </div>'+
'</div>';
    var destination;
    switch(instr_name) {
        case 'AMO':
        case 'SXR':
        case 'XPP': destination = 'FEE'; break;
        case 'XCS':
        case 'CXI':
        case 'MEC': destination = 'XRT'; break;
    }
    if(destination) {
        plot +=
'<div class="table_column table_column_lclson" >'+
'  <div class="table_column_header" >beam in '+destination+'</div>'+
'  <div class="table_column_body" >';
    for( var i in shift_data.beam_destinations[destination].beam_status ) {
        var ival = shift_data.beam_destinations[destination].beam_status[i];
        if( ival.status > 0 ) {
            var left   = 0;
            var top    = Math.floor( total_height *                            ival.begin_rel2start_sec   / ( stop_sec - start_sec));
            var height = Math.ceil ( total_height * ( ival.end_rel2start_sec - ival.begin_rel2start_sec ) / ( stop_sec - start_sec));
            if( !height ) height = 1;
            plot +=
'    <div class="beamon" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " ></div>';
        }
    }
    plot +=
'  </div>'+
'</div>';
    }
    return plot;
}
function display_beams() {

    var stats = $('#tab_beams .instrument_container .statistics');
    stats.find('.delivered_beam_time').html(shift_data.total_beam_destinations);
    stats.find('.beam_time').html(shift_data.total_beam_time);
    stats.find('.estimated_usage').html(shift_data.total_data_taking);

    var stop_sec  = shift_data.stop_sec;
    var start_sec = shift_data.start_sec;

    var plot = time_and_lcls2html(start_sec,stop_sec);

    for( var i in shift_data.beam_destination_names ) {
        var dest_name = shift_data.beam_destination_names[i];
        var dest_data = shift_data.beam_destinations[dest_name];
        plot +=
'<div class="table_column table_column_beamon" >'+
'  <div class="table_column_header" >'+dest_name+'</div>'+
'  <div class="table_column_body" >';
        for( var i in dest_data.beam_status ) {
            var ival   = dest_data.beam_status[i];
            if( ival.status > 0 ) {
                var left   = 0;
                var top    = Math.floor( total_height *                            ival.begin_rel2start_sec   / ( stop_sec - start_sec));
                var height = Math.ceil ( total_height * ( ival.end_rel2start_sec - ival.begin_rel2start_sec ) / ( stop_sec - start_sec));
                if( !height ) height = 1;
                plot +=
'    <div class="beamon" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " ></div>';
            }
        }
        plot +=
'  </div>'+
'</div>';
    }
    plot +=
'<div style="clear:both;"></div>';
    $('#tab_beams .instrument_container .table').html(plot);
}
function system_changed() {
    var editor = $('#comment_editor');
    var system = editor.find('select').val();
    editor.find('input').css('display', ( system ? 'none' : 'block' ));
}
function edit_comment(gap_begin_time_64,instr_name) {
    var html =
'<div id="comment_editor" >'+
'  <div style="margin-top:5px;">'+
'    <b>Comment:</b>'+
'    <br>'+
'    <textarea style="margin-top:5px; padding:2;" rows=8 cols=56 >'+$('#comment_'+instr_name+'_'+gap_begin_time_64).text()+'</textarea>'+
'  </div>'+
'  <div style="margin-top:5px;">'+
'    <div style="float:left; padding-top:4; ">'+
'      <b>System responsible: </b>'+
'    </div>'+
'    <div style="float:left; ">'+
'      <select onchange="system_changed()">';
    var num_systems = 0;
    for( var i in shift_data.systems ) {
        num_systems++;
        var system = shift_data.systems[i];
        html +=
'        <option value="'+system+'">'+system+'</option>';
    }
    if( !num_systems ) {
        html +=
'        <option value="'+instr_name+'">'+instr_name+'</option>';
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
        'Comment Editor',
        html,
        function () {
            var editor = $('#comment_editor');
            var comment = editor.find('textarea').val();
            var system = editor.find('select').val();
            if( !system ) {
                system = editor.find('input').val();
                shift_data.systems.push(system);
            }
            save_comment(gap_begin_time_64, instr_name, comment, system);
        },
        function () {
            //large_dialog( 'test', 'a range of e-log messages will be here soon' );
        },
        //null,
        460,
        280
    );
}
function enable_comment_editor(elem,yes) {
    if(yes) $(elem).addClass   ('comment2edit');
    else    $(elem).removeClass('comment2edit');
}
function display_shift(instr_data,instr_name) {

    var stats = $('#tab_'+instr_name+' .instrument_container .statistics');
    stats.find('.beam_time').html(instr_data.total_beam_time);
    stats.find('.estimated_usage').html(instr_data.total_data_taking);

    var stop_sec    = shift_data.stop_sec;
    var start_sec   = shift_data.start_sec;

    var plot = time_and_lcls2html(start_sec,stop_sec,instr_name);
    $('#tab_'+instr_name+' .instrument_container .table').html(plot);

    plot +=
'<div class="table_column table_column_beamon" >'+
'  <div class="table_column_header" >Beam in hatch</div>'+
'  <div class="table_column_body" >';
    for( var i in instr_data.beam_status ) {
        var ival   = instr_data.beam_status[i];
        if( ival.status > 0 ) {
            var left   = 0;
            var top    = Math.floor( total_height *                            ival.begin_rel2start_sec   / ( stop_sec - start_sec));
            var height = Math.ceil ( total_height * ( ival.end_rel2start_sec - ival.begin_rel2start_sec ) / ( stop_sec - start_sec));
            if( !height ) height = 1;
            plot +=
'    <div class="beamon" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " ></div>';
        }
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_run" >'+
'  <div class="table_column_header" >Taking data</div>'+
'  <div class="table_column_body" >';
    for( var j in instr_data.gaps ) {
        var gap = instr_data.gaps[j];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec ));
        var height = Math.ceil ( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div style="position:absolute; left:'+left+'; top:'+top+'; width:'+run_width+'; height:'+height+'; background-color:#ffffff; "></div>';
    }
    for( var i in instr_data.runs ) {
        var run     = instr_data.runs[i];
        var left   = 0;
        var top    = Math.floor( total_height *                           run.begin_rel2start_sec   / ( stop_sec - start_sec));
        var height = Math.ceil ( total_height * ( run.end_rel2start_sec - run.begin_rel2start_sec ) / ( stop_sec - start_sec));
        if( !height ) height = 1;
        plot +=
'    <div class="data_taking" style="left:'+left+'; top:'+top+'; height:'+height+'; "></div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_comments" >'+
'  <div class="table_column_header" >Comments</div>'+
'  <div class="table_column_body" >';
    for( var i in instr_data.gaps ) {
        var gap = instr_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.ceil ( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        var gap_sec = gap.end_rel2start_sec - gap.begin_rel2start_sec;
        var title = ( gap_sec > 3600 ? Math.floor( gap_sec / 3600 )+'h '+Math.floor(( gap_sec % 3600 ) / 60 )+'m' : Math.floor( gap_sec / 60 )+'m' )+' : click to edit';
        plot +=
'    <div class="comment" id="comment_'+instr_name+'_'+gap.begin_time_64+'" '+
    (can_edit ?
' onclick="edit_comment('+"'"+gap.begin_time_64+"','"+instr_name+"'"+')" onmouseover="enable_comment_editor(this,true)" onmouseout="enable_comment_editor(this,false)"' :
''
    )+
' style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; overflow:auto;" '+(can_edit ? ' title="'+title+'"' : '')+' ></div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_system" >'+
'  <div class="table_column_header" >System</div>'+
'  <div class="table_column_body" >';
    for( var i in instr_data.gaps ) {
        var gap = instr_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.ceil ( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="system" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " >'+
'      <div id="system_'+instr_name+'_'+gap.begin_time_64+'" ></div>'+
'    </div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_uid" >'+
'  <div class="table_column_header" >Posted by</div>'+
'  <div class="table_column_body" >';
    for( var i in instr_data.gaps ) {
        var gap = instr_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.ceil ( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="uid" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " >'+
'      <div id="uid_'+instr_name+'_'+gap.begin_time_64+'" ></div>'+
'    </div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div class="table_column table_column_posted" >'+
'  <div class="table_column_header" >Post time</div>'+
'  <div class="table_column_body" >';
    for( var i in instr_data.gaps ) {
        var gap = instr_data.gaps[i];
        var left = 0;
        var top    = Math.floor( total_height *                           gap.begin_rel2start_sec   / ( stop_sec - start_sec  ));
        var height = Math.ceil ( total_height * ( gap.end_rel2start_sec - gap.begin_rel2start_sec ) / ( stop_sec - start_sec ));
        if( !height ) height = 1;
        plot +=
'    <div class="posted" style="position:absolute; left:'+left+'; top:'+top+'; height:'+height+'; " >'+
'      <div id="posted_'+instr_name+'_'+gap.begin_time_64+'" ></div>'+
'    </div>';
    }
    plot +=
'  </div>'+
'</div>'+
'<div style="clear:both;"></div>';

    $('#tab_'+instr_name+' .instrument_container .table').html(plot);
    for( var i in instr_data.gaps ) {
        var gap = instr_data.gaps[i];
        set_comment( gap.begin_time_64, instr_name, gap.comment );
    }
}
function set_comment(gap_begin_time_64, instr_name, comment) {
    if( comment.available ) {
        $('#comment_'+instr_name+'_'+gap_begin_time_64).text(comment.comment);
        $('#system_' +instr_name+'_'+gap_begin_time_64).text(comment.system);
        $('#uid_'    +instr_name+'_'+gap_begin_time_64).text(comment.posted_by_uid);
        $('#posted_' +instr_name+'_'+gap_begin_time_64).text(comment.post_time);
    } else {
        $('#comment_'+instr_name+'_'+gap_begin_time_64).text('');
        $('#system_' +instr_name+'_'+gap_begin_time_64).text('');
        $('#uid_'    +instr_name+'_'+gap_begin_time_64).text('');
        $('#posted_' +instr_name+'_'+gap_begin_time_64).text('');
    }
}
function load_shift(shift,delta) {
    $.ajax({
        type: 'GET',
        url: '../portal/ws/experiment_time_get.php',
        data: {shift: shift, delta:delta},
        success: function(data) {
            if( data.status != 'success' ) { report_error(data.message); return; }
            shift_data = data;
            $('#controls').find('input[name="shift"]').
                datepicker('setDate',shift_data.shift);
            display_beams();
            for( var i in shift_data.instrument_names ) {
                var instr_name = shift_data.instrument_names[i];
                display_shift(shift_data.instruments[instr_name],instr_name);
            }
        },
        error: function() { report_error('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    });
}

</script>
<!----------------------------------------------------------------->


<?php
    $shift_controls = <<<HERE
  <div id="controls" style="margin-left:20px; font-size:20;">
      <button name="prev_week"  title="go to the previous week" > &lt;&lt; </button>
      <button name="prev_shift" title="go to the previous shift" > &lt; </button>
      <input style="padding:3; font-size:16;" type="text" size=7 name="shift" />
      <button name="next_shift" title="go to the next shift" > &gt; </button>
      <button name="next_week"  title="go to the next week"  > &gt;&gt;</button>
  </div>
HERE;
    DataPortal::body( "LCLS Data Taking Time Monitor: ", null, null, $shift_controls );
?>




<!------------------ Page-specific Document Body ------------------->
<?php

    echo <<<HERE

<div id="tabs">
    <ul>
      <li><a href="#tab_beams">X-Ray destinations</a></li>

HERE;
    foreach( ExpTimeMon::instrument_names() as $instr_name) {
        echo <<<HERE
      <li><a href="#tab_{$instr_name}">{$instr_name}</a></li>

HERE;
    }
    echo <<<HERE
      <li><a href="#tab_alerts">e-mail alerts</a></li>
    </ul>

    <div id="tab_beams" >
      <div class="instrument_container" >
        <div class="statistics" >
          <div class="delivered_beam_time_title" >Beam delivered: </div>
          <div class="delivered_beam_time" ></div>
          <div class="beam_time_title" >Beam in instrument hatches: </div>
          <div class="beam_time" ></div>
          <div class="esimated_usage_title" >Taking data: </div>
          <div class="estimated_usage" ></div>
          <div style="clear:both;"></div>
        </div>
        <div class="table">Loading...</div>
      </div>
    </div>

HERE;
    foreach( ExpTimeMon::instrument_names() as $instr_name) {
        echo <<<HERE
    <div id="tab_{$instr_name}" >
      <div class="instrument_container" >
        <div class="statistics" >
          <div class="beam_time_title" >Beam in hatch: </div>
          <div class="beam_time" ></div>
          <div class="esimated_usage_title" >Taking data: </div>
          <div class="estimated_usage" ></div>
          <div style="clear:both;"></div>
        </div>
        <div class="table">Loading...</div>
      </div>
    </div>

HERE;
    }
    $subscriber = $authdb->authName();
    $address    = $subscriber.'@slac.stanford.edu';
    $is_subscribed = $exptimemon->check_if_subscribed4explanations( $subscriber, $address );
    $subscribe_class   = $is_subscribed ? 'hidden'  : 'visible';
    $unsubscribe_class = $is_subscribed ? 'visible' : 'hidden';
    echo <<<HERE

    <div id="tab_alerts" >
      <div class="alerts_container" >
        <p>This subscription will allow you to receive prompt notificatons on downtime<br>
           explanations posted for gaps between runs.
        </p>
        <div id="subscribe_area" class="{$subscribe_class}">
          Your SLAC account <b>{$subscriber}</b> is <b>NOT</b> subscribed for notifications.<br>
          Subscribe to receive alerts at: <b>{$address}</b>.<br>
          <button id="subscribe_button" style="font-size:9px; margin-top:15px;">Subscribe</button>
        </div>
        <div id="unsubscribe_area" class="{$unsubscribe_class}">
          Your SLAC account <b>{$subscriber}</b> is already subscribed for notifications.<br>
          Alerts are sent to: <b>{$address}</b>.<br>
          <button id="unsubscribe_button" style="font-size:9px; margin-top:15px;">Unsubscribe</button>
        </div>
      </div>
    </div>

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
