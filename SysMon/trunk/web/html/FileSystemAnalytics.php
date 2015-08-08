<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once 'sysmon/sysmon.inc.php' ;

use SysMon\SysMon;

function report_error ($msg) {
    print "<div style=\"color:red;\">Error: {$msg}</div>";
    exit;
}

$file_system_class = isset($_GET['file_system_class']) ? $_GET['file_system_class'] : null;

$state2show_no_selection = '<span style="color:maroon;">No file system selection has been made</span>' ;
$state2show_loading      = '<span style="color:maroon;">Loading...</span>' ;
$state2show_failed       = '<span style="color:maroon;">Loading operation failed.</span>' ;

try {
    SysMon::instance()->begin() ;
    $filesystems = SysMon::instance()->file_systems($file_system_class) ;
?>

<!DOCTYPE html">
<html>
<head>

<title>File System Analytics</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script> 
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>

<script type="text/javascript" src="../webfwk/js/config.js"></script>
<script type="text/javascript" src="../webfwk/js/Table.js"></script>

<style type="text/css">

td.table_hdr {
  background-color:#d0d0d0;
  padding: 2px 8px 2px 8px;
  border: solid 1px #000000;
  border-top: none;
  border-left: none;
  font-family: Arial, sans-serif;
  font-weight: bold;
  font-size: 12px;
}
td.table_cell {
  border:solid 1px #d0d0d0;
  border-top: none;
  border-left: none;
  padding: 2px 8px 2px 8px;
  font-family: Arial, sans-serif;
  font-size: 12px;
}
td.table_cell_left {
  font-weight: bold;
}
td.table_cell_right {
  border-right: none;
}
td.table_cell_bottom {
  border-bottom: none;
}
td.table_cell_within_group {
  border-bottom: none;
}

td.table_cell_highlight {
    background-color:#f0f0f0;
}

input {
  padding-left: 2px;
  padding-right: 2px;
}

.highlighted {
  font-weight:bold;
}

</style>

</head>

<script type="text/javascript">

var config = new config_create('FileSystemAnalytics') ;

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

function web_service_GET (url, params, on_success, on_error) {
    var jqXHR = $.get(url, params, function (data) {
        if (data.status != 'success') { report_error(data.message, null) ; return ; }
        on_success(data) ;
    },
    'JSON').error(function () {
        report_error('Web service request to '+url+' failed because of: '+jqXHR.statusText) ;
        on_error() ;
    }) ;
} ;

var file_system_class = '<?php echo is_null($file_system_class) ? '' : $file_system_class ; ?>' ;
var file_system_id = 0 ;
var num_loadings_in_progress = 0 ;

function disable_selectors () {
    if (num_loadings_in_progress++) return ;
    $('#file_system_selector'    ).attr('disabled', 'disabled') ;
    $('#file_system_selector4all').attr('disabled', 'disabled') ;
}
function enable_selectors () {
    if (--num_loadings_in_progress) return ;
    if (file_system_id) {
        $('#file_system_selector').removeAttr('disabled') ;
    }
    $('#file_system_selector4all').removeAttr('disabled') ;
}

function count_array_length (arr) {
    var result = 0 ;
    for (var i in arr) ++result ;
    return result ;
}

function load_summary_finished (data) {

    enable_selectors() ;

    $('#summary_file_systems').html(data ? count_array_length(data.file_systems) : '<?=$state2show_failed?>') ;
    $('#summary_files'       ).html(data ? data.files                            : '<?=$state2show_failed?>') ;
    $('#summary_directories' ).html(data ? data.directories                      : '<?=$state2show_failed?>') ;
    $('#summary_links'       ).html(data ? data.links                            : '<?=$state2show_failed?>') ;
    $('#summary_others'      ).html(data ? data.others                           : '<?=$state2show_failed?>') ;
    $('#summary_size'        ).html(data ? data.size                             : '<?=$state2show_failed?>') ;
}

var extensions_table = null ;

function load_extensions_finished (data) {

    enable_selectors() ;

    if (data) {
        $('#extensions div.workarea').html('<div id="extensions_table"></div>') ;
        var rows = [] ;
        var total = 0 ;
        for (var extension in data) {
            var e = data[extension] ;
            rows.push ([extension, e.num_files, e]) ;
            total++ ;
        }
        extensions_table = new Table (
            'extensions_table' ,
            [   {name: 'extension', type: Table.Types.Text} ,
                {name: '# files',   type: Table.Types.Number, align: 'right'} ,
                {name: 'size',      type: {
                    to_string     : function(a)   { return a.size; },
                    compare_values: function(a,b) { return this.compare_numbers(a.size_bytes,b.size_bytes); }},
                 align: 'right'}
            ] ,
            rows
        ) ;
        extensions_table.display() ;
        $('#summary_extensions').html(total) ;

    } else {
        $('#extensions div.workarea').html('<?=$state2show_failed?>') ;
        $('#summary_extensions').html('<?=$state2show_failed?>') ;
    }
}

var types_table = null ;

function load_types_finished (data) {

    enable_selectors() ;

    if (data) {
        $('#types div.workarea').html('<div id="types_table"></div>') ;
        var rows = [] ;
        var total  = 0 ;
        for (var i in data) {
            var e = data[i] ;
            rows.push ([e.name, e.num_files, e]) ;
            total++ ;
        }
        types_table = new Table (
            'types_table' ,
            [   {name: 'type',    type: Table.Types.Text} ,
                {name: '# files', type: Table.Types.Number, align: 'right'} ,
                {name: 'size',    type: {
                    to_string     : function(a)   { return a.size; },
                    compare_values: function(a,b) { return this.compare_numbers(a.size_bytes,b.size_bytes); }},
                 align: 'right'}
            ] ,
            rows
        ) ;
        types_table.display() ;
        $('#summary_types').html(total) ;

    } else {
        $('#types div.workarea').html('<?=$state2show_failed?>') ;
        $('#summary_types').html('<?=$state2show_failed?>') ;
    }
}

var sizes_table = null ;

function load_sizes_finished (data) {

    enable_selectors() ;

    if (data) {
        $('#sizes div.workarea').html('<div id="sizes_table"></div>') ;
        var rows = [] ;
        for (var i in data) {
            var e = data[i] ;
            rows.push ([e, e.num_files, e]) ;
        }
        sizes_table = new Table (
            'sizes_table' ,
            [   {name: 'file size',  type: {
                    to_string     : function(a)   { return a.max_size; } ,
                    compare_values: function(a,b) { return this.compare_numbers(a.max_size_bytes,b.max_size_bytes); }} ,
                 align: 'right'} ,
                {name: '# files',    type: Table.Types.Number, align: 'right'} ,
                {name: 'total size', type: {
                    to_string     : function(a)   { return a.size; } ,
                    compare_values: function(a,b) { return this.compare_numbers(a.size_bytes,b.size_bytes); }} ,
                 align: 'right'}
            ] ,
            rows
        ) ;
        sizes_table.display() ;

    } else {
        $('#sizes div.workarea').html('<?=$state2show_failed?>') ;
    }
}

var ctime_table = null ;

function load_ctime_finished (data) {

    enable_selectors() ;

    if (data) {
        $('#ctime div.workarea').html('<div id="ctime_table"></div>') ;
        var rows = [] ;
        for (var i in data) {
            var e = data[i] ;
            rows.push ([e, e.num_files, e]) ;
        }
        ctime_table = new Table (
            'ctime_table' ,
            [   {name: 'file age',  type: {
                    to_string     : function(a)   { return a.ctime_age; } ,
                    compare_values: function(a,b) { return this.compare_numbers(b.ctime_age_sec,a.ctime_age_sec); }} ,
                 align: 'right'} ,
                {name: '# files',    type: Table.Types.Number, align: 'right'} ,
                {name: 'total size', type: {
                    to_string     : function(a)   { return a.size; } ,
                    compare_values: function(a,b) { return this.compare_numbers(a.size_bytes,b.size_bytes); }} ,
                 align: 'right'}
            ] ,
            rows
        ) ;
        ctime_table.display() ;

    } else {
        $('#ctime div.workarea').html('<?=$state2show_failed?>') ;
    }
}


function load_summary () {

    disable_selectors() ;

    $('#summary_file_systems').html('<?=$state2show_loading?>') ;
    $('#summary_files'       ).html('<?=$state2show_loading?>') ;
    $('#summary_directories' ).html('<?=$state2show_loading?>') ;
    $('#summary_links'       ).html('<?=$state2show_loading?>') ;
    $('#summary_others'      ).html('<?=$state2show_loading?>') ;
    $('#summary_size'        ).html('<?=$state2show_loading?>') ;

    web_service_GET (
        '../sysmon/ws/filesystem_info.php' ,
        {file_system_class: file_system_class, id: file_system_id, scope: 'summary'} ,
        function (data) { load_summary_finished(data.summary) ; } ,
        function ()     { load_summary_finished() ; }
    ) ;
}

function load_extensions () {

    disable_selectors() ;

    $('#extensions div.workarea').html('<?=$state2show_loading?>') ;
    $('#summary_extensions').html('<?=$state2show_loading?>') ;

    web_service_GET (
        '../sysmon/ws/filesystem_info.php' ,
        {file_system_class: file_system_class, id: file_system_id, scope: 'extensions'} ,
        function (data) { load_extensions_finished(data.extensions) ; } ,
        function ()     { load_extensions_finished() ; }
    ) ;
}

function load_types () {

    disable_selectors() ;

    $('#types div.workarea').html('<?=$state2show_loading?>') ;
    $('#summary_types').html('<?=$state2show_loading?>') ;

    web_service_GET (
        '../sysmon/ws/filesystem_info.php' ,
        {file_system_class: file_system_class, id: file_system_id, scope: 'types'} ,
        function (data) { load_types_finished(data.types) ; } ,
        function ()     { load_types_finished() ; }
    ) ;
}

function load_sizes () {

    disable_selectors() ;

    $('#sizes div.workarea').html('<?=$state2show_loading?>') ;

    web_service_GET (
        '../sysmon/ws/filesystem_info.php' ,
        {file_system_class: file_system_class, id: file_system_id, scope: 'sizes'} ,
        function (data) { load_sizes_finished(data.sizes) ; } ,
        function ()     { load_sizes_finished() ; }
    ) ;
}

function load_ctime () {

    disable_selectors() ;

    $('#ctime div.workarea').html('<?=$state2show_loading?>') ;

    web_service_GET (
        '../sysmon/ws/filesystem_info.php' ,
        {file_system_class: file_system_class, id: file_system_id, scope: 'ctime'} ,
        function (data) { load_ctime_finished(data.ctime) ; } ,
        function ()     { load_ctime_finished() ; }
    ) ;
}

function load () {
    load_summary   () ;
    load_extensions() ;
    load_types     () ;
    load_sizes     () ;
    load_ctime     () ;
}

$(function () {
    $('#tabs').tabs() ;
    $('#file_system_selector').change(function () {
        file_system_id = $(this).val() ;
        if (file_system_id)
            load() ;
    }) ;
    $('#file_system_selector4all').change(function () {
        var select_all = $(this).attr('checked') ? true : false ;
        if (select_all) {
            file_system_id = 0 ;
            load() ;
        } else {
            file_system_id = $('#file_system_selector').val() ;
            if (file_system_id)
                load() ;
            else
                enable_selectors() ;
        }
    }) ;
}) ;

</script>

<body>

  <div style="padding:10px; padding-left:20px;">

    <h2>File System Analytics</h2>
    <div style="float:left; padding-left:10px; margin-bottom:30px;">
      Select a particular file system
      <select id="file_system_selector" >
        <option value="0"></option>
<?php   foreach ($filesystems as $base_path => $fs) {
            $file_system_id = $fs[0]['id'] ;
            print <<<HERE
        <option value="{$file_system_id}">{$base_path}</option>

HERE;
        }
?>
      </select>
      or check
      <input type="checkbox" id="file_system_selector4all"/>
      for all known file systems
    </div>
    <div id="loading_status" style="float:right; color:maroon;"></div>
    <div style="clear:both;"></div>

    <div id="tabs" style="padding-left:10px; font-size:12px;">

      <ul>
        <li><a href="#summary"    >Summary</a></li>
        <li><a href="#extensions" >Extensions</a></li>
        <li><a href="#types"      >Types</a></li>
        <li><a href="#sizes"      >Sizes</a></li>
        <li><a href="#ctime"      >Creation Time</a></li>
      </ul>

      <div id="summary" >
        <div class="workarea" style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <table><tbody>
            <tr>
              <td class="table_cell table_cell_left" >file systems</td>
              <td class="table_cell table_cell_right" id="summary_file_systems" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >regular files</td>
              <td class="table_cell table_cell_right" id="summary_files" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >directories</td>
              <td class="table_cell table_cell_right" id="summary_directories" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >symbolic links</td>
              <td class="table_cell table_cell_right" id="summary_links" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >other files</td>
              <td class="table_cell table_cell_right" id="summary_others" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >total size</td>
              <td class="table_cell table_cell_right" id="summary_size" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >extensions</td>
              <td class="table_cell table_cell_right" id="summary_extensions" ><?=$state2show_no_selection?></td>
            </tr>
            <tr>
              <td class="table_cell table_cell_left" >file types</td>
              <td class="table_cell table_cell_right" id="summary_types" ><?=$state2show_no_selection?></td>
            </tr>
          <tbody></table>
        </div>
      </div>

      <div id="extensions" >
        <div class="workarea" style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <?=$state2show_no_selection?>
        </div>
      </div>

      <div id="types" >
        <div class="workarea" style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <?=$state2show_no_selection?>
        </div>
      </div>

      <div id="sizes" >
        <div class="workarea" style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <?=$state2show_no_selection?>
        </div>
      </div>

      <div id="ctime" >
        <div class="workarea" style="padding:20px; border:solid 1px #b0b0b0; font-size:12px; " >
          <?=$state2show_no_selection?>
        </div>
      </div>

    </div>
  </div>

  <div id="popupdialogs" ></div>

</body>
</html>

<?php

    SysMon::instance()->commit() ;

} catch (Exception $e) { report_error($e.'<br><pre>'.print_r( $e->getTrace(), true ).'</pre>') ; }

?>