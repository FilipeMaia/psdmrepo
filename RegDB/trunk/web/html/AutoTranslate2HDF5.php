
<?php

require_once 'filemgr/filemgr.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use FileMgr\IfaceCtrlDb ;

use LogBook\LogBook ;
use LogBook\LogBookAuth ;

use RegDB\RegDB ;
use RegDB\RegDBAuth ;

$tables_html      = '' ;
$experiments2load = '' ;

try {
    
    $ifacectrl = IfaceCtrlDb::instance() ;
    $ifacectrl->begin() ;

    $logbook = LogBook::instance() ;
    $logbook->begin() ;

    $regdb = RegDB::instance() ;
    $regdb->begin() ;

    $can_modify = RegDBAuth::instance()->canEdit() ;

    foreach ($logbook->instruments() as $instrument) {

        if ($instrument->is_location()) continue ;

        $tables_html .= <<<HERE
<table><tbody>
  <tr>
    <td class="table_hdr">Instr</td>
    <td class="table_hdr">Exper</td>
    <td class="table_hdr">Id</td>
    <td class="table_hdr">#runs</td>
    <td class="table_hdr">#trans</td>
    <td class="table_hdr">Auto</td>
    <td class="table_hdr">FFB</td>
    <td class="table_hdr">release</td>
    <td class="table_hdr">config</td>
    <td class="table_hdr">Actions</td>
    <td class="table_hdr">Comments</td>
  </tr>
HERE;

        foreach ($logbook->experiments_for_instrument($instrument->name()) as $experiment) {

            if ($experiment->is_facility()) continue ;

            $is_authorized = $can_modify || LogBookAuth::instance()->canPostNewMessages($experiment->id()) ;

            $num_runs = $experiment->num_runs() ;
            $num_runs_str = '' ;
            $loading_comment = '' ;
            if ($num_runs) {
                $num_runs_str = $num_runs ;
                $loading_comment = 'Loading...' ;
                if ($experiments2load == '')
                    $experiments2load = "var experiments2load=[{$experiment->id()}" ;
                else
                    $experiments2load .= ",{$experiment->id()}" ;
            }
            $service = 'STANDARD' ;
            $auto    = $experiment->regdb_experiment()->find_param_by_name(IfaceCtrlDb::$AUTO_TRANSLATE_HDF5[$service]) ? true : false ;
            $dataset = $ifacectrl->get_config_param_val_r('live-mode', 'dataset', $instrument->name(), $experiment->name()) ;
            if (is_null($dataset)) {
                print "ERROR: Interface Controller database has invalid content" ;
                exit (1) ;
            }
            $ffb = $dataset === IfaceCtrlDb::$DATASET_FFB ? true : false ;

            $release_dir = $ifacectrl->get_config_param_val_r('', 'release', $instrument->name(), $experiment->name()) ;
            if (is_null($release_dir)) {
                print "ERROR: Interface Controller database has invalid content" ;
                exit (1) ;
            }
            $config_file = $ifacectrl->get_config_param_val_r('', 'config', $instrument->name(), $experiment->name()) ;
            if (is_null($config_file)) {
                print "ERROR: Interface Controller database has invalid content" ;
                exit (1) ;
            }

            $tables_html .= <<<HERE
  <tr>
    <td class="table_cell">{$experiment->instrument()->name()}</td>
    <td class="table_cell"><a target="_blank" href="../portal/index.php?exper_id={$experiment->id()}&app=hdf:manage" title="open Web Portal of the Experiment in new window/tab">{$experiment->name()}</a></td>
    <td class="table_cell">{$experiment->id()}</td>
    <td class="table_cell">{$num_runs_str}</td>
    <td class="table_cell"><span id="num_translated_{$experiment->id()}"}>{$loading_comment}</td>
HERE;
            if ($is_authorized) {
                $auto_str = $auto ? 'checked="checked"' : '' ;
                $ffb_str  = $ffb  ? 'checked="checked"' : '' ;
                $tables_html .= <<<HERE
    <td class="table_cell"><input type="checkbox" class="params auto"        name="{$experiment->id()}" value=1 {$auto_str} /></td>
    <td class="table_cell"><input type="checkbox" class="params ffb"         name="{$experiment->id()}" value=1 {$ffb_str}  /></td>
    <td class="table_cell"><input type="text"     class="params release_dir" name="{$experiment->id()}" value="{$release_dir}" size="42" /></td></td>
    <td class="table_cell"><input type="text"     class="params config_file" name="{$experiment->id()}" value="{$config_file}" size="36" /></td></td>
    <td class="table_cell">{$config}</td>
    <td class="table_cell"><button id="{$experiment->id()}" disabled="disabled">Save</button></td>
    <td class="table_cell table_cell_right"><span id="comment_{$experiment->id()}"}></span></td>
HERE;
            } else {
                $auto_str = '<div style="height:8px; width:8px; '.($auto ? 'background-color:red;' : '').'" >&nbsp;</div>' ;
                $ffb_str  = '<div style="height:8px; width:8px; '.($ffb  ? 'background-color:red;' : '').'" >&nbsp;</div>' ;
                $tables_html .= <<<HERE
    <td class="table_cell">{$auto_str}</td>
    <td class="table_cell">{$ffb_str}</td>
    <td class="table_cell">{$release_dir}</td>
    <td class="table_cell">{$config_file}</td>
    <td class="table_cell">&nbsp;</td>
    <td class="table_cell table_cell_right">&nbsp;</td>
HERE;
            }
            $tables_html .= <<<HERE
  </tr>
HERE;
        }
        $tables_html .= <<<HERE
</tbody><table>
HERE;
    }
    if ($experiments2load == '') $experiments2load = "var experiments2load=[];\n" ;
    else                         $experiments2load .= "];\n" ;

} catch (Exception $e) { print '<pre>'.print_r($e, true).'</pre>' ; }

?>

<!-- The script for reporting and optionally modifying the auto-translation
     option for HDF5 files of experiments. -->

<!DOCTYPE html>
<html>
<head>

<title>Report and modify automatic translation (XTC to HDF5) parameter for known experiments </title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script> 

<style type="text/css">

body {
    margin:             0;
    padding:            0;
    font-family:        'Source Sans Pro',Arial,sans-serif;
    font-size:          14px;
}
h2 {
    font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
p {
    /*
    font-family:  Lucida Grande, Lucida Sans, Arial, sans-serif;
    */
    font-family:    'Source Sans Pro',Arial,sans-serif;
    font-size:      14px;
    line-height:    1.4;
}
td.table_hdr {
    background-color: #d0d0d0;
    padding:            2px 8px 2px 8px;
    border:             solid 1px #000000;
    border-top:         none;
    border-left:        none;
    /*
    font-family: Arial, sans-serif;
    */
    font-family:        Lucida Grande, Lucida Sans, Arial, sans-serif;
    font-weight:        bold;
    font-size:          13px;
}
td.table_cell {
    border:             solid 1px #d0d0d0;
    border-top:         none;
    border-left:        none;
    padding:            2px 8px 2px 8px;
    font-family:        Arial, sans-serif;
    font-size:          13px;

}
td.table_cell_left {
    font-weight:        bold;
}
td.table_cell_right {
    border-right:       none;
}
td.table_cell_bottom {
    border-bottom:      none;
}
td.table_cell_within_group {
    border-bottom:      none;
}

input {
    padding-left:       2px;
    padding-right:      2px;
    border:             solid 1px #ffffff;
}
input[type="text"]:hover {
    border:             solid 1px #d0d0d0;
}

#descr {
    max-width:          640px;
}
#descr table {
    margin-top:         10px;
    margin-left:        10px;
}
#descr table td.key {
    font-weight:        bold;
    padding-right:      10px;
}
#descr table td.val {
    /*
    font-family:        Lucida Grande, Lucida Sans, Arial, sans-serif;
    */
    font-family:        'Source Sans Pro',Arial,sans-serif;
    font-size:          14px;
    line-height:        1.4;
    padding-bottom:     7px;
}

</style>

<script type="text/javascript">

<?php echo $experiments2load ; ?>

function load_hdf5_files (exper_id) {
    $.ajax ({
        type: 'GET' ,
        url: '../portal/ws/filemgr_files_search.php' ,
        data: {
            exper_id: exper_id ,
            types: 'hdf5'
        } ,
        success: function (data) {
            if (data.Status != 'success') {
                alert(data.Message) ;
                return ;
            }
            var num_files = 0;
            for( var i in data.runs ) {
                num_files += data.runs[i].files.length;
            }
            $('#num_translated_'+exper_id).html(num_files?num_files:'');
        },
        error: function() { alert('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    }) ;
}

function load_translation_config (exper_id) {
    $.ajax ({
        type: 'GET' ,
        url: '../portal/ws/hdf5_config_get.php' ,
        data: {
            exper_id: exper_id
        } ,
        success: function (data) {
            if (data.status != 'success') {
                alert(data.message) ;
                return ;
            }

            $('#num_translated_'+exper_id).html(num_files?num_files:'');
        },
        error: function() { alert('The request can not go through due a failure to contact the server.'); },
        dataType: 'json'
    }) ;
}
$(function () {

    $('button').button().click(function () {

        var exper_id    = this.id ;
        var auto        = $('input.auto[name="'+exper_id+'"]').is(':checked') ;
        var ffb         = $('input.ffb[name="'+exper_id+'"]').is(':checked') ;
        var release_dir = $('input.release_dir[name="'+exper_id+'"]').val() ;
        var config_file = $('input.config_file[name="'+exper_id+'"]').val() ;

        $('button#'+exper_id).button('disable') ;
        $('#comment_'+exper_id).text('saving...') ;

        $.ajax ({
            type : 'POST' ,
            url  : '../portal/ws/hdf5_config_set.php' ,
            data : {
                exper_id:    exper_id ,
                auto:        auto ? 1 : 0 ,
                ffb:         ffb  ? 1 : 0 ,
                release_dir: release_dir ,
                config_file: config_file
            } ,
            success: function (data) {
                if (data.Status != 'success') {
                    $('#comment_'+exper_id).text(data.Message) ;
                    return ;
                }
                $('#comment_'+exper_id).text('saved') ;
            } ,
            error: function () {
                $('button#'+exper_id).button('enable') ;
                $('#comment_'+exper_id).text('failed to submit the request') ;
            } ,
            dataType: 'json'
        }) ;

    }) ;
    $('input.params').change(function () {
        var name = this.name ;
        $('button#'+name).button('enable') ;
    }) ;

    // Begin asynchronious loading of the number of HDF5 files for each
    // experiment which had at least one run taken.
    //
    for (var i in experiments2load) {
       var exper_id = experiments2load[i] ;
       load_hdf5_files        (exper_id) ;
       load_translation_config(exper_id) ;
    }
}) ;

</script>

</head>
    <body>
        <div style="padding:20px;" >

            <h2>View/Modify HDF5 translation options across all experiments</h2>

            <div id="descr" >
                
                <p>This tool is mean to view and (if your account has sufficient privileges) to modify
                values of the following parameters of experiments:
                </p>

                <table>
                    <tbody>
                        <tr><td class="key" valign="top" >Auto</td>
                            <td class="val" >
                                automatically translate regular XTC streams of an experiment as they're
                                produced by the DAQ system and migrated to FFB or OFFLINE storage
                            </td>
                        </tr>
                        <tr><td class="key" valign="top" >FFB</td>
                            <td class="val" >
                                read XTC files as they show up at the FFB storage <b>/reg/d/ffb/</b> instead of
                                from the OFFLINE storage <b>/reg/d/psdm/</b>.
                            </td>
                        </tr>
                        <tr><td class="key" valign="top" >release</td>
                            <td class="val" >
                                an absolute path to an analysis software release from which to run
                                the Translator.
                           </td>
                        </tr>
                        <tr><td class="key" valign="top" >config</td>
                            <td class="val" >
                                a relative path to a <b>psana</b> configuration file for the Translator
                                application. Note that the path is relative to the release directory.
                           </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div style="margin-top:20px; padding-left:10px;"><?php echo $tables_html; ?></div>
        </div>
    </body>
</html>
