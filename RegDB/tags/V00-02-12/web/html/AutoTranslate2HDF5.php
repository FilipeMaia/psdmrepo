<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once 'filemgr/filemgr.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'regdb/regdb.inc.php' ;

use FileMgr\IfaceCtrlDb ;

use LogBook\LogBook ;
use LogBook\LogBookAuth ;

use RegDB\RegDB ;
use RegDB\RegDBAuth ;


$service = 'STANDARD' ;
if (isset($_GET['service'])) {
    $service = strtoupper(trim($_GET['service'])) ;
    switch ($service) {
        case 'STANDARD' :
        case 'MONITORING' : break ;
        default :
            echo "illegal value of the 'service' parameter" ;
            exit(1) ;
    }
}

$tables_html      = '' ;
$experiments2load = array() ;

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
HERE;
        if ($service === 'MONITORING') $tables_html .= <<<HERE
    <td class="table_hdr">#jobs</td>
    <td class="table_hdr">Output folder</td>
    <td class="table_hdr">CC in subdir</td>
    <td class="table_hdr">live timeout</td>
HERE;
        $tables_html .= <<<HERE
    <td class="table_hdr">Actions</td>
    <td class="table_hdr">Comments</td>
  </tr>
HERE;

        foreach ($logbook->experiments_for_instrument($instrument->name()) as $experiment) {

            if ($experiment->is_facility()) continue ;

            $exper_id      = $experiment->id() ;
            $is_authorized = $can_modify || LogBookAuth::instance()->canPostNewMessages($exper_id) ;

            array_push (
                $experiments2load ,
                array (
                    'id'            => $exper_id ,
                    'is_authorized' => $is_authorized ? 1 : 0)) ;

            $num_runs     = $experiment->num_runs() ;
            $num_runs_str = $num_runs ? $num_runs : '' ;

            $tables_html .= <<<HERE
  <tr>
    <td class="table_cell">{$experiment->instrument()->name()}</td>
    <td class="table_cell"><a target="_blank" href="../portal/index.php?exper_id={$exper_id}&app=hdf:manage" title="open Web Portal of the Experiment in new window/tab">{$experiment->name()}</a></td>
    <td class="table_cell">{$exper_id}</td>
    <td class="table_cell">{$num_runs_str}</td>
    <td class="table_cell"><span id="num_translated_{$exper_id}"}>Loading...</td>
HERE;
            if ($is_authorized) {
                $tables_html .= <<<HERE
    <td class="table_cell"><input type="checkbox" class="params auto"        name="{$exper_id}" /></td>
    <td class="table_cell"><input type="checkbox" class="params ffb"         name="{$exper_id}" /></td>
    <td class="table_cell"><input type="text"     class="params release_dir" name="{$exper_id}" value="Loading..." size="32" /></td></td>
    <td class="table_cell"><input type="text"     class="params config_file" name="{$exper_id}" value="Loading..." size="32" /></td></td>
    <td class="table_cell"><input type="text"     class="params njobs"       name="{$exper_id}" value="Loading..." size="2" /></td></td>
    <td class="table_cell"><input type="text"     class="params outdir"      name="{$exper_id}" value="Loading..." size="32" /></td></td>
    <td class="table_cell"><input type="checkbox" class="params ccinsubdir"  name="{$exper_id}" /></td>
    <td class="table_cell"><input type="text"     class="params livetimeout" name="{$exper_id}" value="Loading..." size="4" /></td></td>
    <td class="table_cell"><button id="{$exper_id}" disabled="disabled">SAVE</button></td>
    <td class="table_cell table_cell_right"><span id="comment_{$exper_id}"}></span></td>
HERE;
            } else {
                $tables_html .= <<<HERE
    <td class="table_cell" ><div class="params auto"        name="{$exper_id}" style="height:8px; width:8px;" >&nbsp;</div></td>
    <td class="table_cell" ><div class="params ffb"         name="{$exper_id}" style="height:8px; width:8px;" >&nbsp;</div></td>
    <td class="table_cell" ><div class="params release_dir" name="{$exper_id}"                                >Loading...</div></td>
    <td class="table_cell" ><div class="params config_file" name="{$exper_id}"                                >Loading...</div></td>
    <td class="table_cell" ><div class="params njobs"       name="{$exper_id}"                                >Loading...</div></td>
    <td class="table_cell" ><div class="params outdir"      name="{$exper_id}"                                >Loading...</div></td>
    <td class="table_cell" ><div class="params ccinsubdir"  name="{$exper_id}" style="height:8px; width:8px;" >&nbsp;</div></td>
    <td class="table_cell" ><div class="params livetimeout" name="{$exper_id}"                                >Loading...</div></td>
    <td class="table_cell" >&nbsp;</td>
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

<?php echo 'var service=\''.$service.'\';' ; ?>
<?php echo 'var experiments2load='.json_encode($experiments2load).';' ; ?>

function load_hdf5_requests (exper_id) {
    $.ajax ({
        type: 'GET' ,
        url: '../portal/ws/hdf5_requests_get.php' ,
        data: {
            service:  service ,
            exper_id: exper_id ,
            status:   'FINISHED'
        } ,
        success: function (data) {
            if (data.Status != 'success') {
                alert(data.Message) ;
                return ;
            }
            var num_finished_requests = data.requests.length ;
            $('#num_translated_'+exper_id).html(num_finished_requests ? num_finished_requests : '') ;
        } ,
        error: function () {
            $('span#comment_'+exper_id).html('The request can not go through due a failure to contact the server.') ;
        } , 
        dataType: 'json'
    }) ;
}

function load_translation_config (exper_id, is_authorized) {
    $.ajax ({
        type: 'GET' ,
        url: '../portal/ws/hdf5_config_get.php' ,
        data: {
            service:  service ,
            exper_id: exper_id
        } ,
        success: function (data) {
            if (data.status != 'success') {
                alert(data.message) ;
                return ;
            }
            if (is_authorized) {
                var auto = $('input.auto[name="'+exper_id+'"]') ;
                if (data.config.auto) auto.attr('checked', 'checked') ;
                else                  auto.removeAttr('checked') ;

                var ffb = $('input.ffb[name="'+exper_id+'"]') ;
                if (data.config.ffb) ffb.attr('checked', 'checked') ;
                else                 ffb.removeAttr('checked') ;

                $('input.release_dir[name="'+exper_id+'"]').val(data.config.release_dir) ;
                $('input.config_file[name="'+exper_id+'"]').val(data.config.config_file) ;

                switch (service) {
                    case 'MONITORING' :

                        $('input.njobs[name="'+exper_id+'"]').val(data.config.njobs) ;
                        $('input.outdir[name="'+exper_id+'"]').val(data.config.outdir) ;

                        var ccinsubdir = $('input.ccinsubdir[name="'+exper_id+'"]') ;
                        if (data.config.ccinsubdir) ccinsubdir.attr('checked', 'checked') ;
                        else                        ccinsubdir.removeAttr('checked') ;

                        $('input.livetimeout[name="'+exper_id+'"]').val(data.config.livetimeout) ;
                        
                        break ;
                }

            } else {
                if (data.config.auto) $('div.auto[name="'+exper_id+'"]').css('background-color', 'red') ;
                if (data.config.ffb)  $('div.ffb[name="'+exper_id+'"]').css('background-color', 'red') ;
                $('div.release_dir[name="'+exper_id+'"]').html(data.config.release_dir) ;
                $('div.config_file[name="'+exper_id+'"]').html(data.config.config_file) ;
                $('div.njobs[name="'+exper_id+'"]').html(data.config.njobs) ;
                $('div.outdir[name="'+exper_id+'"]').html(data.config.outdir) ;
                if (data.config.ccinsubdir)  $('div.ccinsubdir[name="'+exper_id+'"]').css('background-color', 'red') ;
                $('div.livetimeout[name="'+exper_id+'"]').html(data.config.livetimeout) ;
            }
        } ,
        error: function () {
            $('span#comment_'+exper_id).html('The request can not go through due a failure to contact the server.') ;
        } ,
        dataType: 'json'
    }) ;
}
$(function () {

    $('button').button().click(function () {

        var exper_id = this.id ;

        var params = {
            service :     service ,
            exper_id :    exper_id ,
            auto :        $('input.auto[name="'+exper_id+'"]').is(':checked') ? 1 : 0 ,
            ffb  :        $('input.ffb[name="'+exper_id+'"]').is(':checked') ? 1 : 0 ,
            release_dir : $('input.release_dir[name="'+exper_id+'"]').val() ,
            config_file : $('input.config_file[name="'+exper_id+'"]').val()
        } ;
        switch (service) {
            case 'MONITORING' :
                params.njobs       = $('input.njobs[name="'+exper_id+'"]').val() ;
                params.outdir      = $('input.outdir[name="'+exper_id+'"]').val() ;
                params.ccinsubdir  = $('input.ccinsubdir[name="'+exper_id+'"]').is(':checked') ? 1 : 0 ;
                params.livetimeout = $('input.livetimeout[name="'+exper_id+'"]').val() ;
                break ;
        }

        $('button#'+exper_id).button('disable') ;
        $('#comment_'+exper_id).text('saving...') ;

        $.ajax ({
            type : 'POST' ,
            url  : '../portal/ws/hdf5_config_set.php' ,
            data : params ,
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
        var exper = experiments2load[i] ;
        load_hdf5_requests     (exper.id) ;
        load_translation_config(exper.id, exper.is_authorized) ;
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
