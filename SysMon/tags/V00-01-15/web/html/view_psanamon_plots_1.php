<?php

try {

    function report_error ($msg) {
        echo $msg ;
        exit(1) ;
    }

    $instr = isset($_GET['instr']) ? trim($_GET['instr']) : null ;
    if (!$instr) report_error('please, provide the name of the instrument') ;

    $instruments = array('AMO', 'SXR', 'XPP', 'XCS', 'CXI', 'MEC') ;
    if (!in_array($instr, $instruments)) report_error("invalid instrument name: {$instr}") ;

    $exper = isset($_GET['exper']) ? trim($_GET['exper']) : null ;
    if (!$exper) report_error('please, provide the name of the experiment') ;

    $plot = isset($_GET['plot']) ? trim($_GET['plot']) : null ;
    if (!$plot) report_error('please, provide the name of a plot') ;

?>
<!DOCTYPE html>
<html>

<head>

<title>psanamon plots for instrument <?= $instr ?></title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link rel="icon" href="/apps/webfwk/img/Portal_favicon.ico"/>

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>

<style>
    #plotter {
        padding:    10px;
    }
    #plotter > table#info {
        border-spacing:     0;
    }
    #plotter > table#info tr > td:first-child {
        padding-right:      4px;
        font-weight:        bold;
    }
    #plotter > table#info tr > td:last-child {
        padding-left:       4px;
    }
    div.plot-cont {
        margin-top:20px;
    }
    div.plot {
        padding:    5px;
        //border:     dashed 1px #b0b0b0;
        box-shadow: 0 5px 10px rgba(0,0,0,0.36);
    }
    div.plot img {
        width:      640px;
        height:     480px;
    }
    div.stat {
        margin-left:    20px;
    }
    div.stat * {
        font-weight:    bold;
        font-size:      32 px;
    }
    div.stat table td.val * {
        color:          maroon;
    }
</style>

<script lang="text/javascript" >

var instr = '<?=$instr?>' ;
var exper = '<?=$exper?>' ;
var plot  = '<?=$plot?>' ;

function FwkSimpleCreator (timer_ival_msec) {

    this._timer_ival_msec = timer_ival_msec ;

    this.report_error = function (msg) {
        console.log('FwkSimpleCreator: '+msg) ;
    } ;
    this.web_service_GET = function (url, params, on_success, on_failure) {
        var jqXHR = $.get(url, params, function (data) {
            if (data.status != 'success') {
                if (on_failure) on_failure(data.message) ;
                else            Fwk.report_error(data.message) ;
                return ;
            }
            if (on_success) on_success(data) ;
        },
        'JSON').error(function () {
            var message = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
            if (on_failure) on_failure(message) ;
            else            Fwk.report_error(message) ;
        }) ;
    } ;
    this.loader_GET = function (url, params, container, on_failure) {
        var jqXHR = $.get(url, params, function (data) {
            container.html(data) ;
        },
        'HTML').error(function () {
            var message = 'Document loading request to '+url+' failed because of: '+jqXHR.statusText ;
            if (on_failure) on_failure(message) ;
            else            Fwk.report_error(message) ;
        }) ;
    } ;
    this.now = function () {
        var date = new Date() ;
        var msec = date.getTime() ;
        return { date: date, sec: Math.floor(msec/1000), msec: msec % 1000 } ; 
    } ;

    this._handlers = {} ;

    this.register_handler   = function (name, f) { this._handlers[name] = f ; }
    this.unregister_handler = function (name)    { delete this._handlers[name] ; }

    this._notify_handers = function () {
        for (var name in this._handlers) {
            var handler = this._handlers[name] ;
            handler(this.now()) ;
        }
    } ;

    this._update_timer = null ;

    this._update_timer_restart = function () {
        this._update_timer = window.setTimeout('Fwk._update_timer_event()', this._timer_ival_msec) ;
    } ;
    this._update_timer_event = function () {
        this._notify_handers() ;
        this._update_timer_restart() ;
    } ;
    this._update_timer_restart() ;
}
var Fwk = new FwkSimpleCreator(20) ;   // 50 Hz

var update_in_progress = false ;
var prev_update_64 = 0 ;

function now2sec () {
    var now = Fwk.now() ;
    var sec_msec = now.sec + now.msec / 1000. ;
    return sec_msec ;
}

var prev_frames_time = now2sec() ;
var prev_frames      = 0 ;
var frames           = 0 ;

$(function () {

    Fwk.register_handler(plot, function (now) {
        if (update_in_progress) return ;
        update_in_progress = true ;
        var params = {
            instr_name: instr ,
            exper_name: exper ,
            name:       plot
        } ;
        Fwk.web_service_GET (
            'https://pswww.slac.stanford.edu/apps-dev/sysmon/ws/psanamon_plot_info.php' ,
            params ,
            function (data) {
                display_plot(data.plot) ;
                update_in_progress = false ;
            } ,
            function () {
                Fwk.report_error('failed to update the plot: '+plot) ;
                update_in_progress = false ;
            }
        ) ;
    }) ;
}) ;

function display_plot(plot_info) {

    /* Update the statistics
     */
    if (prev_update_64 !== plot_info.update_time.time64) frames++ ;

    var delta_f     = frames - prev_frames ;
    var frames_time = now2sec() ;
    var delta_t     = frames_time - prev_frames_time ;
    
    if (delta_t > 1) {

        var fps  = (delta_f / delta_t).toFixed(2) ;

        $('#stat').find('#frames').html(frames) ;
        $('#stat').find('#fps').html(fps) ;

        prev_frames_time = frames_time ;
        prev_frames      = frames ;

        console.log('updated:', frames_time, 'frames:', frames, 'fps:', fps) ;
    }

    /* Update the plot only if its timestamp differs from the
     * previous one.
     */
    if (prev_update_64 === plot_info.update_time.time64) return ;

    prev_update_64  = plot_info.update_time.time64 ;

    $('#plot').find('img').attr('src', 'load_plot/'+plot_info.id) ;
}

</script>

</head>

<body>
    <div id="plotter" >
        <table id="info" >
            <thead>
                <tr><td>Instrument:</td><td><?= $instr ?></td></tr>
                <tr><td>Experiment:</td><td><?= $exper ?></td></tr>
                <tr><td>Plot:      </td><td><?= $plot  ?></td></tr>
            </thead>
        </table>
        <div class="plot-cont" >
            <div id="plot" class="plot" style="float:left;" ><img/></div>
            <div id="stat" class="stat"  style="float:left;" >
                <table>
                    <tbody>
                        <tr><td class="key" >frames:</td><td class="val"><div id="frames" >0</div></td></tr>
                        <tr><td class="key" >FPS:   </td><td class="val"><div id="fps"    >0</div></td></tr>
                    </tbody>
                </table>
            <div style="clear:both;" ></div>
        </div>
    </div>
</body>

</html>

<?php
} catch (Exception $e) {
    print '<pre style="color:red;">'.print_r($e, true).'</pre>' ;
}
?>


