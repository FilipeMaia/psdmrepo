/*
 * ======================================================
 *  Application: Run Tables
 *  DOM namespace of classes and identifiers: runtables-
 *  JavaScript global names begin with: runtables-
 * ======================================================
 */

function runtables_create () {

    var that = this ;

    /* ---------------------------------------
     *  This application specific environment
     * ---------------------------------------
     *
     * NOTE: These variables should be initialized externally.
     */
    this.exp_id  = null ;

    /* The context for v-menu items
     */
    var context2_default = {
        'calib' : '' ,
        'detectors' : ''
    } ;
    this.name = 'runtables' ;
    this.full_name = 'Run Tables' ;
    this.context1 = 'calib' ;
    this.context2 = '' ;
    this.select_default = function () { this.select(this.context1, this.context2) ; } ;
    this.select = function (ctx1, ctx2) {
        this.init() ;
        this.context1 = ctx1 ;
        this.context2 = ctx2 == null ? context2_default[ctx1] : ctx2 ;
        switch (this.context1) {
            case 'calib':     this.calib_update() ; break ;
            case 'detectors': this.detectors_update() ; break ;
        }
    } ;

    /* ---------------------------------------
     *  Menu item: Calibration control table
     * ---------------------------------------
     */
    this.calib_runs = null ;

    this.calib_refresh_button = null ;
    this.calib_reset_button = null ;
    this.calib_from_run = null ;
    this.calib_through_run = null ;

    this.calib_table = null ;

    this.calib_init = function () {

        var that = this ;

        this.calib_refresh_button = $('#runtables-calib').find('button[name="refresh"]').button() ;
        this.calib_reset_button   = $('#runtables-calib').find('button[name="reset"]')  .button() ;
        this.calib_from_run       = $('#runtables-calib').find('input[name="from"]') ;
        this.calib_through_run    = $('#runtables-calib').find('input[name="through"]') ;

        this.calib_refresh_button.click (function () { that.calib_update() ;  }) ;
        this.calib_from_run      .change(function () { that.calib_update() ;  }) ;
        this.calib_through_run   .change(function () { that.calib_update() ;  }) ;

        this.calib_reset_button.click (function () {
            that.calib_from_run.val('') ;
            that.calib_through_run.val('') ;
            that.calib_update() ;
        }) ;

        var table_cont = $('#runtables-calib').find('div#table') ;
        var hdr = [
            {   name: 'Run', type: Table.Types.Number_HTML } ,
            {   name: 'Dark', sorted: false ,
                type: {
                    after_sort: function () {
                        table_cont.find('div.runtables-calib-dark').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.html('<span style="font-size:150%;">'+(calib.dark ? '&diams;' : '&nbsp;')+'</span>') ;
                        }) ;
                        table_cont.find('input.runtables-calib-dark').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            if (calib.dark) elem.attr('checked', 'checked') ;
                            else            elem.removeAttr('checked') ;
                        }) ;
                    }
                }
            } ,
            {   name: 'Flat', sorted: false ,
                type: {
                    after_sort: function () {
                        table_cont.find('div.runtables-calib-flat').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.html('<span style="font-size:150%;">'+(calib.flat ? '&diams;' : '&nbsp;')+'</span>') ;
                        }) ;
                        table_cont.find('input.runtables-calib-flat').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            if (calib.flat) elem.attr('checked', 'checked') ;
                            else            elem.removeAttr('checked') ;
                        }) ;
                    }
                }
            } ,
            {   name: 'Geometry', sorted: false ,
                type: {
                    after_sort: function () {
                        table_cont.find('div.runtables-calib-geom').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.html('<span style="font-size:150%;">'+(calib.geom ? '&diams;' : '&nbsp;')+'</span>') ;
                        }) ;
                        table_cont.find('input.runtables-calib-geom').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            if (calib.geom) elem.attr('checked', 'checked') ;
                            else            elem.removeAttr('checked') ;
                        }) ;
                    }
                }
            } ,
            {   name: 'Comment', sorted: false , hideable: true ,
                type: {
                    after_sort: function () {
                        table_cont.find('div.runtables-calib-comment').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.html('<pre>'+calib.comment+'</pre>') ;
                        }) ;
                        table_cont.find('textarea.runtables-calib-comment').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.text(calib.comment) ;
                        }) ;
                    }
                }
            } ,
            {   name: 'ACTIONS', sorted: false , hideable: true ,
                type: {
                    after_sort: function () {
                        table_cont.find('button.edit_run').
                            button().
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                table_cont.find('button.cancel_run#'+run).button('enable') ;
                                table_cont.find('button.save_run#'+run).button('enable') ;


                                table_cont.find('.view#'+run).addClass('runtables-calib-hdn').removeClass('runtables-calib-vis') ;
                                table_cont.find('.edit#'+run).addClass('runtables-calib-vis').removeClass('runtables-calib-hdn') ;
                            }
                        ) ;
                        table_cont.find('button.cancel_run').
                            button().
                            button('disable').
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                table_cont.find('button.edit_run#'+run).button('enable') ;
                                table_cont.find('button.save_run#'+run).button('disable') ;

                                table_cont.find('.view#'+run).addClass('runtables-calib-vis').removeClass('runtables-calib-hdn') ;
                                table_cont.find('.edit#'+run).addClass('runtables-calib-hdn').removeClass('runtables-calib-vis') ;
                            }
                        ) ;
                        table_cont.find('button.save_run').
                            button().
                            button('disable').
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                table_cont.find('button.edit_run#'+run).button('enable') ;
                                table_cont.find('button.cancel_run#'+run).button('disable') ;

                                table_cont.find('.view#'+run).addClass('runtables-calib-vis').removeClass('runtables-calib-hdn') ;
                                table_cont.find('.edit#'+run).addClass('runtables-calib-hdn').removeClass('runtables-calib-vis') ;

                                var dark = table_cont.find('input.runtables-calib-dark#'+run).attr('checked') ? 1 : 0 ;
                                var flat = table_cont.find('input.runtables-calib-flat#'+run).attr('checked') ? 1 : 0 ;
                                var geom = table_cont.find('input.runtables-calib-geom#'+run).attr('checked') ? 1 : 0 ;
                                var comment = table_cont.find('textarea#'+run).val() ;

                                that.calib_save(run, dark, flat, geom, comment) ;
                            }
                        ) ;
                    }
                 }
             }
        ] ;
        this.calib_table = new Table (
            table_cont ,
            hdr ,
            null ,
            { default_sort_forward: false } ,
            config.handler('runtables', 'calibrations')
        ) ;
        this.calib_update() ;
    } ;

    this.calib_display = function () {
        var title = 'show the run in the e-Log Search panel within the current Portal' ;
        var rows = [] ;
        for (var run in this.calib_runs) {
            rows.push([
                {   number: run ,
                    html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'
                } ,
                '<div class="runtables-calib-dark-cont" >' +
                '  <div      class="runtables-calib-dark view runtables-calib-vis" id="'+run+'" ></div>' +
                '  <input    class="runtables-calib-dark edit runtables-calib-hdn" id="'+run+'" type="checkbox" />' +
                '</div>',
                '<div class="runtables-calib-flat-cont" >' +
                '  <div      class="runtables-calib-flat view runtables-calib-vis" id="'+run+'" ></div>' +
                '  <input    class="runtables-calib-flat edit runtables-calib-hdn" id="'+run+'" type="checkbox" />' +
                '</div>',
                '<div class="runtables-calib-geom-cont" >' +
                '  <div      class="runtables-calib-geom view runtables-calib-vis" id="'+run+'" ></div>' +
                '  <input    class="runtables-calib-geom edit runtables-calib-hdn" id="'+run+'" type="checkbox" />' +
                '</div>',
                '<div class="runtables-calib-comment-cont" >' +
                '  <div      class="runtables-calib-comment view runtables-calib-vis" id="'+run+'" ></div>' +
                '  <textarea class="runtables-calib-comment edit runtables-calib-hdn" id="'+run+'" rows="2" cols="56" ></textarea>' +
                '</div>',
                '<button class="edit_run"   id="'+run+'" title="edit calibration properties of this run"  >Edit</>' +
                '<button class="cancel_run" id="'+run+'" title="discard modifications (if any were made)" >Cancel</>' +
                '<button class="save_run"   id="'+run+'" title="save modifications to the database"       >Save</>'
            ]) ;
        }
        this.calib_table.load(rows) ;
        this.calib_table.display() ;
    } ;

    this.calib_update = function () {
        $('#runtables-calib').find('.runtables-info#updated').html('Loading...') ;
        web_service_GET (
            '../portal/ws/runtable_calib_get.php' ,
            {   exper_id:    this.exp_id ,
                from_run:    parseInt(this.calib_from_run.val()) ,
                through_run: parseInt(this.calib_through_run.val())
            } ,
            function (data) {
                that.calib_runs = data.runs ;
                that.calib_display() ;
                $('#runtables-calib').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;

    this.calib_save = function (run, dark, flat, geom, comment) {
        $('#runtables-calib').find('.runtables-info#updated').html('Saving...') ;
        web_service_POST (
            '../portal/ws/runtable_calib_save.php' , {
                exper_id: this.exp_id ,
                run: run ,
                dark: dark,
                flat: flat,
                geom: geom,
                comment: comment
            } ,
            function (data) {
                that.calib_runs[run] = data.runs[run] ;
                that.calib_display() ;
                $('#runtables-calib').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;

    /* ----------------------------------------
     *  Menu item: DAQ Detectors control table
     * ----------------------------------------
     */
    this.detectors_runs = null ;

    this.detectors_refresh_button = null ;
    this.detectors_reset_button = null ;
    this.detectors_from_run = null ;
    this.detectors_through_run = null ;

    this.detectors_table = null ;

    this.detectors_init = function () {

        var that = this ;

        this.detectors_refresh_button = $('#runtables-detectors').find('button[name="refresh"]').button() ;
        this.detectors_reset_button   = $('#runtables-detectors').find('button[name="reset"]')  .button() ;
        this.detectors_from_run       = $('#runtables-detectors').find('input[name="from"]') ;
        this.detectors_through_run    = $('#runtables-detectors').find('input[name="through"]') ;

        this.detectors_refresh_button.click (function () { that.detectors_update() ;  }) ;
        this.detectors_from_run      .change(function () { that.detectors_update() ;  }) ;
        this.detectors_through_run   .change(function () { that.detectors_update() ;  }) ;

        this.detectors_reset_button.click (function () {
            that.detectors_from_run.val('') ;
            that.detectors_through_run.val('') ;
            that.detectors_update() ;
        }) ;

        this.detectors_show_all_button = $('#runtables-detectors').find('button[name="show_all"]').button().button('disable') ;
        this.detectors_hide_all_button = $('#runtables-detectors').find('button[name="hide_all"]').button().button('disable') ;
        this.detectors_advanced_button = $('#runtables-detectors').find('button[name="advanced"]').button().button('disable') ;

        this.detectors_show_all_button.click(function() { that.detectors_table.display('show_all') ; }) ;
        this.detectors_hide_all_button.click(function() { that.detectors_table.display('hide_all') ; }) ;
        this.detectors_advanced_button.click(function() { that.detectors_advanced() ; }) ;

        this.detectors_update() ;
    } ;
    this.detectors_advanced = function () {

        var detectors_selector = $('#largedialogs') ;

        var html =
'<div style="overflow:auto;">' +
'  <table><tbody>' ;
        var detectors_per_row = 5 ;
        var num_detectors = 0 ;
        var header_info = this.detectors_table.header_info() ;
        for (var i in header_info) {
            var col = header_info[i] ;
            if (col.hideable) {
                if (!num_detectors) {
                    html +=
'    <tr>' ;
                } else if (!(num_detectors % detectors_per_row)) {
                    html +=
'    </tr>' +
'    <tr>' ;
                }
                html +=
'      <td class="table_cell table_cell_borderless">' +
'        <div style="float:left;"><input type="checkbox" class="detector" name="'+col.number+'" '+(col.hidden?'':'checked="checked"')+' /></div>' +
'        <div style="float:left; margin-left:5px; font-weight:bold;">'+col.name+'</div>' +
'        <div style="clear:both;"></div>' +
'      </td>' ;
                num_detectors++ ;
            }
        }
        if (num_detectors % detectors_per_row) {
            html +=
'    </tr>' ;
        }
        html +=
'  </tbody></table>' ;
'</div>';
        detectors_selector.html (html) ;
        detectors_selector.dialog ({
            minHeight: 240 ,
            width: 720 ,
            resizable: true ,
            modal: true ,
            buttons: {
                "Close": function() {
                    $( this ).dialog('close') ;
                }
            } ,
            title: 'Switch on/off detectors'
        });
        detectors_selector.find('input.detector').change(function() {
            var col_number = this.name ;
            that.detectors_table.display(this.checked ? 'show' : 'hide', parseInt(col_number)) ;
        }) ;
    } ;
    this.detectors_display = function () {
        var table_cont = $('#runtables-detectors').find('div#table') ;
        var hdr = [
            {name: 'RUN', type: Table.Types.Number_HTML}
        ] ;
        for (var name in that.detectors) {
            var det_dev = name.split('|') ;
            var det = det_dev[0] ;
            var dev = det_dev[1] ;
            hdr.push({
                name: '<div><div>'+det+'</div><div>'+dev+'</div></div>' ,
                hideable: true ,
                align: 'center' ,
                style: ' white-space: nowrap;'
            }) ;
        }
        this.detectors_table = new Table (
            table_cont ,
            hdr ,
            null ,
            { default_sort_forward: false } ,
            config.handler('runtables', 'detectors')
        ) ;
        var title = 'show the run in the e-Log Search panel within the current Portal' ;
        var rows = [] ;
        for (var run in this.detectors_runs) {
            var row = [] ;
            row.push(
                {   number: run ,
                    html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'
                }
            ) ;
            var run_detectors = this.detectors_runs[run] ;
            for (var name in that.detectors) {
                row.push('<span style="font-size:150%;">'+(run_detectors[name] ? '&diams;' : '&nbsp;')+'</span>')  ;
            }
            rows.push(row) ;
        }
        this.detectors_table.load(rows) ;
        this.detectors_table.display() ;
    } ;

    this.detectors_update = function () {
        $('#runtables-detectors').find('.runtables-info#updated').html('Loading...') ;
        that.detectors_show_all_button.button('disable') ;
        that.detectors_hide_all_button.button('disable') ;
        that.detectors_advanced_button.button('disable') ;
        web_service_GET (
            '../portal/ws/runtable_detectors_get.php' ,
            {   exper_id:    this.exp_id ,
                from_run:    parseInt(this.detectors_from_run.val()) ,
                through_run: parseInt(this.detectors_through_run.val())
            } ,
            function (data) {
                that.detectors      = data.detectors ;
                that.detectors_runs = data.runs ;
                that.detectors_display() ;
                $('#runtables-detectors').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                that.detectors_show_all_button.button('enable') ;
                that.detectors_hide_all_button.button('enable') ;
                that.detectors_advanced_button.button('enable') ;
            }
        ) ;
    } ;

    /* ----------------------------------
     *  Application initialization point
     * ----------------------------------
     *
     * RETURN: true if the real initialization took place. Return false
     *         otherwise.
     */
    this.is_initialized = false ;
    this.init = function () {
        if(that.is_initialized) return false ;
        this.is_initialized = true ;
        this.calib_init() ;
        this.detectors_init() ;
        return true ;
    } ;

    /* --------------
     *  Report error
     * --------------
     */
    function report_error (msg) {
        $('#popupdialogs').html(
            '<span class="ui-icon ui-icon-alert" style="float:left;"></span> '+msg
        ) ;
        $('#popupdialogs').dialog({
            resizable: false ,
            modal: true ,
            buttons: {
                Cancel: function () {
                    $(this).dialog('close') ;
                }
            } ,
            title: 'Error Message'
        }) ;
    } ;
}

var runtables = new runtables_create() ;
