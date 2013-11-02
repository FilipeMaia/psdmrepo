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
    this.is_calib_editor = false ;

    /* The context for v-menu items
     */
    var context2_default = {
        'calib'     : '' ,
        'detectors' : '' ,
        'epics'     : ''
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
            case 'calib':     this.calib_update() ;     break ;
            case 'detectors': this.detectors_update() ; break ;
            case 'epic':      this.epics_update() ;     break ;
        }
    } ;

    /* ------------------------------
     *  Menu item: Calibration table
     * ------------------------------
     */
    this.calib_runs = null ;

    this.calib_refresh_button = null ;
    this.calib_reset_button = null ;
    this.calib_from_run = null ;
    this.calib_through_run = null ;

    this.calib_table = null ;

    this.calib_init = function () {

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
                        if (that.is_calib_editor)
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
                        if (that.is_calib_editor)
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
                        if (that.is_calib_editor)
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
                        if (that.is_calib_editor)
                            table_cont.find('textarea.runtables-calib-comment').each(function () {
                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                var calib = that.calib_runs[run] ;
                                elem.text(calib.comment) ;
                            }) ;
                    }
                }
            }
        ] ;
        if (this.is_calib_editor) hdr.push (
            {   name: 'ACTIONS', sorted: false , hideable: true ,
                type: {
                    after_sort: function () {
                        table_cont.find('button.save_run').
                            button().
                            button('disable').
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                var dark = table_cont.find('input.runtables-calib-dark#'+run).attr('checked') ? 1 : 0 ;
                                var flat = table_cont.find('input.runtables-calib-flat#'+run).attr('checked') ? 1 : 0 ;
                                var geom = table_cont.find('input.runtables-calib-geom#'+run).attr('checked') ? 1 : 0 ;
                                var comment = table_cont.find('textarea#'+run).val() ;

                                that.calib_save(run, dark, flat, geom, comment, function () {
                                    table_cont.find('button#'+run+'.save_run').button('disable') ;
                                    table_cont.find('.edit#'+run).parent().parent().parent().css('background-color', '') ;
                                }) ;
                            }
                        ) ;
                        table_cont.find('.edit').change(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            table_cont.find('button#'+run+'.save_run').button('enable') ;
                            $(this).parent().parent().parent().css('background-color', 'rgb(255, 220, 220)') ;
                        }) ;
                    }
                 }
             }
       ) ;
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
        for (var run in this.calib_runs)
            rows.push(this.is_calib_editor ?
                [   {number: run, html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'} ,
                    '<div class="runtables-calib-dark-cont"    ><input    class="runtables-calib-dark edit"    id="'+run+'" type="checkbox" /></div>' ,
                    '<div class="runtables-calib-flat-cont"    ><input    class="runtables-calib-flat edit"    id="'+run+'" type="checkbox" /></div>' ,
                    '<div class="runtables-calib-geom-cont"    ><input    class="runtables-calib-geom edit"    id="'+run+'" type="checkbox" /></div>' ,
                    '<div class="runtables-calib-comment-cont" ><textarea class="runtables-calib-comment edit" id="'+run+'" rows="2" cols="56" ></textarea></div>' ,
                    '<button class="save_run"   id="'+run+'" title="save modifications to the database" >Save</>'] :
                [   {number: run, html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'} ,
                    '<div class="runtables-calib-dark-cont"    ><div      class="runtables-calib-dark"    id="'+run+'" ></div></div>' ,
                    '<div class="runtables-calib-flat-cont"    ><div      class="runtables-calib-flat"    id="'+run+'" ></div></div>' ,
                    '<div class="runtables-calib-geom-cont"    ><div      class="runtables-calib-geom"    id="'+run+'" ></div></div>' ,
                    '<div class="runtables-calib-comment-cont" ><div      class="runtables-calib-comment" id="'+run+'" ></div></div>' ]) ;

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

    this.calib_save = function (run, dark, flat, geom, comment, on_success) {
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
                if (on_success) on_success() ;
                else {
                    that.calib_runs[run] = data.runs[run] ;
                    that.calib_display() ;
                }
                $('#runtables-calib').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;

    /* --------------------------------
     *  Menu item: DAQ Detectors table
     * --------------------------------
     */
    this.detectors_runs = null ;

    this.detectors_refresh_button = null ;
    this.detectors_reset_button = null ;
    this.detectors_from_run = null ;
    this.detectors_through_run = null ;

    this.detectors_table = null ;

    this.detectors_init = function () {

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

    /* -------------------------
     *  Menu item: EPICS tables
     * -------------------------
     */

    this.epics_sections       = null ;
    this.epics_parameters     = null ;
    this.epics_runs           = null ;
    this.epics_refresh_button = null ;
    this.epics_reset_button   = null ;
    this.epics_from_run       = null ;
    this.epics_through_run    = null ;

    this.epics_init = function () {

        this.epics_refresh_button = $('#runtables-epics').find('button[name="refresh"]').button() ;
        this.epics_reset_button   = $('#runtables-epics').find('button[name="reset"]')  .button() ;
        this.epics_from_run       = $('#runtables-epics').find('input[name="from"]') ;
        this.epics_through_run    = $('#runtables-epics').find('input[name="through"]') ;

        this.epics_refresh_button.click (function () { that.epics_update() ;  }) ;
        this.epics_from_run      .change(function () { that.epics_update() ;  }) ;
        this.epics_through_run   .change(function () { that.epics_update() ;  }) ;

        this.epics_reset_button.click (function () {
            that.epics_from_run.val('') ;
            that.epics_through_run.val('') ;
            that.epics_update() ;
        }) ;

        this.epics_update() ;
    } ;

    this.epics_tabs = null ,
    this.epics_tables = {} ,

    this.epics_display = function () {

        if (!this.epics_tabs) {
            this.epics_tabs = $('#runtables-epics').find('.runtables-body') ;
            var html =
'<div id="tabs">' +
'  <ul>' ;
            var html_body = '' ;
            for (var i in this.epics_sections) {
                var section = this.epics_sections[i] ;
                html +=
'    <li><a href="#tab_'+section.name+'">'+section.title+'</a></li>' ;
                html_body +=
'  <div id="tab_'+section.name+'">' +
'    <div class="runtables-body-tab-cont">' +
'      <div id="table-controls" style="margin-bottom:10px;">' +
'        <table><tbody>' +
'          <tr style="font-size:12px;">' +
'            <td valign="center">' +
'              <button id="'+section.name+'" class="control-button" name="show_all" title="show all columns">Show all</button></td>' +
'            <td valign="center">' +
'              <button id="'+section.name+'" class="control-button" name="hide_all" title="hide all columns">Hide all</button></td>' +
'            <td valign="center">' +
'              <button id="'+section.name+'" class="control-button" name="advanced" title="open a dialog to select which columns to show/hide">Select columns</td>' +
'          </tr>' +
'        </tbody></table>' +
'      </div>' +
'      <div id="table"></div>' +
'    </div>' +
'  </div>' ;
            }
            html +=
'  </ul>' +
html_body +
'</div>' ;
            this.epics_tabs.html(html) ;
            this.epics_tabs.tabs() ;

            for (var i in this.epics_sections)
                this.epics_create_table(this.epics_sections[i]) ;

        }
        var title = 'show the run in the e-Log Search panel within the current Portal' ;
        for (var i in this.epics_sections) {

            var section      = this.epics_sections[i] ;
            var section_name = section.name ;
            var table        = this.epics_tables[section_name] ;

            var rows = [] ;
            for (var run in this.epics_runs) {
                var row = [] ;
                row.push(
                    {   number: run ,
                        html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'
                    }
                ) ;
                var param2value = this.epics_runs[run] ;
                for (var i in section.parameters) {
                    var name  = section.parameters[i] ;
                    var value = name in param2value ? param2value[name] : '' ;
                    row.push(value === '' ? '&nbsp;' : value)  ;
                }
                rows.push(row) ;
            }
            table.load(rows) ;
        }
        this.epics_tabs.find('button[name="show_all"]').button('enable') ;
        this.epics_tabs.find('button[name="hide_all"]').button('enable') ;
        this.epics_tabs.find('button[name="advanced"]').button('enable') ;
    } ;

    this.epics_advanced = function (section_name) {
        alert('epics_advanced: '+section_name) ;
    } ;

    this.epics_create_table = function (section)  {

        var section_name = section.name ;

        var tab_body   = this.epics_tabs.find('#tab_'+section_name) ;
        var table_cont = tab_body.find('div#table') ;

        var hdr = [
            {name: 'RUN', type: Table.Types.Number_HTML}
        ] ;
        for (var i in section.parameters) {

            var name = section.parameters[i] ;
//            var html_name = name ;
            var html_name = '' ;
            var name_split = name.split(':') ;
            for (var j in name_split)
                html_name += '<div>'+name_split[j]+'</div>' ;

            hdr.push({
                name:     '<div>'+html_name+'</div>' ,
                hideable: true ,
                align:    'center' ,
                style:    ' white-space: nowrap;'
            }) ;
        }
        var table = new Table (
            table_cont ,
            hdr ,
            null ,
            { default_sort_forward: false } ,
            config.handler('runtables', section_name)
        ) ;
        table.display() ;

        this.epics_tables[section_name] = table ;

        tab_body.find('button[name="show_all"]').button().button('disable').click(function() { that.epics_tables[section_name].display('show_all') ; }) ;
        tab_body.find('button[name="hide_all"]').button().button('disable').click(function() { that.epics_tables[section_name].display('hide_all') ; }) ;
        tab_body.find('button[name="advanced"]').button().button('disable').click(function() { that.epics_advanced(section_name) ; }) ;
    } ;

    this.epics_update = function () {
        $('#runtables-epics').find('.runtables-info#updated').html('Loading...') ;

        web_service_GET (
            '../portal/ws/runtable_epics_get.php' ,
            {   exper_id:    this.exp_id ,
                from_run:    parseInt(this.epics_from_run.val()) ,
                through_run: parseInt(this.epics_through_run.val())
            } ,
            function (data) {
                that.epics_sections   = data.sections ;
                that.epics_parameters = data.parameters ;
                that.epics_runs       = data.runs ;
                that.epics_display() ;
                $('#runtables-epics').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
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
        this.epics_init() ;
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
