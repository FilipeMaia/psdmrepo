/**
 * The application for displaying the run table with EPICS variables
 *
 * @returns {Runtables_EPICS}
 */
function Runtables_EPICS (experiment, access_list) {

    var that = this ;
    
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    FwkApplication.call(this) ;

    // ------------------------------------------------
    // Override event handler defined in the base class
    // ------------------------------------------------

    this.on_activate = function() {
        this.on_update() ;
    } ;

    this.on_deactivate = function() {
        this.init() ;
    } ;

    this.on_update = function (sec) {
        if (this.active) {
            this.init() ;
        }
    } ;

    // -----------------------------
    // Parameters of the application
    // -----------------------------

    this.experiment  = experiment ;
    this.access_list = access_list ;

    // --------------------
    // Own data and methods
    // --------------------

    this.is_initialized = false ;

    this.wa = null ;

    this.sections       = null ;
    this.parameters     = null ;
    this.runs           = null ;
    this.refresh_button = null ;
    this.reset_button   = null ;
    this.from_run       = null ;
    this.through_run    = null ;

    this.init = function () {

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        this.container.html('<div id="runtables-epics"></div>') ;
        this.wa = this.container.find('div#runtables-epics') ;

        if (!this.access_list.runtables.read) {
            this.wa.html(this.access_list.no_page_access_html) ;
            return ;
        }
        var html =
'<div class="runtables-epics-ctrl">' +
'  <table><tbody>' +
'    <tr>' +
'      <td valign="center">' +
'        <span style="font-weight:bold;">Select runs from</span></td>' +
'      <td valign="center">' +
'        <input type="text" name="from" size="2" title="The first run of the interval. If no input is provided then the very first known run will be assumed." /></td>' +
'      <td valign="center">' +
'        <span style="font-weight:bold; margin-left:0px;">through</span></td>' +
'      <td valign="center">' +
'        <input name="through" type="text" size="2" title="The last run of the interval. If no input is provided then the very last known run will be assumed"/ ></td>' +
'      <td valign="center">' +
'        <button class="control-button" style="margin-left:20px;" name="reset" title="reset the form">Reset Form</button></td>' +
'      <td valign="center">' +
'        <button class="control-button" name="refresh" title="check if there were any updates on this page">Refresh</button></td>' +
'    </tr>' +
'  </tbody></table>' +
'</div>' +
'<div class="runtables-epics-wa">' +
'  <div class="runtables-epics-info" id="info"    style="float:left;" >&nbsp;</div>' +
'  <div class="runtables-epics-info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div> ' +
'  <div class="runtables-epics-body"></div>' +
'</div>' ;
        this.wa.html(html) ;

        this.refresh_button = this.wa.find('button[name="refresh"]').button() ;
        this.reset_button   = this.wa.find('button[name="reset"]')  .button() ;
        this.from_run       = this.wa.find('input[name="from"]') ;
        this.through_run    = this.wa.find('input[name="through"]') ;

        this.refresh_button.click (function () { that.load() ;  }) ;
        this.from_run      .change(function () { that.load() ;  }) ;
        this.through_run   .change(function () { that.load() ;  }) ;

        this.reset_button.click (function () {
            that.from_run.val('') ;
            that.through_run.val('') ;
            that.load() ;
        }) ;

        this.load() ;
    } ;

    this.tabs = null ;
    this.tables = {} ;

    this.display = function () {

        if (!this.tabs) {
            this.tabs = this.wa.find('.runtables-epics-body') ;
            var html =
'<div id="tabs">' +
'  <ul>' ;
            var html_body = '' ;
            for (var i in this.sections) {
                var section = this.sections[i] ;
                html +=
'    <li><a href="#tab_'+section.name+'">'+section.title+'</a></li>' ;
                html_body +=
'  <div id="tab_'+section.name+'">' +
'    <div class="runtables-epics-body-tab-cont">' +
'      <div id="table-controls" style="margin-bottom:10px;">' +
'        <table><tbody>' +
'          <tr>' +
'            <td valign="center"><button id="'+section.name+'" class="control-button" name="show_all" title="show all columns">Show all</button></td>' +
'            <td valign="center"><button id="'+section.name+'" class="control-button" name="hide_all" title="hide all columns">Hide all</button></td>' +
'            <td valign="center"><button id="'+section.name+'" class="control-button" name="advanced" title="open a dialog to select which columns to show/hide">Select columns</td>' +
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
            this.tabs.html(html) ;
            this.tabs.tabs() ;

            for (var i in this.sections)
                this.create_table(this.sections[i]) ;

        }
        var title = 'show the run in the e-Log Search panel within the current Portal' ;
        for (var i in this.sections) {

            var section      = this.sections[i] ;
            var section_name = section.name ;
            var table        = this.tables[section_name] ;

            var rows = [] ;
            for (var run in this.runs) {
                var row = [] ;
                row.push(
                    {   number: run ,
                        html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'
                    }
                ) ;
                var param2value = this.runs[run] ;
                for (var i in section.parameters) {
                    var name  = section.parameters[i] ;
                    var value = name in param2value ? param2value[name] : '' ;
                    row.push(value === '' ? '&nbsp;' : value)  ;
                }
                rows.push(row) ;
            }
            table.load(rows) ;
        }
        this.tabs.find('button[name="show_all"]').button('enable') ;
        this.tabs.find('button[name="hide_all"]').button('enable') ;
        this.tabs.find('button[name="advanced"]').button('enable') ;
    } ;

    this.advanced = function (section_name) {

        var detectors_selector = $('#fwk-largedialogs') ;

        var html =
'<div style="overflow:auto;">' +
'  <table><tbody>' ;
        var detectors_per_row = 5 ;
        var num_detectors = 0 ;
        var table = this.tables[section_name] ;
        var header_info = table.header_info() ;
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
            table.display(this.checked ? 'show' : 'hide', parseInt(col_number)) ;
        }) ;
    } ;

    this.create_table = function (section)  {

        var section_name = section.name ;

        var tab_body   = this.tabs.find('#tab_'+section_name) ;
        var table_cont = tab_body.find('div#table') ;

        var hdr = [
            {name: 'RUN', type: Table.Types.Number_HTML}
        ] ;
        for (var i in section.parameters) {

            var name = section.parameters[i] ;
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
            Fwk.config_handler('runtables', section_name)
        ) ;
        table.display() ;

        this.tables[section_name] = table ;

        tab_body.find('button[name="show_all"]').button().button('disable').click(function() { that.tables[section_name].display('show_all') ; }) ;
        tab_body.find('button[name="hide_all"]').button().button('disable').click(function() { that.tables[section_name].display('hide_all') ; }) ;
        tab_body.find('button[name="advanced"]').button().button('disable').click(function() { that.advanced(section_name) ; }) ;
    } ;

    this.load = function () {
        this.wa.find('.runtables-epics-info#updated').html('Loading...') ;
        Fwk.web_service_GET (
            '../portal/ws/runtable_epics_get.php' ,
            {   exper_id:    this.experiment.id ,
                from_run:    parseInt(this.from_run.val()) ,
                through_run: parseInt(this.through_run.val())
            } ,
            function (data) {
                that.sections   = data.sections ;
                that.parameters = data.parameters ;
                that.runs       = data.runs ;
                that.display() ;
                that.wa.find('.runtables-epics-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;
}
define_class (Runtables_EPICS, FwkApplication, {}, {});
