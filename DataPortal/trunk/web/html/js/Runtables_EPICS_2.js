/**
 * The application for displaying the run table with EPICS variables
 *
 * @returns {Runtables_EPICS}
 */
function Runtables_EPICS (experiment, access_list) {

    var _that = this ;
    
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
        this._init() ;
    } ;

    this.on_update = function (sec) {
        if (this.active) {
            this._init() ;
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

    this._is_initialized = false ;

    this._wa = null ;
    this._tabs = null ;

    this._tables = {} ;

    this._section_names = null ;
    this._sections = null ;

    this._init = function () {

        if (this._is_initialized) return ;
        this._is_initialized = true ;

        this.container.html('<div id="runtables-epics"></div>') ;
        this._wa = this.container.find('div#runtables-epics') ;

        if (!this.access_list.runtables.read) {
            this._wa.html(this.access_list.no_page_access_html) ;
            return ;
        }

        this._wa.html('Loading...') ;

        Fwk.web_service_GET (
            '../portal/ws/runtable_epics_sections.php' ,
            {exper_id: this.experiment.id} ,
            function (data) {
                _that._section_names = data.section_names ;
                _that._sections = data.sections ;

                var html =
'<div id="tabs">' +
'  <ul>' ;
                for (var i in _that._section_names) {
                    var s_name = _that._section_names[i] ;
                    var s = _that._sections[s_name] ;
                    html +=
'    <li><a href="#'+s_name+'">'+s.title+'</a></li>' ;
                }
                html +=
'  </ul>' ;
                for (var i in _that._section_names) {
                    var s_name = _that._section_names[i] ;
                    html +=
'  <div id="'+s_name+'">' +
'    <div id="ctrl">' +
'      <div class="group">' +
'        <span class="label">From run</span>' +
'        <input class="update-trigger" type="text" name="'+s_name+':first" value="" size=8 />' +
'        <span class="label">through</span>' +
'        <input class="update-trigger" type="text" name="'+s_name+':last"  value="" size=8 />' +
'      </div>' +
'      <div class="buttons">' +
'        <button class="control-button" name="'+s_name+':search" title="search and display results" >Search</button>' +
'        <button class="control-button" name="'+s_name+':reset"  title="reset the form"             >Reset</button>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'    <div id="body">' +
'      <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'      <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'      <div style="clear:both;"></div>' +
'      <div>' +
'        <button class="control-button" name="'+s_name+':show_all" title="show all columns"                  >Show all</button>' +
'        <button class="control-button" name="'+s_name+':hide_all" title="hide all columns"                  >Hide all</button>' +
'        <button class="control-button" name="'+s_name+':advanced" title="select which columns to show/hide" >Select columns</button>' +
'      </div>' +
'      <div id="table" class="table"></div>' +
'    </div>' +
'  </div>' ;
                }
                html +=
'</div>' ;

                _that._wa.html(html) ;
                _that._tabs = _that._wa.children('#tabs').tabs() ;

                _that._tabs.find('.control-button').button().click(function () {
                    var s2 = this.name.split(':') ;
                    var section_name = s2[0] ;
                    var op = s2[1] ;
                    switch (op) {
                        case 'search'   : _that._load (section_name) ; break ;
                        case 'reset'    : _that._reset(section_name) ; break ;
                        case 'show_all' : _that._tables[section_name].display('show_all') ;  break ;
                        case 'hide_all' : _that._tables[section_name].display('hide_all') ;  break ;
                        case 'advanced' : _that._advanced(section_name) ;  break ;
                    }
                }) ;
                _that._tabs.find('.update-trigger').change(function () {
                    var s2 = this.name.split(':') ;
                    var section_name = s2[0] ;
                    _that._load(section_name) ;
                }) ;

                for (var i in _that._section_names) {
                    var s_name = _that._section_names[i] ;
                    _that._tables[s_name] = _that._create_table(s_name) ;
                    _that._load(s_name) ;
                    console.log(s_name) ;
                }
            } ,
            function (msg) {
                _that._wa.html(msg) ;
            }
        ) ;
    } ;
    this._reset = function (section_name) {
        var tab_body = this._tabs.find('div#'+section_name) ;
        tab_body.find('.update-trigger').val('') ;
        this._load (section_name) ;
    } ;

    this._create_table = function (section_name)  {

        var section = this._sections[section_name] ;

        var tab_body   = this._tabs.find('div#'+section_name) ;
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

        this._tables[section_name] = table ;

        return table ;
    } ;

    this._advanced = function (section_name) {

        var detectors_selector = $('#fwk-largedialogs') ;

        var html =
'<div style="overflow:auto;">' +
'  <table><tbody>' ;
        var detectors_per_row = 5 ;
        var num_detectors = 0 ;
        var table = this._tables[section_name] ;
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

    this._display = function (section_name) {

        var table = this._tables[section_name] ;

        var title = 'show the run in the e-Log Search panel within the current Portal' ;

        var section = this._sections[section_name] ;

        var rows = [] ;
        for (var run in section.runs) {
            var row = [] ;
            row.push({
                number: run ,
                html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'
            }) ;
            var param2value = section.runs[run] ;
            for (var i in section.parameters) {
                var name  = section.parameters[i] ;
                var value = name in param2value ? param2value[name] : '' ;
                row.push(value === '' ? '&nbsp;' : value)  ;
            }
            rows.push(row) ;
        }
        table.load(rows) ;
    } ;

    this._load = function (section_name) {

        var updated = this._tabs.find('div#'+section_name).find('#updated') ;
        updated.html('Loading...') ;

        Fwk.web_service_GET (
            '../portal/ws/runtable_epics_section_get.php' ,
            {   exper_id:    this.experiment.id ,
                from_run:    parseInt(this._tabs.find('input[name="'+section_name+':first"]').val()) ,
                through_run: parseInt(this._tabs.find('input[name="'+section_name+':last"]').val()) ,
                section:     section_name
            } ,
            function (data) {
                _that._sections[section_name].runs = data.runs ;
                _that._display(section_name) ;
                updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;
}
define_class (Runtables_EPICS, FwkApplication, {}, {});
