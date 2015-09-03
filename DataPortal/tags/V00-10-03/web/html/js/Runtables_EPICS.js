define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/SmartTable', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, SmartTable, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Runtables_EPICS.css') ;

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

        this._update_interval_sec = 30 ;
        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > this._update_interval_sec) {
                    this._prev_update_sec = now_sec ;
                    this._load_all() ;
                }
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
'      <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'      <div style="clear:both;"></div>' +
'      <div class="group">' +
'        <span class="label">Runs</span>' +
'        <select class="update-trigger" name="'+s_name+':runs" >' +
'          <option value="20" >last 20</option>' +
'          <option value="100">last 100</option>' +
'          <option value="200">last 200</option>' +
'          <option value="range">specific range</option>' +
'          <option value="all">all</option>' +
'        </select>' +
'      </div>' +
'      <div class="group">' +
'        <span class="label">First</span>' +
'        <input class="update-trigger" type="text" name="'+s_name+':first" value="-20" size=8 disabled />' +
'        <span class="label">Last</span>' +
'        <input class="update-trigger" type="text" name="'+s_name+':last"  value="" size=8 disabled />' +
'      </div>' +
'      <div class="buttons">' +
'        <button class="control-button"               name="'+s_name+':reset"  title="reset the form"             >RESET FORM</button>' +
'        <button class="control-button update-button" name="'+s_name+':update" title="update and display results" ><img src="../webfwk/img/Update.png" /></button>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'    <div id="body">' +
'      <div id="table_ctrl">' +
'        <span>Column display mode</span>' +
'        <select class="display-trigger" name="'+s_name+':column_mode">' +
'          <option value="descr"        >description</option>' +
'          <option value="pv"           >name</option>' +
'          <option value="descr_and_pv" >description (name)</option>' +
'        </select>' +
'      </div>' +
'      <div id="table" class="table" ></div>' +
'    </div>' +
'  </div>' ;
                    }
                    html +=
'</div>' ;
                    _that._wa.html(html) ;
                    _that._tabs = _that._wa.children('#tabs').tabs() ;

                    _that._tabs.find('.control-button').button().click(function () {
                        var s2 = this.name.split(':') ;
                        var s_name = s2[0] ;
                        var op = s2[1] ;
                        switch (op) {
                            case 'update'   : _that._load (s_name) ; break ;
                            case 'reset'    : _that._reset(s_name) ; break ;
                        }
                    }) ;
                    _that._tabs.find('select.update-trigger').change(function () {
                        var s2 = this.name.split(':') ;
                        var s_name = s2[0] ;
                        var range = $(this).val() ;
                        if (range === 'range') {
                            _that._tabs.find('input.update-trigger[name="'+s_name+':first"]').removeAttr('disabled') ;
                            _that._tabs.find('input.update-trigger[name="'+s_name+':last"]') .removeAttr('disabled') ;
                        } else {
                            _that._tabs.find('input.update-trigger[name="'+s_name+':first"]').attr('disabled', 'disabled') ;
                            _that._tabs.find('input.update-trigger[name="'+s_name+':last"]') .attr('disabled', 'disabled') ;
                            switch (range) {
                                case '20' :
                                case '100' :
                                case '200' :
                                    _that._tabs.find('input[name="'+s_name+':first"]').val(-parseInt(range)) ;
                                    _that._tabs.find('input[name="'+s_name+':last"]').val('') ;
                                    break ;
                                case 'all' :
                                    _that._tabs.find('input[name="'+s_name+':first"]').val('') ;
                                    _that._tabs.find('input[name="'+s_name+':last"]').val('') ;
                                    break ;
                            }
                        }
                    }) ;
                    _that._tabs.find('.update-trigger').change(function () {
                        var s2 = this.name.split(':') ;
                        var s_name = s2[0] ;
                        _that._load(s_name) ;
                    }) ;
                    _that._tabs.find('select.display-trigger').change(function () {
                        var s2 = this.name.split(':') ;
                        var s_name = s2[0] ;
                        console.log('tab: '+s_name+', display: '+$(this).val()) ;
                        _that._tables[s_name] = _that._create_table(s_name) ;
                        _that._display(s_name) ;
                    }) ;

                    for (var i in _that._section_names) {
                        var s_name = _that._section_names[i] ;
                        _that._tables[s_name] = _that._create_table(s_name) ;
                        _that._load(s_name) ;
                    }
                } ,
                function (msg) {
                    _that._wa.html(msg) ;
                }
            ) ;
        } ;
        this._reset = function (s_name) {
            var tab_body = this._tabs.find('div#'+s_name) ;
            tab_body.find('.update-trigger').val('') ;
            this._load (s_name) ;
        } ;

        this._create_table = function (s_name)  {

            var section = this._sections[s_name] ;

            var tab_body = this._tabs.find('div#'+s_name) ;

            var column_mode = this._tabs.find('select.display-trigger[name="'+s_name+':column_mode"]').val() ;

            var hdr = [
                'RUN'
            ] ;
            for (var i in section.parameters) {
                var p = section.parameters[i] ;
                switch (column_mode) {
                    case 'descr'        : hdr.push(p.descr) ; break ;
                    case 'pv'           : hdr.push(p.name) ; break ;
                    case 'descr_and_pv' : hdr.push(p.descr + '&nbsp; &nbsp;('+p.name+')') ; break ;
                }
            }
            var rows = null ;
            var num_hdr_rows = 2 ;
            var max_hdr_rows = 5 ;
            var table = new SmartTable (
                hdr ,
                rows ,
                num_hdr_rows ,
                max_hdr_rows
            ) ;
            table.display(tab_body.find('div#table')) ;
            this._tables[s_name] = table ;

            return table ;
        } ;

        this._display = function (s_name) {

            var table = this._tables[s_name] ;

            var title = 'show the run in the e-Log Search panel within the current Portal' ;

            var section = this._sections[s_name] ;

            var rows = [] ;
            for (var run in section.runs) {
                var row = [] ;
                row.push(
                    '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="link">'+run+'</a>'
                ) ;
                var param2value = section.runs[run] ;
                for (var i in section.parameters) {
                    var p = section.parameters[i] ;
                    var value = p.name in param2value ? param2value[p.name] : '' ;
                    row.push(value)  ;
                }
                rows.push(row) ;
            }
            rows.reverse() ;
            table.load(rows) ;
        } ;

        this._load_all = function () {
            if (this._section_names)
                for (var i in this._section_names) {
                    var s_name = _that._section_names[i] ;
                    this._load(s_name) ;
                }
        } ;
        this._load = function (s_name) {

            var updated = this._tabs.find('div#'+s_name).find('#updated') ;
            updated.html('Loading...') ;

            var params = {
                exper_id: this.experiment.id ,
                section:  s_name
            } ;
            var range = this._tabs.find('select[name="'+s_name+':runs"]').val() ;
            switch (range) {
                case '20' :
                case '100' :
                case '200' :
                    params.from_run    = -parseInt(range) ;
                    params.through_run = 0 ;
                    break ;
                case 'range' :
                    params.from_run    = parseInt(this._tabs.find('input[name="'+s_name+':first"]').val()) ;
                    params.through_run = parseInt(this._tabs.find('input[name="'+s_name+':last"]').val()) ;
                    break ;
                case 'all' :
                default :
                    params.from_run    = 0 ;
                    params.through_run = 0 ;
                    break ;
            }

            Fwk.web_service_GET (
                '../portal/ws/runtable_epics_section_get.php' ,
                params ,
                function (data) {
                    _that._sections[s_name].runs = data.runs ;
                    _that._display(s_name) ;
                    updated.html('Updated: <b>'+data.updated+'</b>') ;
                }
            ) ;
        } ;
    }
    Class.define_class (Runtables_EPICS, FwkApplication, {}, {}) ;

    return Runtables_EPICS ;
}) ;
