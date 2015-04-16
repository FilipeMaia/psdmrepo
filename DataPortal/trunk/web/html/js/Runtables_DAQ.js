define ([
    'webfwk/CSSLoader', 'webfwk/SmartTable' ,
    'webfwk/Class',     'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader, SmartTable ,
    Class,     FwkApplication, Fwk) {

    cssloader.load('../portal/css/Runtables_DAQ.css') ;

    /**
     * The application for displaying and managing DAQ tables.
     *
     * @returns {Runtables_DAQ}
     */
    function Runtables_DAQ (experiment, access_list) {

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

        this._sections = [
            {id: 'detectors', title: 'Detectors'} ,
            {id: 'totals',    title: 'Detector Totals'}
        ] ;
        this._section_data = {
            'detectors' : { parameters: [], runs: []} ,
            'totals'    : { parameters: [], runs: []}
        } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ;
                if (!this_html) {
                    this_html =
'<div id="runtables-daq" > ' +
  '<div id="tabs"> ' +

    '<ul> ' ;
                    for (var i in this._sections) {
                        var s = this._sections[i] ;
                        this_html +=
      '<li><a href="#'+s.id+'">'+s.title+'</a></li> ' ;
                    }
                    this_html +=
    '</ul> ' ;
                    for (var i in this._sections) {
                        var s = this._sections[i] ;
                        this_html +=
'  <div id="'+s.id+'" class="panel" >'  +
'    <div id="ctrl">' +
'      <div class="info" id="updated" style="float:right;" >Loading...</div> ' +
'      <div style="clear:both;"></div> ' +
'      <div class="group"> ' +
'        <span class="label">Runs</span> ' +
'        <select class="update-trigger" name="'+s.id+':runs" > ' +
'          <option value="20" >last 20</option> ' +
'          <option value="100">last 100</option> ' +
'          <option value="200">last 200</option> ' +
'          <option value="range">specific range</option> ' +
'          <option value="all">all</option> ' +
'        </select> ' +
'      </div> ' +
'      <div class="group"> ' +
'        <span class="label" >First</span> ' +
'        <input class="update-trigger" type="text" name="'+s.id+':first" value="-20" size=8 disabled /> ' +
'        <span class="label" >Last</span> ' +
'        <input class="update-trigger" type="text" name="'+s.id+':last"  value="" size=8 disabled /> ' +
'      </div> ' +
'      <div class="buttons" > ' +
'        <button class="control-button"               name="'+s.id+':reset"  title="reset the form"             >RESET FORM</button> ' +
'        <button class="control-button update-button" name="'+s.id+':update" title="update and display results" ><img src="../webfwk/img/Update.png" /></button> ' +
'      </div> ' +
'      <div style="clear:both;" ></div> ' +
'    </div> ' +
'    <div id="body" > ' +
'      <div id="table_ctrl" > ' +
'        <span>Column display mode</span> ' +
'        <select class="display-trigger" name="'+s.id+':column_mode" disabled > ' +
'          <option value="descr"          >description</option> ' +
'          <option value="name"           >name</option> ' +
'          <option value="descr_and_name" >description (name)</option> ' +
'        </select> ' +
'      </div> ' +
'      <div id="table" class="table" ></div> ' +
'    </div> ' +
'  </div> ' ;
                    }
                    this_html +=
  '</div> ' +
'</div>' ;
                }
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('div#runtables-daq') ;
            }
            return this._wa_elem ;
        } ;
        this._tabs = function () {
            if (!this._tabs_elem) {
                this._tabs_elem = this._wa().children('#tabs') ;
                this._tabs_elem.tabs() ;
            }
            return this._tabs_elem ;
        } ;
        this._panel = function (s_id) {
            if (!this._panel_elem)       this._panel_elem       = {} ;
            if (!this._panel_elem[s_id]) this._panel_elem[s_id] = this._tabs().children('.panel#'+s_id) ;
            return this._panel_elem[s_id] ;
        } ;
        this._ctrl = function (s_id) {
            if (!this._ctrl_elem)       this._ctrl_elem       = {} ;
            if (!this._ctrl_elem[s_id]) this._ctrl_elem[s_id] = this._panel(s_id).children('#ctrl') ;
            return this._ctrl_elem[s_id] ;
        } ;
        this._set_updated = function (s_id, html) {
            if (!this._updated_elem)       this._updated_elem       = {} ;
            if (!this._updated_elem[s_id]) this._updated_elem[s_id] = this._ctrl(s_id).find('#updated') ;
            this._updated_elem[s_id].html(html) ;
        } ;
        this._body = function (s_id) {
            if (!this._body_elem)       this._body_elem       = {} ;
            if (!this._body_elem[s_id]) this._body_elem[s_id] = this._panel(s_id).children('#body') ;
            return this._body_elem[s_id] ;
        } ;
        this._table_ctrl = function (s_id) {
            if (!this._table_ctrl_elem)       this._table_ctrl_elem       = {} ;
            if (!this._table_ctrl_elem[s_id]) this._table_ctrl_elem[s_id] = this._body(s_id).children('#table_ctrl') ;
            return this._table_ctrl_elem[s_id] ;
        } ;
        this._column_mode = function (s_id) {
            if (!this._column_mode_elem)       this._column_mode_elem       = {} ;
            if (!this._column_mode_elem[s_id]) this._column_mode_elem[s_id] = this._table_ctrl(s_id).find('select.display-trigger[name="'+s_id+':column_mode"]') ;
            return this._column_mode_elem[s_id] ;
        } ;
        this._table = function (s_id) {
            if (!this._table_elem)       this._table_elem       = {} ;
            if (!this._table_elem[s_id]) this._table_elem[s_id] = this._body(s_id).find('#table') ;
            return this._table_elem[s_id] ;
        } ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this.access_list.runtables.read) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }
            this._tabs().find('.control-button').button().click(function () {
                var s2   = this.name.split(':') ;
                var s_id = s2[0] ;
                var op   = s2[1] ;
                switch (op) {
                    case 'update'   : _that._load (s_id) ; break ;
                    case 'reset'    : _that._reset(s_id) ; break ;
                }
            }) ;
            this._tabs().find('select.update-trigger').change(function () {
                var s2    = this.name.split(':') ;
                var s_id  = s2[0] ;
                var range = $(this).val() ;
                if (range === 'range') {
                    _that._tabs().find('input.update-trigger[name="'+s_id+':first"]').removeAttr('disabled') ;
                    _that._tabs().find('input.update-trigger[name="'+s_id+':last"]') .removeAttr('disabled') ;
                } else {
                    _that._tabs().find('input.update-trigger[name="'+s_id+':first"]').attr('disabled', 'disabled') ;
                    _that._tabs().find('input.update-trigger[name="'+s_id+':last"]') .attr('disabled', 'disabled') ;
                    switch (range) {
                        case  '20' :
                        case '100' :
                        case '200' :
                            _that._tabs().find('input[name="'+s_id+':first"]').val(-parseInt(range)) ;
                            _that._tabs().find('input[name="'+s_id+':last"]') .val('') ;
                            break ;
                        case 'all' :
                            _that._tabs().find('input[name="'+s_id+':first"]').val('') ;
                            _that._tabs().find('input[name="'+s_id+':last"]') .val('') ;
                            break ;
                    }
                }
            }) ;
            this._tabs().find('.update-trigger').change(function () {
                var s2   = this.name.split(':') ;
                var s_id = s2[0] ;
                _that._load(s_id) ;
            }) ;
            this._tabs().find('select.display-trigger').change(function () {
                var s2   = this.name.split(':') ;
                var s_id = s2[0] ;
                _that._display(s_id) ;
            }) ;
            this._load_all() ;
        } ;
        this._reset = function (s_id) {
            this._tabs(s_id).find('.update-trigger').val('') ;
            this._load (s_id) ;
        } ;
        this._load_all = function () {
            for (var i in this._sections) {
                var s = this._sections[i] ;
                this._load(s.id) ;
            }
        } ;
        this._load = function (s_id) {

            this._set_updated(s_id, 'Loading...') ;

            var params = {
                exper_id: this.experiment.id ,
                section:  s_id
            } ;
            var range = this._tabs().find('select[name="'+s_id+':runs"]').val() ;
            switch (range) {
                case '20' :
                case '100' :
                case '200' :
                    params.from_run    = -parseInt(range) ;
                    params.through_run = 0 ;
                    break ;
                case 'range' :
                    params.from_run    = parseInt(this._tabs().find('input[name="'+s_id+':first"]').val()) ;
                    params.through_run = parseInt(this._tabs().find('input[name="'+s_id+':last"]').val()) ;
                    break ;
                case 'all' :
                default :
                    params.from_run    = 0 ;
                    params.through_run = 0 ;
                    break ;
            }

            Fwk.web_service_GET (
                '../portal/ws/runtable_daq_get.php' ,
                params ,
                function (data) {
                    _that._set_updated(s_id, 'Updated: <b>'+data.updated+'</b>') ;
                    _that._section_data[s_id].runs         = data.runs ;
                    _that._section_data[s_id].names        = data.names ;
                    _that._section_data[s_id].descriptions = data.descriptions ;
                    _that._display(s_id) ;
                    _that._column_mode(s_id).removeAttr('disabled') ;
                }
            ) ;
        } ;
        function _value_display_trait_for (s_id) {
            return s_id === 'detectors' ?
                function (val) { return val ? '<div style="width:100%; text-align:center; font-size:14px; color:red;">&diams;</div>' : '' ; } :
                function (val) { return val ? val : '' ; } ;
        }
        this._display = function (s_id) {

            var value_display = _value_display_trait_for(s_id) ;

            var s_data = this._section_data[s_id] ;

            var hdr = [
                'RUN'
            ] ;
            for (var i in s_data.names) {
                var name = s_data.names[i] ;
                switch (this._column_mode(s_id).val()) {
                    case 'descr'          : hdr.push(s_data.descriptions[name]) ; break ;
                    case 'name'           : hdr.push(name) ; break ;
                    case 'descr_and_name' : hdr.push(s_data.descriptions[name] + '&nbsp; &nbsp; ('+name+')') ; break ;
                }
            }

            var title = 'show the run in the e-Log Search panel within the current Portal' ;
            var rows  = [] ;
            for (var run in s_data.runs) {
                var row = [] ;
                row.push(
                    '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="link" >'+run+'</a>'
                ) ;
                var name2value4run = s_data.runs[run] ;
                for (var i in s_data.names) {
                    var name  = s_data.names[i] ;
                    var value = name in name2value4run ? name2value4run[name] : '' ;
                    row.push(value_display(value))  ;
                }
                rows.push(row) ;
            }
            rows.reverse() ;

            var num_hdr_rows = 2 ;
            var max_hdr_rows = 5 ;

            var table = new SmartTable (
                hdr ,
                rows ,
                num_hdr_rows ,
                max_hdr_rows
            ) ;
            table.display(this._table(s_id)) ;
        } ;
    }
    Class.define_class (Runtables_DAQ, FwkApplication, {}, {}) ;

    return Runtables_DAQ ;
}) ;
