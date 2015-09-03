define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class' ,
    'webfwk/FwkApplication' ,
    'webfwk/Fwk' ,
    'webfwk/SimpleTable' ,
    'webfwk/SelectOption' ,
    'webfwk/StackOfRows' ,
    'regdb/DRPMgr_Defs'] ,

function (
    cssloader ,
    Class ,
    FwkApplication ,
    Fwk ,
    SimpleTable ,
    SelectOption ,
    StackOfRows ,
    DRPMgr_Defs) {

    cssloader.load('../regdb/css/DRPMgr_Instrument.css') ;

    var _DOCUMENT = {
        filesystem:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Narrow a search down to the specified file system.') ,
        year:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Narrow a search down to the specified year when the last \n' +
                'of an experiment began.') ,
        purge:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Begin purging files in this context. The operation can be \n' +
                'stopped if needed by pressing the STOP button.') ,
        review:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Review which data can be purged in this context. Note, this \n' +
                'operation will not trigger any purge.') ,
        reset:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Reset this form back to the default state which \n' +
                'would include all known experiments. \n' +
                'ATTENTION: this can be a lengthy operation!') ,
        stop:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Stop the on-going purge. The operation can be resumed later.') ,
        update:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Click this button to make a new search and update \n' +
                'the table with the present search criteria. \n' +
                'ATTENTION: this can be a lengthy operation!') ,
        experiment_info:
            DRPMgr_Defs.DOCUMENT_METHOD (
                'Open a new tab/window with the Data Manager application \n' +
                'of this experiment.')
    } ;

    var _TABLE_HDR = [{
        name: '', sorted: false, align: 'right'}, {
        name: 'SHORT-TERM',  coldef: [{
            name: 'files',   sorted: false}, {
            name: 'GB',      sorted: false}, {
            name: 'ACTIONS', sorted: false , type: {
                to_string:
                    function (a) {
                        var html = '' ;
                        var storage_class = 'SHORT-TERM' ;
                        if (a[storage_class].review_allowed) html +=
                            SimpleTable.html.Button ('REVIEW', {
                                id:      'review' ,
                                name:    storage_class+':'+a.category ,
                                classes: 'control-button' ,
                                extra:   _DOCUMENT.review}) ;
                        if (a[storage_class].purge_allowed) html +=
                            SimpleTable.html.Button ('PURGE', {
                                id:      'purge' ,
                                name:    storage_class+':'+a.category ,
                                classes: 'control-button control-button-important' ,
                                extra:   _DOCUMENT.purge}) ;
                        return html ;
                    }}}]} , {
        name: 'MEDIUM-TERM', coldef: [{
            name: 'files',   sorted: false}, {
            name: 'GB',      sorted: false}, {
            name: 'ACTIONS', sorted: false, type: {
                to_string:
                    function (a) {
                        var html = '' ;
                        var storage_class = 'MEDIUM-TERM' ;
                        if (a[storage_class].review_allowed) html +=
                            SimpleTable.html.Button ('REVIEW', {
                                id:      'review' ,
                                name:    storage_class+':'+a.category ,
                                classes: 'control-button' ,
                                extra:   _DOCUMENT.review}) ;
                        if (a[storage_class].purge_allowed) html +=
                            SimpleTable.html.Button ('PURGE', {
                                id:      'purge' ,
                                name:    storage_class+':'+a.category ,
                                classes: 'control-button control-button-important' ,
                                extra:   _DOCUMENT.purge}) ;
                        return html ;
                    }}}]} , {
        name: 'HPSS',      coldef: [{
            name: 'files', sorted: false}, {
            name: 'GB',    sorted: false}]
    }] ;

    function _DEFAULT_TABLE_DATA () {
        var data = [
            {   category: 'total' ,
                title:    'TOTAL' ,
                'SHORT-TERM': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 0 ,
                    purge_allowed:  0} ,
                'MEDIUM-TERM': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 0 ,
                    purge_allowed:  0} ,
                'HPSS': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 0 ,
                    purge_allowed:  0}
            } , {
                category: 'expired' ,
                title:    'Expired (by Policy)' ,
                'SHORT-TERM': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 1 ,
                    purge_allowed:  1} ,
                'MEDIUM-TERM': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 1 ,
                    purge_allowed:  1} ,
                'HPSS': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 0 ,
                    purge_allowed:  0}
            }
        ] ;
        for (var m = 24; m > 0; m--) {
            data.push ({
                category: m ,
                title:    m+' m' ,
                'SHORT-TERM': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 1 ,
                    purge_allowed:  1} ,
                'MEDIUM-TERM': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 1 ,
                    purge_allowed:  1} ,
                'HPSS': {
                    'xtc':  {num_files: 0, size_gb: 0} ,
                    'hdf5': {num_files: 0, size_gb: 0} ,
                    review_allowed: 0 ,
                    purge_allowed:  0}
            }) ;
        }
        return data ;
    }

    function ExperimentBody (parent, year, experiment) {
        
        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        StackOfRows.StackRowBody.call(this) ;

        // ------------------------
        // Parameters of the object
        // ------------------------

        this._parent     = parent ;         // may need to notify the parent when 'PURGE' is over
        this._year       = year ;
        this._experiment = experiment ;

        // ----------------------
        // Local data and methods
        // ----------------------

        this._tabs = function () {
            if (!this._tabs_elem) {
                this._tabs_elem = this.container.children('.experiment').children('#tabs').tabs() ;
            }
            return this._tabs_elem ;
        } ;

        /**
         * Find the specified (by its identifiewr) table panel or return null.
         * Note that the panel (if found) will be cached. And some panels
         * can also be removed from teh tabs and from the cache by a separate
         * function: _close_tab_panel()
         * 
         * @see this._close_tab_panel()
         * 
         * @param {String} id
         * @returns {JQueryObject}
         */
        this._tab_panel = function (id) {
            if (!this._tab_panels_elem) { this._tab_panels_elem = {} ; }
            if (!this._tab_panels_elem[id]) {
                var elem = this._tabs().children('#'+id) ;
                if (!elem.length) return null ;
                this._tab_panels_elem[id] = elem ;
            }
            return this._tab_panels_elem[id] ;
        } ;

        /**
         * Close the specified panel and remove it from the cache
         *
         * @see this._close_tab_panel()
         *
         * @param {String} panelId
         * @returns {undefined}
         */
        this._close_tab_panel = function (panelId) {
            this._tabs().find('a[href="#'+panelId+'"]').closest('li').remove() ;
            this._tabs().children('#'+panelId).remove() ;
            this._tabs().tabs('refresh') ;
            // Don't forget to delete the panel from the cache. Otherwise
            // the panel selector would get confused.
            delete this._tab_panels_elem[panelId] ;
        } ;

        /**
         * Activate atab panel by its by index. The index for N panels
         * can be both positive:
         *   0,1,..,N-1
         * or negative (counting from the last panel):
         *   -1,-2,..,(N-2),(N-1)
         *
         * @param {Number} idx
         * @returns {undefined}
         */
        this._select_tab_panel = function (idx) {
            if (idx != this._tabs().tabs('option', 'active')) {
                this._tabs().tabs('option', 'active', idx) ;
            }
        } ;

        /**
         * Focus at the specified tab panel
         *
         * @param {String} panelId
         * @returns {undefined}
         */
        this._focus_at_panel = function (panelId) {
            var focus_at = this._tab_panel(panelId) ;
            var focus_relative_to = $('#fwk-center') ;
            var offset_top = focus_relative_to.scrollTop() + focus_at.position().top - 24;
            focus_relative_to.animate({scrollTop: offset_top}, 'slow') ;
            focus_at.focus() ;
        } ;

        this._table = function () {
            if (!this._table_obj) {
                this._table_obj = new SimpleTable.constructor (
                    this._tab_panel('all').children('.tab-cont').children('.table') ,
                    _TABLE_HDR ,
                    [] ,
                    {   text_when_empty: null ,
                        common_after_sort: function () {
                            _that._table().get_container().find('.control-button').button().click(function () {
                                var op = this.id ;
                                var context = this.name ;
                                console.log('DRPMgr_Instrument.ExperimentBody._table() '+op+' "'+context+'"') ;
                                var storage_category = context.split(':') ;
                                switch (op) {
                                    case 'review':
                                        var storage  = storage_category[0] ;
                                        var category = storage_category[1] ;
                                        _that._review_category(storage, category) ;
                                        break ;
                                }
                            }) ;
                        }
                    }
                ) ;
                this._table_obj.display() ;
            }
            return this._table_obj ;
        } ;
        this._update_table = function () {
            if (!this._is_rendered) return ;

            var data = this._experiment_data || _DEFAULT_TABLE_DATA() ;
            var rows = [] ;
            for (var i in data) {
                var t = data[i] ;

                var short_num_files  = t['SHORT-TERM']['xtc'].num_files + t['SHORT-TERM']['hdf5'].num_files ,
                    short_size_gb    = t['SHORT-TERM']['xtc'].size_gb   + t['SHORT-TERM']['hdf5'].size_gb ;

                var medium_num_files = t['MEDIUM-TERM']['xtc'].num_files + t['MEDIUM-TERM']['hdf5'].num_files ,
                    medium_size_gb   = t['MEDIUM-TERM']['xtc'].size_gb   + t['MEDIUM-TERM']['hdf5'].size_gb ;

                var hpss_num_files   = t['HPSS']['xtc'].num_files + t['HPSS']['hdf5'].num_files ,
                    hpss_size_gb     = t['HPSS']['xtc'].size_gb   + t['HPSS']['hdf5'].size_gb ;

                rows.push([
                    t.title ,
                    short_num_files  ? short_num_files  : '' ,
                    short_size_gb    ? short_size_gb    : '' ,
                    t ,
                    medium_num_files ? medium_num_files : '' ,
                    medium_size_gb   ? medium_size_gb   : '' ,
                    t ,
                    hpss_num_files   ? hpss_num_files   : '' ,
                    hpss_size_gb     ? hpss_size_gb     : ''
                ]) ;
            }
            this._table().load(rows) ;
            
            // Prevent any operations with files until teh data
            // are fully loaded.
            if (!this._experiment_data)
                _that._table().get_container().find('.control-button').button('disable') ;
        } ;

        this._experiment_data = null ;      // lazy loaded on teh first open or forced externally
        this._last_updated    = null ;
        this._is_loading      = false ;

        this.ready2load = function () {
            return _.isNull(this._experiment_data) && !this._is_loading ;
        } ;
        this.load = function () {
            if (!this.ready2load()) return ;

            this._is_loading = true ;

            Fwk.web_service_GET (
                '../regdb/ws/drp_experiment_get.php' ,
                { exper_id: this._experiment.id} ,
                function (data) {

                    _that._experiment_data = data.experiment_data ;
                    _that._last_updated    = data.updated ;

                    // Update the table when done.
                    _that._update_table() ;

                    // Notify parent when done
                    _that._parent.finished_loading (
                        _that._year ,
                        _that._experiment ,
                        _that._experiment_data ,
                        _that._last_updated) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._is_loading = true ;
                }
            ) ;
        } ;
        this._review_category = function (storage, category) {

            var panelId = storage+'-'+category ;

            // Check if the panel is already open. Activate it if found.
            if (this._tab_panel(panelId)) {
                this._tabs().children('div').each(function (i) {
                    if (this.id == panelId) {
                        _that._select_tab_panel(i) ;
                        _that._focus_at_panel(panelId) ;
                    }
                }) ;
                return ;
            }

            // Add a new panel to the tab
            this._tabs().children('.ui-tabs-nav').append (
'<li><a href="#'+panelId+'" style="color:red;">'+storage+':'+category+'</a> <span class="ui-icon ui-icon-close remove_review_'+panelId+'" >Remove Tab</span></li> '
            );
            var html =
'<div id="'+panelId+'"> ' +
  '<div class="tab-cont" > ' +
    'Loading...' +
  '</div>' +
'</div>' ;
            this._tabs().append(html) ;
            this._tabs().tabs('refresh') ;
            this._select_tab_panel(-1) ;
            this._focus_at_panel(panelId) ;

            // Allow closing the tab panel by clicking on the cross on
            // the panel's header.
            //
            // IMPLEMENTATION NOTE:
            //   Note that we need to turn the element into a simple HTML
            //   element then turn it back into a JQuery object to remove
            //   the search context stored witin the original JQuery object.
            //   Otherwise the binding won't work, and it will results in
            //   the run-time complain.
            $(this._tabs().find('span.ui-icon-close.remove_review_'+panelId).get(0)).live('click', function () {
                var panelId = $(this).closest('li').attr('aria-controls') ;
                _that._close_tab_panel(panelId) ;
            });

            // Initiate loading detail information about which data will
            // be removed in this contex.
            ;
        } ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this._is_rendered = false ;

        this.render = function () {

            if (!this._is_rendered) {
                this._is_rendered = true ;

                var experiment_link = '../portal?exper_id='+this._experiment.id ;
                var html =
'<div class="experiment" > ' +
  '<div class="experiment_info" > ' +
    '<div class="param" >Data Manager of experiment:</div> ' +
    '<div class="value" '+_DOCUMENT.experiment_info+' ><a class="link" target="_blank" href="'+experiment_link+' " >'+this._experiment.name+'</a></div> ' +
    '<div class="end" ></div> ' +
  '</div> ' +
  '<div id="tabs" > ' +
    '<ul> ' +
      '<li><a href="#all" >All files</a></li> ' +
    '</ul> ' +

    '<div id="all" > ' +
      '<div class="tab-cont" > ' +
        '<div class="table" ></div> ' +
      '</div> ' +
    '</div> ' +
  '</div> ' +
'</div> ' ;
                this.container.html(html) ;

                // Make sure the loading starts when rendering the object
                // for the first time.
                this.load() ;
            }
            this._update_table() ;
        } ;
    }
    Class.define_class (ExperimentBody, StackOfRows.StackRowBody, {}, {}) ;

    /**
     * The application for displaying and managin experiment-specific
     * policy exceptions.
     *
     * @returns {DRPMgr_Instrument}
     */
    function DRPMgr_Instrument (app_config, instr_name) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this._init() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        // This page only gets the manual updates to prevent any uncontrolled
        // interferences with long operations, such as loading detaild experiment
        // info or purging data.
        this.on_update = function () {
            if (this.active) {
                this._init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._app_config = app_config ;
        this._instr_name = instr_name ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        // The filesystem and experiment infor is loaded first

        this._experiments  = null ;
        this._filesystems  = null ;
        this._years        = null ;

        // The detailed info on experiments is loaded one-by-one
        // via a queue defined below. This phase will be temporarily
        // disabled until file system & experiment names are known.

        this._loading_is_allowed = false ;
        this._load_queue = [] ;
        this._load_queue_length = 0 ;
        this._load_queue_num_loaded = 0 ;

        this._populate_load_queue = function (experiments) {
            this._load_queue = [] ;
            this._load_queue_length = 0 ;
            this._load_queue_num_loaded = 0 ;
            for (var year in experiments) {
                var experiments_per_year = experiments[year] ;
                for (var i in experiments_per_year) {
                    var experiment = experiments_per_year[i] ;
                    this._load_queue.push({
                        year: year, exper_id: experiment.id
                    }) ;
                    this._load_queue_length++ ;
                }
            }
        } ;

        // ----------------------------------------------------------
        // Aggregated stats accross all experiments are being updated
        // as more experiments are being loaded (or cleared).

        this._all_years = {} ;
        this._reset_all_years = function () {
            this._all_years = _DEFAULT_TABLE_DATA() ;
        } ;

        // The cache of the StackOfRow objects (one per an experiment)
        // groupped by a year when the last run of an experiment started.

        this._year2stack = {} ;

        /**
         * The convenience method for creating configuration handlers
         * of the user input elements.
         *
         * @param {string} parameter
         * @returns {_FwkConfigHandlerCreator}
         */
        this._config_handler = function (parameter) {
            var config_scope = this.application_name+':'+this.context1_name+':'+this.context2_name ;
            return Fwk.config_handler(config_scope, parameter) ;
        } ;
        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ;
                if (!this_html) {
                    this_html =
'<div class="drpmgr-instrument" id="'+this._instr_name+'" >' +

  '<div class="info" id="updated" style="float:right;" >Loading...</div> ' +
  '<div style="clear:both;" ></div> ' +

  '<div id="ctrl" > ' +

    '<div class="control-group" '+_DOCUMENT.filesystem+' > ' +
      '<div class="control-group-title" >Filesystem</div> ' +
      '<div class="control-group-selector" > ' +
        '<select name="filesystem" ></select> ' +
      '</div> ' +
    '</div> ' +

    '<div class="control-group" '+_DOCUMENT.year+' > ' +
      '<div class="control-group-title" >Year</div> ' +
      '<div class="control-group-selector" > ' +
        '<select name="year" ></select> ' +
      '</div> ' +
    '</div> ' +

    '<div class="control-group control-group-buttons" > ' +
      '<button name="reset"  class="control-button" '                         +_DOCUMENT.reset +' >RESET</button> ' +
      '<button name="stop"   class="control-button control-button-important" '+_DOCUMENT.stop  +' >STOP</button> ' +
      '<button name="update" class="control-button" '                         +_DOCUMENT.update+' ><img src="../webfwk/img/Update.png" /></button> ' +
    '</div> ' +

    '<div class="control-group-end" ></div> ' +

  '</div>' +

  '<div id="tabs" >Loading...</div> ' +

'</div>' ;
                }
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('.drpmgr-instrument#'+this._instr_name) ;
            }
            return this._wa_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) {
                this._updated_elem = this._wa().children('#updated') ;
            }
            this._updated_elem.html(html) ;
        } ;
        this._ctrl = function () {
            if (!this._ctrl_elem) {
                this._ctrl_elem = this._wa().children('#ctrl') ;
            }
            return this._ctrl_elem ;
        } ;
        this._filesystem_selector = function (filesystems) {
            
            // Allow object initializion with complete or empty list
            // of choices. Force re-initialization if a complete list
            // is provided.
            if (!this._filesystem_selector_obj || !_.isUndefined(filesystems)) {

                // Clean up the previous object if exists.
                if (this._filesystem_selector_obj)
                    delete this._filesystem_selector_obj ;

                this._filesystem_selector_obj = new SelectOption (
                    this._ctrl().find('div.control-group-selector').children('select[name="filesystem"]') ,
                    {   disabled: true ,
                        options:
                            _.reduce (
                                _.isUndefined(filesystems) ? [] : filesystems ,
                                function (options, path) {
                                    options.push({value: path, text: path}) ;
                                    return options ;
                                } ,
                                [{value: '', default: true}]) ,
                        on_change:
                            function () {
                                _that._load() ;
                            } ,
                        config_handler: this._config_handler('filesystem')
                    }
                ) ;
            }
            return this._filesystem_selector_obj ;
        } ;
        this._year_selector = function (years) {
            
            // Allow object initializion with complete or empty list
            // of choices. Force re-initialization if a complete list
            // is provided.
            if (!this._year_selector_obj || !_.isUndefined(years)) {

                // Clean up the previous object if exists.
                if (this._year_selector_obj)
                    delete this._year_selector_obj ;
                
                this._year_selector_obj = new SelectOption (
                    this._ctrl().find('div.control-group-selector').children('select[name="year"]') ,
                    {   disabled: true ,
                        options:
                            _.reduce (
                                _.isUndefined(years) ? [] : years ,
                                function (options, year) {
                                    options.push({value: year.toString(), text: year.toString()}) ;
                                    return options ;
                                } ,
                                [{value: '', default: true}]) ,
                        on_change:
                            function () {
                                _that._load() ;
                            } ,
                        config_handler: this._config_handler('year')
                    }
                ) ;
            }
            return this._year_selector_obj ;
        } ;

        this._button_reset = function () {
            if (!this._button_reset_elem) {
                this._button_reset_elem = this._ctrl().find('.control-button[name="reset"]').button() ;
            }
            return this._button_reset_elem ;
        } ;
        this._button_stop = function () {
            if (!this._button_stop_elem) {
                this._button_stop_elem = this._ctrl().find('.control-button[name="stop"]').button() ;
            }
            return this._button_stop_elem ;
        } ;
        this._button_load = function () {
            if (!this._button_load_elem) {
                this._button_load_elem = this._ctrl().find('.control-button[name="update"]').button() ;
            }
            return this._button_load_elem ;
        } ;
        this._tabs = function (html) {

            // Allow object initializion with complete or empty list
            // of choices. Force re-initialization if a complete list
            // is provided.
            if (!this._tabs_elem || !_.isUndefined(html)) {

                // Reset caches in case of the re-initialization
                if (this._tabs_elem) {

                    if (this._all_years_table_obj)
                        delete this._all_years_table_obj ;

                    if (this._tab_elem) {
                        for (var i in this._tab_elem) {
                            delete this._tab_elem[i] ;
                        }
                        delete this._tab_elem ;
                    }
                    this._tabs_elem.tabs('destroy') ;
                    delete this._tabs_elem ;
                }
                this._tabs_elem = this._wa().children('#tabs') ;
                this._tabs_elem.html(_.isUndefined(html) ? '' : html) ;
                this._tabs_elem.tabs() ;
            }
            return this._tabs_elem ;
        } ;
        this._tab = function (id) {
            if (!this._tab_elem)     { this._tab_elem    = {} ; }
            if (!this._tab_elem[id]) { this._tab_elem[id] = this._tabs().children('#'+id) ; }
            return this._tab_elem[id] ;
        } ;
        this._all_years_table = function () {
            if (!this._all_years_table_obj) {
                this._all_years_table_obj = new SimpleTable.constructor (
                    this._tab('all-years').find('#table') ,
                    _TABLE_HDR ,
                    [] ,
                    {   text_when_empty: null ,
                        common_after_sort: function () {
                            _that._all_years_table().get_container().find('.control-button').button().click(function () {
                                var op = this.id ;
                                var context = this.name ;
                                console.log('DRPMgr_Instrument._all_years_table() '+op+' "'+context+'"') ;
                            }) ;
                        }
                    }
                ) ;
                this._all_years_table_obj.display() ;
            }
            return this._all_years_table_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // Render UI elements and put them into the disabled
            // state before the initial loading has finished.
            this._filesystem_selector().enable(false) ;
            this._year_selector()      .enable(false) ;

            this._button_reset()
                .button('disable')
                .click(function () { _that._reset() ; }) ;

            this._button_load()
                .button('disable')
                .click(function () { _that._load() ; }) ;

            // This operation only makes a sense for an on-going purge
            this._button_stop()
                .button('disable')
                .click(function () { _that._stop_purge() ; }) ;

            // Pre-load file systems and experiment names
            this._pre_load() ;
        } ;
        this._pre_load = function () {

            this._set_updated('Loading file systems and experiment names...') ;

            Fwk.web_service_GET (
                '../regdb/ws/drp_instruments_get.php' ,
                {   instr_name: this._instr_name ,
                    fs:         this._filesystem_selector().value() ,
                    year:       this._year_selector()      .value()
                } ,
                function (data) {

                    _that._experiments = data.experiments ;
                    
                    // Iniialize selector of filesystems and years only onece when
                    // loading this information for the first time.
                    if (!_that._filesystems) {
                        _that._filesystems = data.filesystems ;
                        _that._years       = data.years ;
                        _that._filesystem_selector(_that._filesystems) ;
                        _that._year_selector      (_that._years) ;
                    }
                    _that._reset_all_years() ;

                    // Update the control elements
                    // Initialize tabs and and display tables with experiment info
                    _that._pre_display() ;

                    _that._filesystem_selector().enable(true) ;
                    _that._year_selector()      .enable(true) ;

                    _that._button_reset().button('enable') ;
                    _that._button_load ().button('enable') ;

                    // Feed all experiments into the load queue and initiate
                    // the main loading process.
                    _that._populate_load_queue(data.experiments) ;
                    _that._loading_is_allowed = true ;
                    _that._load_next() ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;
        this._pre_display = function () {

            // Initialize tabs only for experiments which have been
            // reported by the service
            
            var years = _.filter (
                this._years ,
                function (y) {
                    return _.has(_that._experiments, y) ;
                }
            ) ;
            var html =
'<ul>' +
  '<li><a href="#all-years" >All years</a></li> ' + _.reduce(years, function (html, y) { return html +=
  '<li><a href="#'+y+'" >'+y+'</a></li> ' ; }, '') +
'</ul> ' +
'<div id="all-years" > ' +
  '<div class="tab-cont" > ' +
    '<div id="table"></div>' +
  '</div>' +
'</div>' +                                          _.reduce(years, function (html, y) { return html +=
'<div id="'+y+'" > ' +
  '<div class="tab-cont" > ' +
    '<div class="stack" ></div> ' +
  '</div>' +
'</div>' ; }, '') ;
            this._tabs(html) ;

            // Display the initial state of the summary table. It will
            // get updated over time once individual experiments will get
            // loaded.
            this._update_all_years_table() ;
            
            // Display the initial state of experiments in each year. They will
            // be pdated as more detailed stats will be getting loaded for
            // each experiment.
            this._display() ;
        } ;
        this._update_all_years_table = function () {
            var rows = [] ;
            for (var i in this._all_years) {
                var t = this._all_years[i] ;
//                rows.push([
//                    t.title ,
//                    t['SHORT-TERM'].num_files  ? t['SHORT-TERM'].num_files : '' ,
//                    t['SHORT-TERM'].size_gb    ? t['SHORT-TERM'].size_gb   : '' ,
//                    t ,
//                    t['MEDIUM-TERM'].num_files ? t['MEDIUM-TERM'].num_files : '' ,
//                    t['MEDIUM-TERM'].size_gb   ? t['MEDIUM-TERM'].size_gb   : '' ,
//                    t ,
//                    t['HPSS'].num_files ? t['HPSS'].num_files : '' ,
//                    t['HPSS'].size_gb   ? t['HPSS'].size_gb   : ''
//                ]) ;

                var short_num_files  = t['SHORT-TERM']['xtc'].num_files + t['SHORT-TERM']['hdf5'].num_files ,
                    short_size_gb    = t['SHORT-TERM']['xtc'].size_gb   + t['SHORT-TERM']['hdf5'].size_gb ;

                var medium_num_files = t['MEDIUM-TERM']['xtc'].num_files + t['MEDIUM-TERM']['hdf5'].num_files ,
                    medium_size_gb   = t['MEDIUM-TERM']['xtc'].size_gb   + t['MEDIUM-TERM']['hdf5'].size_gb ;

                var hpss_num_files   = t['HPSS']['xtc'].num_files + t['HPSS']['hdf5'].num_files ,
                    hpss_size_gb     = t['HPSS']['xtc'].size_gb   + t['HPSS']['hdf5'].size_gb ;

                rows.push([
                    t.title ,
                    short_num_files  ? short_num_files  : '' ,
                    short_size_gb    ? short_size_gb    : '' ,
                    t ,
                    medium_num_files ? medium_num_files : '' ,
                    medium_size_gb   ? medium_size_gb   : '' ,
                    t ,
                    hpss_num_files   ? hpss_num_files   : '' ,
                    hpss_size_gb     ? hpss_size_gb     : ''
                ]) ;
            }
           this._all_years_table().load(rows) ;
           
           // The butons will be enable when the loading is over
           this._all_years_table().get_container().find('.control-button').button('disable') ;
        } ;

        this._reset = function () {
            this._filesystem_selector().set_value('') ;
            this._year_selector()      .set_value('') ;
            this._load() ;
        } ;
        this._load = function () {
            // Render UI elements and put them into the disabled
            // state before the initial loading has finished.
            this._filesystem_selector().enable(false) ;
            this._year_selector()      .enable(false) ;

            this._button_reset().button('disable') ;
            this._button_stop ().button('disable') ;
            this._button_load ().button('disable') ;

            // Pre-load file systems and experiment names
            this._pre_load() ;
        } ;

        /**
         * Check the loading queue and initiate loading of the first eligible
         * experiment. Return 'true' if teh loading has started. Return 'false'
         * otherwise.
         * 
         * @returns {Boolean}
         */
        this._load_next = function () {
            if (!this._loading_is_allowed) return false;
            do {
                var e = this._load_queue.shift() ;
                if (_.isUndefined(e)) break ;

                var stack  = this._year2stack[e.year].stack ;
                var row_id = this._year2stack[e.year].exper_id2row_id[e.exper_id] ;

                var row_body = stack.get_row_by_id(row_id).data_object.body ;
                this._load_queue_num_loaded++ ;
                if (row_body.ready2load()) {
                    row_body.load() ;
                    this._set_updated('Loading experiments: '+this._load_queue_num_loaded+' / '+this._load_queue_length+'&nbsp; [ <b>'+row_body._experiment.name+'</b> ]') ;
                    return true ;
                }

            } while(true) ;

            // The queue is empty, no more work
            return false ;
        } ;

        this._display = function () {
            var hdr = [
                {id: 'experiment',      title: 'Experiment',     width:  90} ,
                {id: 'last_run',        title: 'Last Run',       width:  90} ,
                {id: 'total',           title: 'Total',          width:  65, align: "right"} ,
                {id: 'sterm',           title: 'S-TERM',         width:  65, align: "right"} ,
                {id: 'sterm_separator', title: '&nbsp;',         width:   5, align: "center"} ,
                {id: 'sterm_expired',   title: 'expired',        width:  55, align: "left"} ,
                {id: 'mterm',           title: 'M-TERM',         width:  60, align: "right"} ,
                {id: 'mterm_separator', title: '&nbsp;',         width:   5, align: "center"} ,
                {id: 'mterm_expired',   title: 'expired',        width:  55, align: "left"} ,
                {id: '>'} ,
                {id: 'fs',              title: 'Filesystem',     width: 120} ,
                {id: 'access',          title: 'ATIME (months ago) [0,1][2] .. [23][24+]', width: 260, align: "left"}
            ] ;
            var options = {
                expand_buttons: false ,
//                theme: 'stack-theme-brown' ,
                theme: 'stack-theme-aliceblue' ,
                allow_replicated_headers: true
            } ;
            for (var year in this._experiments) {

                var stack = new StackOfRows.StackOfRows (
                    hdr ,
                    null ,      // rows will be added later
                    options
                ) ;
                this._year2stack[year] = {
                    stack: stack ,
                    // There will be numeric identifiers of rows for each experiment.
                    // They will be used for further communications when loading
                    // data of teh experiments.
                    exper_id2row_id: {}
                } ;

                var experiments = this._experiments[year] ;
                for (var i in experiments) {
                    var experiment = experiments[i] ;
                    this._year2stack[year].exper_id2row_id[experiment.id] = stack.add_row({
                        title: {
                            experiment:      experiment.name ,
                            last_run:        experiment.last_run.day ,
                            total:           '<span class="info" >Loading...</span>' ,
                            sterm:           '&nbsp;' ,
                            sterm_separator: '&nbsp;' ,
                            sterm_expired:   '&nbsp;' ,
                            mterm:           '&nbsp;' ,
                            mterm_separator: '&nbsp;' ,
                            mterm_expired:   '&nbsp;' ,
                            access:          '&nbsp;' ,
                            fs:              experiment.fs
                        } ,
                        body: new ExperimentBody(this, year, experiment) ,
                        block_common_expand: true
                    }) ;
                }
                stack.display(this._tab(year).children('.tab-cont').children('.stack')) ;
            }
        } ;
        
        /**
         * Stop the on-going purge
         *
         * @returns {undefined}
         */
        this._stop_purge = function () {
        } ;

        // --------------------------------
        // Callbacks from the child objects
        // --------------------------------

        function access2html (data) {
            var html = '' ;
            for (var i in data) {
                var t = data[i] ;
                switch (t.category) {
                    case 'total':
                    case 'expired':
                        break ;
                    default:
                        // ATTENTION: Prepending to reverse the order.
                        var disk  = t['SHORT-TERM'].size_gb + t['MEDIUM-TERM'].size_gb ;
                        var html2add = disk ?
                            '<div style="float:left; width:9px;height:100%; border-right: solid 1px #A6C9E2; background-color: black;">&nbsp;</div>' :
                            '<div style="float:left; width:9px;height:100%; border-right: solid 1px #A6C9E2;">&nbsp;</div>' ;
                        html = html2add + html ;
                }
            }
            html = '<div style="float:left; width:0px;height:100%; border-right: solid 1px #A6C9E2;">&nbsp;</div>' + html ;
            html += '<div style="clear:both"></div>' ;
            return html ;
        }
        this.finished_loading = function (year, experiment, data, updated) {

            this._set_updated('Last updated: <b>'+updated+'</b>') ;
            
            // Update the title of the corresponding table row
            var stack  = this._year2stack[year].stack ;
            var row_id = this._year2stack[year].exper_id2row_id[experiment.id] ;
            var row    = stack.get_row_by_id(row_id) ;

            var title_data = {
                experiment:      experiment.name ,
                last_run:        experiment.last_run.day ,
                total:           '&lt;1' ,
                sterm:           '&nbsp;' ,
                sterm_separator: '&nbsp;' ,
                sterm_expired:   '' ,
                mterm:           '&nbsp;' ,
                mterm_separator: '&nbsp;' ,
                mterm_expired:   '' ,
                access:          access2html(data) ,
                fs:              experiment.fs
            } ;
            if (experiment.id == 445) {
                console.log(experiment, data) ;
            }
            for (var i in data) {
                var t = data[i] ;
                switch (t.category) {
                    case 'total':
                        var hpss_size_gb   = t['HPSS']       ['xtc'].size_gb + t['HPSS']       ['hdf5'].size_gb ;
                        var short_size_gb  = t['SHORT-TERM'] ['xtc'].size_gb + t['SHORT-TERM'] ['hdf5'].size_gb ;
                        var medium_size_gb = t['MEDIUM-TERM']['xtc'].size_gb + t['MEDIUM-TERM']['hdf5'].size_gb ;

                        if (hpss_size_gb)   title_data.total = hpss_size_gb ;
                        if (short_size_gb)  title_data.sterm = short_size_gb ;
                        if (medium_size_gb) title_data.mterm = medium_size_gb ;
                        break ;
                    case 'expired':
                        var short_size_gb  = t['SHORT-TERM'] ['xtc'].size_gb + t['SHORT-TERM'] ['hdf5'].size_gb ;
                        var medium_size_gb = t['MEDIUM-TERM']['xtc'].size_gb + t['MEDIUM-TERM']['hdf5'].size_gb ;
                        if (short_size_gb)  title_data.sterm_expired = short_size_gb ;
                        if (medium_size_gb) title_data.mterm_expired = medium_size_gb ;
                        break ;
                }
            }
            title_data.sterm_expired = title_data.sterm_expired ? '<span style="color:red;">'+title_data.sterm_expired+'</span>' : '&nbsp;' ;
            title_data.mterm_expired = title_data.mterm_expired ? '<span style="color:red;">'+title_data.mterm_expired+'</span>' : '&nbsp;' ;
            row.update_title(title_data) ;

            // Recalculate counters in the "all years" table
            ;

            // Check if all experiments are loaded and if so then enable
            // control buttons in the "all years" table. At this point
            // an operator may take actions.
            if (!this._load_next()) {
                console.log('DRPMgr_Instrument.finished_loading() all experiments are now loaded') ;
                this._filesystem_selector().enable(true) ;
                this._year_selector()      .enable(true) ;
                this._button_reset().button('enable') ;
                this._button_load ().button('enable') ;
            }
        } ;

    }
    Class.define_class (DRPMgr_Instrument, FwkApplication, {}, {}) ;
    
    return DRPMgr_Instrument ;
}) ;