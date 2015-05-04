define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class',       'webfwk/FwkApplication', 'webfwk/Fwk' ,
    'webfwk/SimpleTable', 'webfwk/TextInput',      'webfwk/SelectOption' ,
    'webfwk/Checkbox'] ,

function (
    cssloader ,
    Class,       FwkApplication, Fwk ,
    SimpleTable, TextInput,      SelectOption ,
    Checkbox) {

    cssloader.load('../sysmon/css/DMMon_FM_Status.css') ;

    /**
     * The application for displaying the status of the file migration
     *
     * @returns {DMMon_History}
     */
    function DMMon_FM_Status (app_config) {

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

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 60 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._app_config = app_config ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        // The loading scheduler allows for keeping track of other loading
        // requests while processing one. Once the active one is over
        // the schedule may instantly trigger the next one if the counter
        // of requests in teh queue is not 0.
        this._is_loading = false ;
        this._load_queue = 0 ;      // num requests in the loading queue

        this._files = {} ;
        this._experiments = {} ;

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

        var _DELAYS_DEF = [
            {sec:    2*3600, name:  '2 hours'} ,
            {sec:    4*3600, name:  '4 hours'} ,
            {sec:    8*3600, name:  '8 hours'} ,
            {sec:   12*3600, name: '12 hours'} ,
            {sec:   18*3600, name: '18 hours'} ,
            {sec:   24*3600, name:      'day'} ,
            {sec: 2*24*3600, name:   '2 days'} ,
            {sec: 7*24*3600, name:     'week'}
        ] ;
        function _DOCUMENT_METHOD (str) {
            return 'data="'+str+'"' ;
        }
        var _DOCUMENT = {
            instr:
                _DOCUMENT_METHOD (
                    'Narrow a search down to the specified instrument. \n' +
                    'Otherwise all instruments will be assumed.') ,
            since:
                _DOCUMENT_METHOD (
                    'Specify how far to look back into the history of \n' +
                    'past transfers.') ,
            delay:
                _DOCUMENT_METHOD (
                    'Specify a threshold for delayed transfers to be ignored. \n' +
                    'The default value of 0 would include all transfer regardless \n' +
                    'of how long they are delayed.') ,
            complete:
                _DOCUMENT_METHOD (
                    'Specify whether to search for full completed transcations. \n' +
                    'These are the ones for which teh files are successfully archived \n' +
                    'to HPSS.') ,
            reset: 
                _DOCUMENT_METHOD (
                    'Click this button to reset the form to its default state. \n' +
                    'This will also update the table with fresh results for \n' +
                    'the new search criteria.') ,
            search:
                _DOCUMENT_METHOD (
                    'Click this button to make a new search and update the table \n' +
                    'with the present search criteria.')
        } ;
        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dmmon-fm-status" >' +

  '<div class="info" id="updated" style="float:right;" ></div> ' +
  '<div style="clear:both;" ></div> ' +

  '<div id="ctrl" style="float:left;" > ' +

    '<div class="control-group" '+_DOCUMENT.instr+' > ' +
      '<div class="control-group-title" >Instr.</div> ' +
      '<div class="control-group-selector" > ' +
        '<select name="instr" ></select> ' +
      '</div> ' +
    '</div> ' +

    '<div class="control-group" '+_DOCUMENT.since+' > ' +
      '<div class="control-group-title" >Search last</div> ' +
      '<div class="control-group-selector" > ' +
        '<select name="since" ></select> ' +
      '</div> ' +
    '</div> ' +
  
    '<div class="control-group" '+_DOCUMENT.delay+' > ' +
      '<div class="control-group-title" >Delayed by [sec]</div> ' +
      '<div class="control-group-selector" > ' +
        '<input type="text" size="1" name="delay" /> ' +
      '</div> ' +
    '</div> ' +
  
    '<div class="control-group" '+_DOCUMENT.complete+' > ' +
      '<div class="control-group-title" >Incl. complete</div> ' +
      '<div class="control-group-selector" > ' +
        '<input type="checkbox" name="complete" /> ' +
      '</div> ' +
    '</div> ' +
    
    '<div class="control-group control-group-buttons" > ' +
      '<button class="control-button" name="reset"  '+_DOCUMENT.reset+ ' >RESET</button> ' +
      '<button class="control-button" name="update" '+_DOCUMENT.search+' ><img src="../webfwk/img/Update.png" /></button> ' +
    '</div> ' +
    
    '<div class="control-group-end" ></div> ' +
    
  '</div>' +
  '<div style="clear:both;" ></div> ' +

  '<div id="files" >' +
  '</div>' +
'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dmmon-fm-status') ;
            }
            return this._wa_elem ;
        } ;

        this._instr_selector = function () {
            if (!this._instr_selector_obj) {
                this._instr_selector_obj = new SelectOption (
                    this._wa().find('div.control-group-selector').children('select[name="instr"]') ,
                    {   disabled: true ,
                        options:
                            _.reduce (
                                this._app_config.instruments ,
                                function (options, instr_name) {
                                    options.push({value:instr_name, text: instr_name}) ;
                                    return options ;
                                } ,
                                [{value: '', default: true}]) ,
                        on_change:
                            function () { _that._load() ; } ,
                        config_handler:
                            this._config_handler('instr')
                    }
                ) ;
            }
            return this._instr_selector_obj ;
        } ;
        this._since_selector = function () {
            if (!this._since_selector_obj) {
                this._since_selector_obj = new SelectOption (
                    this._wa().find('div.control-group-selector').children('select[name="since"]') ,
                    {   disabled: true ,
                        options:
                            _.reduce (
                                _DELAYS_DEF ,
                                function (options, d) {
                                    options.push({value:''+d.sec, text: d.name}) ;
                                    return options ;
                                } ,
                                []) ,
                        on_change:
                            function () { _that._load() ; } ,
                        config_handler:
                            this._config_handler('since')
                    }
                ) ;
            }
            return this._since_selector_obj ;
        } ;
        this._delay_selector = function () {
            if (!this._delay_selector_obj) {
                this._delay_selector_obj = new TextInput (
                    this._wa().find('div.control-group-selector').children('input[name="delay"]') ,
                    {   disabled: true ,
                        default_value:  '0' ,
                        on_validate:
                            function (str) { var delay = parseInt(str) ; return delay >= 0 ? delay : 0 ; } ,
                        on_change:
                            function () { _that._load() ; } ,
                        config_handler:
                            this._config_handler('delay')
                    }
                ) ;
            }
            return this._delay_selector_obj ;
        } ;
        this._complete_selector = function () {
            if (!this._complete_selector_obj) {
                this._complete_selector_obj = new Checkbox (
                    this._wa().find('div.control-group-selector').children('input[name="complete"]') ,
                    {   disabled: true ,
                        default_value:  1 ,
                        on_change:
                            function () { _that._load() ; } ,
                        config_handler:
                            this._config_handler('complete')
                    }
                ) ;
            }
            return this._complete_selector_obj ;
        } ;
        function _begin_delay2html (stage) {
            switch (stage.status) {
                case 'P' :
                case 'C' :
                    return '<span style="font-weight:bold;" >' + stage.begin_delay+ '</span>' ;
            }
            return '<span style="color:red;" >' + stage.begin_delay+ '</span>' ;
        }
        function _end_delay2html (stage) {
            switch (stage.status) {
                case 'C' :
                    return '<span style="font-weight:bold;" >' + stage.end_delay+ '</span>' ;
            }
            return '<span style="color:red;" >' + stage.end_delay+ '</span>' ;
        }
        function _rate2html (stage) {
            switch (stage.status) {
                case 'C' :
                    return stage.rate ;
            }
            return '' ;
        }
        this._table = function () {

            if (!this._table_obj) {
                var rows = [] ;
                var hdr = [
                    {   name: 'experiment', align: 'right'} ,
                    {   name: 'run',  sorted: false, align: 'right'} ,
                    {   name: 'file'} ,
                    {   name: 'size', align: 'right' ,
                        type: {
                            to_string:      function (a)   { return a.value  + '&nbsp;&nbsp;<span style="font-weight:bold;" >' + a.units+ '</span>' ; } ,
                            compare_values: function (a,b) { return a.bytes - b.bytes ; }
                        }
                    } ,
                    {   name: 'created' ,
                        type: {
                            to_string:      function (a)   { return a.day  + '&nbsp;&nbsp;<span style="font-weight:bold;" >' + a.hms+ '</span>' ; } ,
                            compare_values: function (a,b) { return a.sec - b.sec ; }
                        }
                    } ,
                    {   name: 'DSS &#8674; FFB', coldef: [
                            {   name: 'DSS host'} ,
                            {   name: 'delay', coldef: [
                                    {   name: 'begin', align: 'right' ,
                                        type: {
                                            to_string:      function (a)   { return _begin_delay2html(a) ; } ,
                                            compare_values: function (a,b) { return a.begin_delay - b.begin_delay ; }
                                        }
                                    } ,
                                    {   name: 'end', align: 'right' ,
                                        type: {
                                            to_string:      function (a)   { return _end_delay2html(a) ; } ,
                                            compare_values: function (a,b) { return a.end_delay - b.end_delay ; }
                                        }
                                    }
                                ]
                            } ,
                            {   name: 'MB/s', align: 'right' ,
                                type: {
                                    to_string:      function (a)   { return _rate2html(a) ; } ,
                                    compare_values: function (a,b) { return a.rate - b.rate ; }
                                }
                            }
                        ]
                    } ,
                    {   name: 'FFB &#8674; ANA', coldef: [
                            {   name: 'FFB host'} ,
                            {   name: 'delay', coldef: [
                                    {   name: 'begin', align: 'right' ,
                                        type: {
                                            to_string:      function (a)   { return _begin_delay2html(a) ; } ,
                                            compare_values: function (a,b) { return a.begin_delay - b.begin_delay ; }
                                        }
                                    } ,
                                    {   name: 'end',   align: 'right' ,
                                        type: {
                                            to_string:      function (a)   { return _end_delay2html(a) ; } ,
                                            compare_values: function (a,b) { return a.end_delay - b.end_delay ; }
                                        }
                                    }
                                ]
                            } ,
                            {   name: 'MB/s', align: 'right' ,
                                type: {
                                    to_string:      function (a)   { return _rate2html(a) ; } ,
                                    compare_values: function (a,b) { return a.rate - b.rate ; }
                                }
                            }
                        ]
                    } ,
                    {   name: 'ANA &#8674; HPSS', coldef: [
                            {   name: 'delay', coldef: [
                                    {   name: 'begin', align: 'right' ,
                                        type: {
                                            to_string:      function (a)   { return _begin_delay2html(a) ; } ,
                                            compare_values: function (a,b) { return a.begin_delay - b.begin_delay ; }
                                        }
                                    } ,
                                    {   name: 'end',   align: 'right' ,
                                        type: {
                                            to_string:      function (a)   { return _end_delay2html(a) ; } ,
                                            compare_values: function (a,b) { return a.end_delay - b.end_delay ; }
                                        }
                                    }
                                ]
                            } ,
                            {   name: 'MB/s', align: 'right' ,
                                type: {
                                    to_string:      function (a)   { return _rate2html(a) ; } ,
                                    compare_values: function (a,b) { return a.rate - b.rate ; }
                                }
                            }
                        ]
                    } ,
                    {   name: 'status'}
                ] ;
                this._table_obj = new SimpleTable.constructor (
                    this._wa().find('#files') ,
                    hdr ,
                    rows ,
                    {
                        default_sort_column:  4 ,           /* created */
                        default_sort_forward: false ,       /* sort in ascending order */
                        text_when_empty:      null
                    }) ;
                this._table_obj.display() ;
            }
            return this._table_obj ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._button_reset = function () {
            if (!this._button_reset_elem) {
                this._button_reset_elem = this._wa().find('.control-button[name="reset"]').button() ;
            }
            return this._button_reset_elem ;
        } ;
        this._button_load = function () {
            if (!this._button_load_elem) {
                this._button_load_elem = this._wa().find('.control-button[name="update"]').button() ;
            }
            return this._button_load_elem ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // Touch visual objects to make sure they're displayed before
            // going for any loading of data
            this._button_reset().click(function () { _that._reset() ; }) ;
            this._button_load() .click(function () { _that._load() ;  }) ;

            this._instr_selector()   .enable() ;
            this._since_selector()   .enable() ;
            this._delay_selector()   .enable() ;
            this._complete_selector().enable() ;

            this._table() ;

            // Proceed to the first loading
            this._load() ;
        } ;
        this._reset = function () {
            this._instr_selector()   .set_value('') ;                       // all instruments
            this._since_selector()   .set_value(''+_DELAYS_DEF[0].sec) ;    // the first option
            this._delay_selector()   .set_value('0') ;                      // all delays
            this._complete_selector().set_value(1) ;
            this._load() ;
        } ;
        this._load = function () {

            if (!this._is_initialized) return ;

            if (this._is_loading) {
                // Schedule another loading when the current one is over
                this._load_queue++ ;
                return ;
            }
            this._is_loading = true ;

            this._set_updated('Loading...') ;
            this._button_reset().button('disable') ;
            this._button_load() .button('disable') ;
            var selectors = this._wa().find('.control-group-selector').children() ;
            selectors.attr('disabled', 'disabled') ;

            var params = {
                begin_time:       Fwk.now().sec - parseInt(this._since_selector()   .value()) ,
                min_delay_sec:                    parseInt(this._delay_selector()   .value()) ,
                include_complete:                          this._complete_selector().value() ? 1 : 0
            } ;
            var instr_name = this._instr_selector().value() ;
            if (instr_name !== '') params.instr_name = instr_name ;

            Fwk.web_service_GET (
                '../sysmon/ws/dmmon_fm_get.php' ,
                params ,
                function (data) {
                    _that._files = data.files ;
                    _that._experiments = data.experiments ;
                    _that._display() ;
                    _that._set_updated('Last updated: <b>'+data.updated+'</b>') ;
                    _that._button_reset().button('enable') ;
                    _that._button_load() .button('enable') ;
                    selectors.removeAttr('disabled') ;

                    // When done with this loading check if there is another
                    // outstanding request and proceed to teh one.
                    _that._is_loading = false ;
                    if (_that._load_queue) {
                        _that._load_queue = 0 ;
                        _that._load() ;
                    }
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._button_reset().button('enable') ;
                    _that._button_load().button('enable') ;
                    selectors.removeAttr('disabled') ;
                    
                    // When done with this loading check if there is another
                    // outstanding request and proceed to teh one.
                    _that._is_loading = false ;
                    if (_that._load_queue) {
                        _that._load_queue = 0 ;
                        _that._load() ;
                    }
                }
            ) ;
        } ;

        function _experiment2html (exper_name, exper_id) {
            var title =
                'Open Web Portal of the experiment in new window/tab' ;
            var html =
                '<a class="link" target="_blank" href="../portal/index.php?exper_id='+exper_id+'" ' +
                'data="'+title+'" ' +
                '>'+exper_name+'</a>' ;
            return html ;
        }
        function _run2html (run, exper_id) {
            var title =
                'Open Web Portal of the experiment in new window/tab \n' +
                'and find the run in a vicinity of e-Log messages' ;
            var html =
                '<a class="link" target="_blank" href="../portal/index.php?exper_id='+exper_id +
                '&app=elog:search&params=run:'+run+'" ' +
                'data="'+title+'" ' +
                '>'+run+'</a>' ;
            return html ;
        }
        function _status2html (f) {
            switch (f.DSS2FFB.status) {
                case 'W': return 'DSS' ;
                case 'P': return 'DSS &#8674; FFB' ;
            }
            switch (f.FFB2ANA.status) {
                case 'W': return 'FFB' ;
                case 'P': return 'FFB &#8674; ANA' ;
            }
            switch (f.ANA2HPSS.status) {
                case 'W': return 'ANA' ;
                case 'P': return 'ANA &#8674; HPSS' ;
            }
            return '' ;
        }
        this._display = function () {
            var rows = [] ;
            for (var i in this._files) {
                var f = this._files[i] ;
                rows.push([
                    _experiment2html(this._experiments[f.exper_id], f.exper_id) ,
                    _run2html(f.run, f.exper_id) ,
                    f.name + '.' + f.type ,
                    f.size ,
                    f.created ,
                    f.DSS2FFB.host ,
                    f.DSS2FFB ,
                    f.DSS2FFB ,
                    f.DSS2FFB ,
                    f.FFB2ANA.host ,
                    f.FFB2ANA ,
                    f.FFB2ANA ,
                    f.FFB2ANA ,
                    f.ANA2HPSS ,
                    f.ANA2HPSS ,
                    f.ANA2HPSS ,
                    _status2html(f)
                ]) ;
            }
            this._table().load(rows) ;
        } ;
    }
    Class.define_class (DMMon_FM_Status, FwkApplication, {}, {}) ;
    
    return DMMon_FM_Status ;
}) ;

