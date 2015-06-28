define ([
    'webfwk/CSSLoader', 'webfwk/PropList' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader, PropList ,
    Class,     FwkApplication, Fwk) {

    cssloader.load('../portal/css/HDF5_Translator.css') ;

    /**
     * The application for displaying and managing HDF5 translation
     * requests of the experiment.
     *
     * @returns {HDF5_Translator}
     */
    function HDF5_Translator (service, experiment, access_list) {

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

        this._update_interval_sec = 300 ;
        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > this._update_interval_sec) {
                    this._prev_update_sec = now_sec ;
                    this._status_load() ;
                    this._config_load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.service     = service;
        this.experiment  = experiment ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._can_manage = function () { return this.access_list.hdf5.is_data_administrator ; } ;

        this._status_reverse_order = false ;
        this._status_last_request  = [] ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ||
'<div id="hdf5-translator" > ' +

  '<div id="tabs"> ' +

    '<ul> ' +
      '<li><a href="#status">Status</a></li> ' +
      '<li><a href="#config">Configuration</a></li> ' +
    '</ul> ' +

    '<div id="status" class="panel" > ' +
      '<div id="ctrl">' +
        '<div class="info" id="updated" style="float:right;" >Loading...</div> ' +
        '<div style="clear:both;" ></div> ' +
        '<div class="group" title="' +
          'Put a run number or a range of runs to activate the filter.\n ' +
          'Range can be specified like this:\n ' +
          '  10\n ' +
          '  10-20\n ' +
          '  10-\n ' +
          '  -20\n ' +
          'Note that the range cane be open on the either end.\n ' +
          'Then press RETURN to activate search." > ' +
          '<span class="label" >Search runs:</span> ' +
          '<input class="update-trigger" type="text" name="runs" value="" />' +
        '</div> ' +
        '<div class="group" title="Select non-blank option to activate the filter"> ' +
          '<span class="label" >Translation state:</span> ' +
          '<select class="update-trigger" name="state" > ' +
            '<option>any</option> ' +
            '<option>FINISHED</option> ' +
            '<option>FAILED</option> ' +
            '<option>TRANSLATING</option> ' +
            '<option>QUEUED</option> ' +
            '<option>NOT-TRANSLATED</option> ' +
          '</select> ' +
        '</div> ' +
        '<div class="buttons" style="float:left;" > ' +
          '<button class="control-button" name="reset"  title="reset the form" >RESET FORM</button> ' +
          '<button class="control-button" name="update" title="click to update the information from the database" ><img src="../webfwk/img/Update.png" /></button> ' +
        '</div> ' +
        '<div style="clear:both;" ></div> ' +
      '</div> ' +
      '<div id="body" > ' +
        '<div class="info" id="info"    style="float:left;" >Loading...</div> ' +
        '<div style="clear:both;" ></div> ' +
        '<div id="ctrl" > ' +
          '<button class="control-button" name="reverse" >REVERSE ORDER</button> ' +
          '<button class="control-button" name="translate" >TRANSLATE ALL</button> ' +
          '<button class="control-button" name="stop" >STOP ALL</button> ' +
        '</div> ' +
        '<div id="viewer" style="width:100%;" ></div> ' +
      '</div> ' +
    '</div> ' +

    '<div id="config" class="panel" > ' +
      '<div id="ctrl" > ' +
        '<div class="info" id="updated" style="float:right;" >&nbsp;</div> ' +
        '<div style="clear:both;" ></div> ' +
        '<div style="float:left;" class="hdf5-notes" > ' +
          'This application presents configuration parameters of the translation service. ' +
          'Please, be aware that any changes made to the values of the parameter would ' +
          'be applied to all future requests. ' +
          'The detailed description of various parameters of the service in this document: ' +
          '"<a class="link" stype="white-space:nowrap;" target="_blank" href="https://confluence.slac.stanford.edu/display/PSDM/The+XTC-to-HDF5+Translator" >The XTC-to-HDF5 Translator</a>". ' +
        '</div> ' +
        '<div style="float:left;" class="buttons" > ' +
          '<button name="edit"    class="control-button control-button-important" title="edit configuration options"         >EDIT  </button> ' +
          '<button name="save"    class="control-button"                          title="save modifications to the database" >SAVE  </button> ' +
          '<button name="cancel"  class="control-button"                          title="cancel the editing session"         >CANCEL</button> ' +
          '<button name="update"  class="control-button"                          title="update from the database"           ><img src="../webfwk/img/Update.png" /></button> ' +
        '</div> ' +
        '<div style="clear:both;" ></div> ' +
      '</div> ' +
      '<div id="body" > ' +
        '<div id="editor" ></div> ' +
      '</div> ' +
    '</div> ' +

  '</div> ' ;
'</div>' ;
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('div#hdf5-translator') ;
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



        this._status_panel = function () {
            if (!this._status_panel_elem) {
                this._status_panel_elem = this._tabs().children('.panel#status') ;
            }
            return this._status_panel_elem ;
        } ;
        this._status_ctrl = function () {
            if (!this._status_ctrl_elem) {
                this._status_ctrl_elem = this._status_panel().children('#ctrl') ;
            }
            return this._status_ctrl_elem ;
        } ;
        this._status_runs_selector = function () {
            if (!this._status_runs_selector_elem) {
                this._status_runs_selector_elem = this._status_ctrl().find('input[name="runs"]') ;
            }
            return this._status_runs_selector_elem ;
        } ;
        this._status_state_selector = function () {
            if (!this._status_state_selector_elem) {
                this._status_state_selector_elem = this._status_ctrl().find('select[name="state"]') ;
            }
            return this._status_state_selector_elem ;
        } ;
        this._status_set_info = function (html) {
            if (!this._status_info_elem) {
                this._status_info_elem = this._status_body().find('#info') ;
            }
            this._status_info_elem.html(html) ;
        } ;
        this._status_set_updated = function (html) {
            if (!this._status_updated_elem) {
                this._status_updated_elem = this._status_ctrl().find('#updated') ;
            }
            this._status_updated_elem.html(html) ;
        } ;
        this._status_button_reset = function () {
            if (!this._status_button_reset_elem) {
                this._status_button_reset_elem = this._status_ctrl().find('button[name="reset"]').button() ;
            }
            return this._status_button_reset_elem ;
        } ;
        this._status_button_search = function () {
            if (!this._status_button_search_elem) {
                this._status_button_search_elem = this._status_ctrl().find('button[name="update"]').button() ;
            }
            return this._status_button_search_elem ;
        } ;
        this._status_body = function () {
            if (!this._status_body_elem) {
                this._status_body_elem = this._status_panel().children('#body') ;
            }
            return this._status_body_elem ;
        } ;
        this._status_body_ctrl = function () {
            if (!this._status_body_ctrl_elem) {
                this._status_body_ctrl_elem = this._status_body().children('#ctrl') ;
            }
            return this._status_body_ctrl_elem ;
        } ;
        this._status_button_reverse = function () {
            if (!this._status_button_reverse_elem) {
                this._status_button_reverse_elem = this._status_body_ctrl().find('button[name="reverse"]').button() ;
            }
            return this._status_button_reverse_elem ;
        } ;
        this._status_button_translate = function () {
            if (!this._status_button_translate_elem) {
                this._status_button_translate_elem = this._status_body_ctrl().find('button[name="translate"]').button() ;
            }
            return this._status_button_translate_elem ;
        } ;
        this._status_button_stop = function () {
            if (!this._status_button_stop_elem) {
                this._status_button_stop_elem = this._status_body_ctrl().find('button[name="stop"]').button() ;
            }
            return this._status_button_stop_elem ;
        } ;
        this._status_viewer = function () {
            if (!this._status_viewer_elem) {
                this._status_viewer_elem = this._status_body().find('#viewer') ;
            }
            return this._status_viewer_elem ;
        } ;





        this._config_panel = function () {
            if (!this._config_panel_elem) {
                this._config_panel_elem = this._tabs().children('.panel#config') ;
            }
            return this._config_panel_elem ;
        } ;
        this._config_ctrl = function () {
            if (!this._config_ctrl_elem) {
                this._config_ctrl_elem = this._config_panel().children('#ctrl') ;
            }
            return this._config_ctrl_elem ;
        } ;
        this._config_set_updated = function (html) {
            if (!this._config_updated_elem) {
                this._config_updated_elem = this._config_ctrl().find('#updated') ;
            }
            this._config_updated_elem.html(html) ;
        } ;
        this._config_button_edit = function () {
            if (!this._config_button_edit_elem) {
                this._config_button_edit_elem = this._config_ctrl().find('button[name="edit"]').button() ;
            }
            return this._config_button_edit_elem ;
        } ;
        this._config_button_save = function () {
            if (!this._config_button_save_elem) {
                this._config_button_save_elem = this._config_ctrl().find('button[name="save"]').button() ;
            }
            return this._config_button_save_elem ;
        } ;
         this._config_button_cancel = function () {
            if (!this._config_button_cancel_elem) {
                this._config_button_cancel_elem = this._config_ctrl().find('button[name="cancel"]').button() ;
            }
            return this._config_button_cancel_elem ;
        } ;
         this._config_button_update = function () {
            if (!this._config_button_update_elem) {
                this._config_button_update_elem = this._config_ctrl().find('button[name="update"]').button() ;
            }
            return this._config_button_update_elem ;
        } ;
        this._config_body = function () {
            if (!this._config_body_elem) {
                this._config_body_elem = this._config_panel().children('#body') ;
            }
            return this._config_body_elem ;
        } ;
        this._config_editor = function () {
            if (!this._config_editor_obj) {
                var propdef = [{

                    group:     'STANDARD OPTIONS'} , {

                    name:      'release_dir' ,
                    text:      'Release directory' ,
                    title:     'An absolute path to a release directory from which \n' +
                               'to run the Translator application.' ,
                    edit_mode: true, editor: 'text', edit_size: 64} , {

                    name:      'config_file' ,
                    text:      'Configuration file' ,
                    title:     'An absolute or a relative path to a psana configuration file \n' +
                               'which will be used by the Translator. Note that a relative path \n' +
                               'will be atatched to the release directory' ,
                    edit_mode: true, editor: 'text', edit_size: 64} , {

                    name:      'auto' ,
                    text:      'Enable Auto-Translation' ,
                    title:     'If this mode is turned on then the translations ervice will automatically \n' +
                               'detect new runs and initiate the translation for teh runs. \n' +
                               'to run the Translator application.' ,
                    edit_mode: true, editor: 'checkbox'} , {

                    name:      'ffb' ,
                    text:      'Input from FFB' ,
                    title:     'The FFB mode allows to begin translating runs while the data \n' +
                               'are still located on the FFB file system of the corresponding instrument. \n' +
                               'Usually this would result in a shorted ratency before one could see \n' +
                               'the HDF5 files. Note that one possible donwside of teh mode is that it may \n' +
                               'potentially put too much load onto the FFB file system which may negatively \n' +
                               'affect data migration processes.' ,
                    edit_mode: true, editor: 'checkbox'} , {

                    name:      'stream' ,
                        text:      'DAQ stream filter' ,
                        title:     'An optional filter of the DAQ streams which should be \n' +
                                   'selected for the translation. This parameter may be needed to \n' +
                                   'bypass streams (like IOC s80, s81, etc.) which are delayed \n' +
                                   'to moved to the input file system (FFB or ANA). Here are a few \n' +
                                   'examples of the valid value of the parameter: \n' +
                                   '  0-79 \n' +
                                   '  0,3-4,80,81 \n' +
                                   'Clear the input to turn this filter off.' +
                                   'Note that using this filter will results in not having the corresponding \n' +
                                   'data carried by non-mentioned streams in the generated HDF5 files.' ,
                        edit_mode: true, editor: 'text', edit_size: 32}
                ] ;
                switch (this.service) {
                    case 'Monitoring':
                        propdef.push({

                            name:      'outdir' ,
                            text:      'Output directory' ,
                            title:     'An absolute or a relative path to a folder where to place \n' +
                                       'translated HDF5 files. Note that a relative path will be attached \n' +
                                       'to the base data directory of the experiment' ,
                            edit_mode: true, editor: 'text', edit_size: 64} , {

                            name:      'ccinsubdir' ,
                            text:      'CC files in a subdir' ,
                            title:     'Place Calib Cycle files at a separate subfolder of the output directory' ,
                            edit_mode: true, editor: 'checkbox'} , {

                            group:     'ADVANCED OPTIONS'} , {

                            name:      'exclusive' ,
                            text:      'Exclusive use of nodes (MPI)' ,
                            title:     'Request that each of the experiments runs will be translated \n' +
                                       'with exclusive use of its nodes (batch system -x option). May help \n' +
                                       'achieve fastest per-run translation at the expense of overall queue \n' +
                                       'utilization. For example, translation of subsequent runs may start \n' +
                                       'later while waiting for sufficient system resources. \n' +
                                       'RESTRICTIONS: this parameter will only be used in the FFB mode.' ,
                            edit_mode: true, editor: 'checkbox'} , {

                            name:      'njobs' ,
                            text:      '# parallel processes (MPI)' ,
                            title:     'The number of paralell MPI processes used by the translation service. \n' +
                                       'Remove any number from the cell to assume the default value of \n' +
                                       'the parameter.' ,
                            edit_mode: true, editor: 'text',    edit_size: 6} , {

                            name:      'ptile' ,
                            text:      '# processes per node (MPI)' ,
                            title:     'The maximum number of paralell MPI processes per node. \n' +
                                       'Smaller values (1,2,3) may help achieve fastest per-run translation, but \n' +
                                       'at the expense of overall queue utilization. For example, translation of \n' +
                                       'subsequent runs may start later while waiting for sufficient system resources. \n' +
                                       'Remove any number from the cell to assume the default value of \n' +
                                       'the parameter. \n' +
                                       'RESTRICTIONS: this parameter will only be used in the FFB mode.' ,
                            edit_mode: true, editor: 'text',    edit_size: 6} , {

                            name:      'livetimeout' ,
                            text:      'Live Mode Timeout [s]' ,
                            title:     'The number of seconds to wait for migrating files in the live \n' +
                                       'mode before to give up and abort the translation. \n' +
                                       'Remove any number from the cell to assume the default value of \n' +
                                       'the parameter.' ,
                            edit_mode: true, editor: 'text', edit_size: 6}
                        ) ;
                        break ;
                }
                this._config_editor_obj = new PropList(propdef) ;
                this._config_editor_obj.display(this._config_body().children('#editor')) ;
            }
            return this._config_editor_obj ;
        } ;

        /**
         * Returns and (optionally) set the editing mode for the configuration
         * options. Changes control buttons accordingly.
         *
         * @param {Boolean} editing
         * @returns {Boolean}
         */
        this._config_edit_mode = function (editing) {
            if (this._config_editing === undefined) this._config_editing = false ;
            if (editing !== undefined) {
                if (this._can_manage()) {
                    this._config_button_update().button(editing ? 'disable' : 'enable') ;
                    this._config_button_edit  ().button(editing ? 'disable' : 'enable') ;
                    this._config_button_save  ().button(editing ? 'enable'  : 'disable') ;
                    this._config_button_cancel().button(editing ? 'enable'  : 'disable');
                    this._config_editing = editing ;
                } else {
                    this._config_button_update().button('enable') ;
                    this._config_button_edit  ().button('disable') ;
                    this._config_button_save  ().button('disable') ;
                    this._config_button_cancel().button('disable');
                }
            }
            return this._editing ;
        } ;


        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // Just in case

            switch (this.service) {
                case 'Standard':   break ;
                case 'Monitoring': break ;
                default:
                    this._wa('This application is broken. Please, report this problem to the support team.') ;
                    console.log('HDF5_Translator._init(): unsupported translation service: '+this.service);
                    return ;
            } ;

            if (!this.access_list.hdf5.read) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }

            /* ----------------------------------------------------------------------
             *   Initialize UI and set up event handlers for the status of requests
             * ----------------------------------------------------------------------
             */

            this._status_button_reverse().click(function () {
                _that._status_reverse_order = !_that._status_reverse_order ;
                if (_that._status_last_request) _that._status_last_request.requests.reverse() ;
                _that._status_display() ;
            }) ;
            this._status_ctrl            ().find('.update-trigger').change(function () { _that._status_load() ; }) ;
            this._status_button_reset    ()                        .click (function () { _that._status_reset() ; }) ;
            this._status_button_search   ()                        .click (function () { _that._status_load() ; }) ;
            this._status_button_translate()                        .click (function () { _that._status_translate_all() ; }) ;
            this._status_button_stop     ()                        .click (function () { _that._status_stop_all() ; }) ;

            if (!this.access_list.hdf5.manage) {
                this._status_button_translate().button('disable') ;
                this._status_button_stop     ().button('disable') ;
            }        
            this._status_load() ;

            /* -------------------------------------------------------------------------
             *   Initialize UI and set up event handlers for the configuration options
             * -------------------------------------------------------------------------
             */
            this._config_button_update().click(function () { _that._config_load() ; }) ;
            this._config_button_edit  ().click(function () { _that._config_edit() ; }) ;
            this._config_button_save  ().click(function () { _that._config_edit_save() ; }) ;
            this._config_button_cancel().click(function () { _that._config_edit_cancel() ; }) ;


            this._config_editor().set_value('release_dir', 'Loading...') ;
            this._config_editor().set_value('config_file', 'Loading...') ;
            this._config_editor().set_value('auto',        'Loading...') ;
            this._config_editor().set_value('ffb',         'Loading...') ;
            this._config_editor().set_value('stream',      'Loading...') ;
            switch (this.service) {
                case 'Standard':
                    break ;
                case 'Monitoring':
                    this._config_editor().set_value('outdir',      'Loading...') ;
                    this._config_editor().set_value('ccinsubdir',  'Loading...') ;
                    this._config_editor().set_value('exclusive',   'Loading...') ;
                    this._config_editor().set_value('njobs',       'Loading...') ;
                    this._config_editor().set_value('ptile',       'Loading...') ;
                    this._config_editor().set_value('livetimeout', 'Loading...') ;
                    break ;
            }

            this._config_edit_mode(false) ;
            this._config_load() ;
        } ;

        this._status_reset = function () {
            this._status_runs_selector ().val('') ;
            this._status_state_selector().val('any') ;
            this._status_load() ;
        } ;

        function comparator (a, b) {
            if (typeof a.state.run_number !== 'number') a.state.run_number = parseInt(a.state.run_number) ;
            if (typeof b.state.run_number !== 'number') b.state.run_number = parseInt(b.state.run_number) ;
            if (a.state.run_number < b.state.run_number) return -1 ;
            if (a.state.run_number > b.state.run_number) return  1 ;
            return 0 ;
        }
        this._status_load = function () {

            var params = {
                service: this.service ,
                exper_id: this.experiment.id ,
                show_files: '' ,
                json: ''
            } ;
            var runs  = this._status_runs_selector ().val() ; if (runs)            params.runs   = runs ;
            var state = this._status_state_selector().val() ; if (state !== 'any') params.status = state ;

            this._status_set_updated('Updating...') ;

            Fwk.web_service_GET (
                '../portal/ws/hdf5_requests_get.php' ,
                params ,
                function (data) {

                    _that._status_last_request = data ;
                    _that._status_last_request.requests.sort(comparator) ;  // to guarantee the acsending order

                    if (!_that._status_reverse_order) _that._status_last_request.requests.reverse() ;

                    _that._status_set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._status_display() ;
                }
            ) ;
        } ;
        this._status_display = function () {
            var html =
'<table class="requests" border="0" cellspacing="0" cellpadding="0" >' +
'  <thead>' +
'    <tr align="left" >' +
'      <td >Run</td>' +
'      <td >End of Run</td>' +
'      <td >State</td>' +
'      <td >Last Change</td>' +
'      <td >Log File</td>' +
'      <td >Priority</td>' +
'      <td >Actions</td>' +
'      <td >Comments</td>' +
'    </tr>' +
'  </thead>' +
'  <tbody>' ;

            var summary = {
                'FINISHED'       : 0 ,
                'FAILED'         : 0 ,
                'TRANSLATING'    : 0 ,
                'QUEUED'         : 0 ,
                'NOT-TRANSLATED' : 0
            } ;
            for (var i in this._status_last_request.requests) {
                var request = this._status_last_request.requests[i] ;
                var state = request.state ;
                summary[state.status]++ ;
                var run_url = '<a class="link" href="javascript:global_elog_search_run_by_num('+state.run_number+',true)" title="click to see a LogBook record for this run" >'+state.run_number+'</a>' ;
                var log_url = state.log_available ? '<a class="link" href="translate/'+this.service+'/'+state.id+'/'+state.id+'.log" target="_blank" title="click to see the log file for the last translation attempt">log</a>' : '' ;
                var state_color = 'black' ;
                switch (state.status) {
                    case 'FAILED'         : state_color = 'red'   ; break ;
                    case 'NOT-TRANSLATED' : state_color = state.actions ? 'green' : '#b0b0b0' ; break ;
                }
                var decorated_state = '<span style="font-weight:normal; color:'+state_color+';">'+state.status+'</span>' ;
                html +=
'  <tr class="run-header" id="'+state.run_number+'" >' +
'    <td >'                   +run_url+         '</td>' +
'    <td >'                   +state.end_of_run+'</td>' +
'    <td >'                   +decorated_state   +'</td>' +
'    <td >'                   +state.changed      +'</td>' +
'    <td >'                   +log_url            +'</td>' +
'    <td class="priority" >'  +state.priority     +'</td>' +
'    <td >&nbsp;'             +state.actions      +'</td>' +
'    <td class="comment"  >'  +state.comments     +'&nbsp;</td>' +
'  </tr>' ;
            } ;
            html +=
'  </tbody>' +
'</table>' ;
            this._status_viewer().html(html) ;
            this._status_viewer().find('.control-button')
                .button()
                .button(!this.access_list.hdf5.manage ? 'disable' : 'enable')
                .click(function () {
                    var val = parseInt($(this).attr('value')) ;
                    switch (this.name) {
                        case 'translate' :
                            var runnum = val ;
                            _that._status_translate(runnum, $(this)) ;
                            break ;
                        case 'escalate' :
                            var run_icws_id = val ;
                            _that._status_escalate(run_icws_id) ;
                            break ;
                        case 'stop' :
                            var run_icws_id = val ;
                            _that._status_stop(run_icws_id) ;
                            break ;
                    }
                }) ;
            if (!this.access_list.hdf5.is_data_administrator) {
                this._status_viewer().find('.control-button.retranslate').button('disable') ;
            }
            var summary_html = '' ;
            for (var status in summary) {
                var counter = summary[status] ;
                if (counter) {
                    if (summary_html) summary_html += ', ' ;
                    summary_html += status+': <b>'+counter+'</b>' ;
                }
            }
            this._status_set_info('<b>'+this._status_last_request.requests.length+'</b> runs [ '+summary_html+' ]') ;
        } ;
    
        /**
         * Translate all eligible runs shown in the table.
         *
         * @returns {undefined}
         */
        this._status_translate_all = function () {

            var num_runs = 0;
            for (var i in this._status_last_request.requests)
                if (this._status_last_request.requests[i].state.ready4translation)
                    ++num_runs ;

            Fwk.ask_yes_no (
                'Confirm HDF5 Translation Request' ,
                'You are about to request HDF5 translaton of <b>'+num_runs+'</b> runs. ' +
                'This may take a while. Are you sure you want to proceed with this operation?' ,
                function() {
                    _that._status_viewer().find('button[name="translate"]').each(function () {
                        var runnum = parseInt($(this).val()) ;
                        _that._status_translate(runnum, $(this)) ;
                    }) ;
                }
            );
        } ;
        this._status_translate = function (runnum, button_translate) {

            var tr = this._status_viewer().find('tr.run-header#'+runnum) ;
            var comment  = tr.find('td.comment') ;

            comment.html('<span style="color:red;">Processing...</span>') ;

            button_translate.button('disable') ;

            Fwk.web_service_GET (
                '../portal/ws/hdf5_request_new.php' ,
                {   service: this.service ,
                    exper_id: this.experiment.id ,
                    runnum: runnum
                } ,
                function () {
                    comment.html('<span style="color:green;">Translation request was queued</span>') ;
                } ,
                function (msg) {
                    button_translate.button('enable') ;
                    comment.html('<span style="color:red;">Translation request was rejected</span>') ;
                    Fwk.report_error(msg) ;
                }
            );
        } ;

        /**
         * Withdraw all queued requests show in the table.
         *
         * @returns {undefined}
         */
        this._status_stop_all = function () {

            var num_runs = 0;
            for (var i in this._status_last_request.requests)
                if (this._status_last_request.requests[i].state.status === 'QUEUED')
                    ++num_runs ;

            Fwk.ask_yes_no (
                'Confirm HDF5 Translation Request Withdrawal' ,
                'You are about to withdraw HDF5 translaton requests for <b>'+num_runs+'</b> sitting in the translation queue. ' +
                'This may take a while. Are you sure you want to proceed with this operation?' ,
                function() {
                    _that._status_viewer().find('button[name="stop"]').each(function () {
                        $(this).button('disable') ;
                        var run_icws_id = parseInt($(this).val()) ;
                        _that._status_stop(run_icws_id) ;
                    }) ;
                }
            );
        } ;
        this._status_stop = function (run_icws_id) {

            var runnum = 0 ;
            for (var i in this._status_last_request.requests) {
                var state = this._status_last_request.requests[i].state ;
                if (state.id === run_icws_id) {
                    runnum = state.run_number ;
                    break ;
                }
            }
            if (!runnum) {
                Fwk.report_error('internal error: no run number found for request id: '+run_icws_id) ;
                return ;
            }
            var tr = this._status_viewer().find('tr.run-header#'+runnum) ;
            var comment  = tr.find('td.comment') ;

            comment.html('<span style="color:red;">Processing...</span>') ;

            Fwk.web_service_GET (
                '../portal/ws/hdf5_request_delete.php' ,
                {   service: this.service ,
                    id: run_icws_id
                } ,
                function (data) {
                    comment.html('<span style="color:green;">Translation was stopped</span>') ;
                } ,
                function (msg) {
                    comment.html('<span style="color:red;">Failed</span>') ;
                    Fwk.report_error(msg) ;
                }
            );
        } ;

        /**
         * Escalate the priority of a queued request
         *
         * @param {number} run_icws_id - an identifier of the request
         * @returns {undefined}
         */
        this._status_escalate = function (run_icws_id) {

            var runnum = 0 ;
            for (var i in this._status_last_request.requests) {
                var state = this._status_last_request.requests[i].state ;
                if (state.id === run_icws_id) {
                    runnum = state.run_number ;
                    break ;
                }
            }
            if (!runnum) {
                Fwk.report_error('internal error: no run number found for request id: '+run_icws_id) ;
                return ;
            }

            var tr = this._status_viewer().find('tr.run-header#'+runnum) ;
            var comment  = tr.find('td.comment') ;
            var priority = tr.find('td.priority') ;

            comment.html('<span style="color:red;">Processing...</span>') ;

            Fwk.web_service_GET (
                '../portal/ws/hdf5_request_escalate.php' ,
                {   service: this.service ,
                    exper_id: this.experiment.id ,
                    id: run_icws_id } ,
                function (data) {
                    comment.html('') ;
                    priority.html(data.Priority) ;
                } ,
                function (msg) {
                    comment.html('<span style="color:red;">Failed</span>') ;
                    Fwk.report_error(msg) ;
                }
            );
        } ;

        this._config_load = function () {

            if (!this._config_edit_mode(false)) {
                var params = {
                    service: this.service ,
                    exper_id: this.experiment.id
                } ;
                this._config_set_updated('Updating...') ;

                Fwk.web_service_GET (
                    '../portal/ws/hdf5_config_get.php' ,
                    params ,
                    function (data) {
                        _that._config = data.config ;
                        _that._config_set_updated('Updated: <b>'+data.updated+'</b>') ;
                        _that._config_display() ;
                    }
                ) ;
            }
        } ;
        this._config_edit = function () {
            this._config_edit_mode(true) ;
            this._config_editor().edit_value('release_dir') ;
            this._config_editor().edit_value('config_file') ;
            this._config_editor().edit_value('auto') ;
            this._config_editor().edit_value('ffb') ;
            this._config_editor().edit_value('stream') ;
            switch (this.service) {
                case 'Standard':
                    break ;
                case 'Monitoring':
                    this._config_editor().edit_value('outdir') ;
                    this._config_editor().edit_value('ccinsubdir') ;
                    this._config_editor().edit_value('exclusive') ;
                    this._config_editor().edit_value('njobs') ;
                    this._config_editor().edit_value('ptile') ;
                    this._config_editor().edit_value('livetimeout') ;
                    break ;
            } ;
        } ;
        this._config_edit_save = function () {
            this._config_edit_mode(false) ;
            var params = {
                service:     this.service ,
                exper_id:    this.experiment.id ,
                release_dir: this._config_editor().get_value('release_dir') ,
                config_file: this._config_editor().get_value('config_file') ,
                auto:        this._config_editor().get_value('auto') ? 1 : 0 ,
                ffb:         this._config_editor().get_value('ffb')  ? 1 : 0 ,
                stream:      this._config_editor().get_value('stream')
            } ;
            switch (this.service) {
                case 'Monitoring':
                    params.outdir      = this._config_editor().get_value('outdir') ;
                    params.ccinsubdir  = this._config_editor().get_value('ccinsubdir') ? 1 : 0 ;
                    params.exclusive   = this._config_editor().get_value('exclusive') ? 1 : 0 ;
                    params.njobs       = this._config_editor().get_value('njobs') ;
                    params.ptile       = this._config_editor().get_value('ptile') ;
                    params.livetimeout = this._config_editor().get_value('livetimeout') ;
                    break ;
            } ;
            this._config_set_updated('Saving...') ;

            Fwk.web_service_POST (
                '../portal/ws/hdf5_config_set.php' ,
                params ,
                function (data) {
                    _that._config = data.config ;
                    _that._config_set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._config_editor().view_value('release_dir') ;
                    _that._config_editor().view_value('config_file') ;
                    _that._config_editor().view_value('auto') ;
                    _that._config_editor().view_value('ffb') ;
                    _that._config_editor().view_value('stream') ;
                    switch (_that.service) {
                        case 'Monitoring':
                            _that._config_editor().view_value('outdir') ;
                            _that._config_editor().view_value('ccinsubdir') ;
                            _that._config_editor().view_value('exclusive') ;
                            _that._config_editor().view_value('njobs') ;
                            _that._config_editor().view_value('ptile') ;
                            _that._config_editor().view_value('livetimeout') ;
                            break;
                    }
                    _that._config_display() ;
                }
            ) ;
        } ;
        this._config_edit_cancel = function () {
            this._config_edit_mode(false) ;
            this._config_editor().view_value('release_dir') ;
            this._config_editor().view_value('config_file') ;
            this._config_editor().view_value('auto') ;
            this._config_editor().view_value('ffb') ;
            this._config_editor().view_value('stream') ;
            switch (this.service) {
                case 'Monitoring':
                    this._config_editor().view_value('outdir') ;
                    this._config_editor().view_value('ccinsubdir') ;
                    this._config_editor().view_value('exclusive') ;
                    this._config_editor().view_value('njobs') ;
                    this._config_editor().view_value('ptile') ;
                    this._config_editor().view_value('livetimeout') ;
                    break ;
            } ;
            this._config_display() ;
        } ;
        this._config_display = function () {
            this._config_editor().set_value('release_dir', this._config.release_dir) ;
            this._config_editor().set_value('config_file', this._config.config_file) ;
            this._config_editor().set_value('auto',        this._config.auto) ;
            this._config_editor().set_value('ffb',         this._config.ffb) ;
            this._config_editor().set_value('stream',      this._config.stream) ;
            switch (this.service) {
                case 'Monitoring':
                    this._config_editor().set_value('outdir',      this._config.outdir) ;
                    this._config_editor().set_value('ccinsubdir',  this._config.ccinsubdir) ;
                    this._config_editor().set_value('exclusive',   this._config.exclusive) ;
                    this._config_editor().set_value('njobs',       this._config.njobs) ;
                    this._config_editor().set_value('ptile',       this._config.ptile) ;
                    this._config_editor().set_value('livetimeout', this._config.livetimeout) ;
                    break ;
            } ;
        } ;
    }
    Class.define_class (HDF5_Translator, FwkApplication, {}, {}) ;

    return HDF5_Translator ;
}) ;
