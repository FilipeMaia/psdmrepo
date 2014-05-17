define ([
    'webfwk/CSSLoader'
] ,

function (cssloader) {

    cssloader.load('../portal/css/HDF5_Manage.css') ;

    /**
     * The application for displaying and managing HDF5 translation requests of the experiment
     *
     * @returns {HDF5_Manage}
     */
    function HDF5_Manage (experiment, access_list) {

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
                    this._load() ;
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
        this._runs   = null ;
        this._status = null ;
        this._info    = null ;
        this._updated = null ;
        this._body_ctrl = null ;
        this._reverse_order = false ;
        this._hide_xtc = null ;
        this._auto = null ;
        this._viewer = null ;

        this._last_request = [] ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;
            this.container.html('<div id="hdf5-manage"></div>') ;
            this._wa = this.container.find('div#hdf5-manage') ;

            if (!this.access_list.hdf5.read) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }
            var html =
'<div id="ctrl">' +
'  <div class="group" title=' +
'"Put a run number or a range of runs to activate the filter. \n' +
'Range can be specified like this: \n' +
'  10 \n' +
'  10-20 \n' +
'  10- \n' +
'  -20 \n' +
'Note that the range cane be open on the either end. \n' +
'Press RETURN to activate search." >' +
'    <span class="label">Search runs:</span>' +
'    <input class="update-trigger" type="text" name="runs" value="" />' +
'  </div>' +
'  <div class="group" title="Select non-blank option to activate the filter">' +
'    <span class="label">Translation status:</span>' +
'    <select class="update-trigger" name="status">' +
'      <option>any</option>' +
'      <option>FINISHED</option>' +
'      <option>FAILED</option>' +
'      <option>TRANSLATING</option>' +
'      <option>QUEUED</option>' +
'      <option>NOT-TRANSLATED</option>' +
'    </select>' +
'  </div>' +
'  <div class="buttons" style="float:left;" >' +
'    <button class="control-button" name="reset"   title="reset the form">RESET FORM</button>' +
'    <button class="control-button" name="refresh" title="click to refresh the list of files">SEARCH</button>' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div id="body" >' +
'  <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div id="ctrl" >' +
'    <div>' +
'      <button class="control-button" name="reverse"   >SHOW IN REVERSE ORDER</button>' +
'      <span class="label">hide XTC files</span><input type="checkbox" name="hide" />' +
'    </div>' +
'    <div>' +
'      <button class="control-button" name="translate" >TRANSLATE SELECTED RUNS</button>' +
'      <button class="control-button" name="stop"      >STOP TRANSLATION OF SELECTED RUNS</button>' +
'      <span class="label">enable auto-translation</span><input type="checkbox" name="auto" />' +
'    </div>' +
'  </div>' +
'  <div id="viewer"></div>' +
'</div>' ;
            this._wa.html(html) ;

            var ctrl = this._wa.find('#ctrl') ;

            this._runs   = ctrl.find('input[name="runs"]') ;
            this._status = ctrl.find('select[name="status"]') ;

            ctrl.find('.update-trigger').change(function () { _that._load() ; }) ;
            ctrl.find('.control-button').button().click(function () {
                var op = this.name ;
                console.log('operation: '+op) ;
                switch (op) {
                    case 'reset'   : _that._reset() ; break ;
                    case 'refresh' : _that._load() ; break ;
                }
            }) ;

            var body = this._wa.find('#body') ;

            this._info    = body.find('#info') ;
            this._updated = body.find('#updated') ;

            this._body_ctrl = body.find('#ctrl') ;
            this._body_ctrl.find('button.control-button').button().click(function () {
                var op = this.name ;
                switch (op) {
                    case 'reverse' :
                        _that._reverse_order = !_that._reverse_order ;
                        if (_that._last_request) _that._last_request.requests.reverse() ;
                        _that._display() ;
                        break ;
                    case 'translate' :
                        _that._translate_all() ;
                        break ;
                    case 'stop' :
                        _that._stop_all() ;
                        break ;
                }
            }) ;
            this._auto = this._body_ctrl.find('input[name="auto"]') ;
            this._auto.change(function () {
                var on = $(this).attr('checked') ? true : false ;
                _that._auto_translation(on) ;
            }) ;
            this._hide_xtc = this._body_ctrl.find('input[name="hide"]') ;
            this._hide_xtc.change(function () {
                _that._display() ;
            }) ;

            if (!this.access_list.hdf5.manage) {
                this._body_ctrl.find('button.control-button[name="translate"]').button('disable') ;
                this._body_ctrl.find('button.control-button[name="stop"]').button('disable') ;
                this._auto.attr('disabled', 'disabled') ;
            }        

            this._viewer = body.find('#viewer') ;

            this._load() ;
        } ;

        this._reset = function () {
            this._runs.val('') ;
            this._status.val('any') ;
            this._load() ;
        } ;

        function comparator (a, b) {
            if (typeof a.state.run_number !== 'number') a.state.run_number = parseInt(a.state.run_number) ;
            if (typeof b.state.run_number !== 'number') b.state.run_number = parseInt(b.state.run_number) ;
            if (a.state.run_number < b.state.run_number) return -1 ;
            if (a.state.run_number > b.state.run_number) return  1 ;
            return 0 ;
        }
        this._load = function () {

            var params = {
                exper_id: this.experiment.id ,
                show_files: '' ,
                json: ''
            } ;
            var runs   = this._runs  .val() ; if (runs)             params.runs   = runs ;
            var status = this._status.val() ; if (status !== 'any') params.status = status ;

            this._updated.html('Updating...') ;

            Fwk.web_service_GET (
                '../portal/ws/SearchRequests.php' ,
                params ,
                function (data) {

                    _that._last_request = data ;
                    _that._last_request.requests.sort(comparator) ;  // to guarantee the acsending order

                    if (!_that._reverse_order) _that._last_request.requests.reverse() ;

                    _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display() ;
                }
            ) ;
        } ;
        this._display = function () {
            var hide_xtc = this._hide_xtc.attr('checked') ? true : false ;
            if (this._last_request.autotranslate2hdf5) this._auto.attr('checked', 'checked') ;
            var html =
'<table class="requests" border="0" cellspacing="0" cellpadding="0" >' +
'  <thead>' +
'    <tr align="left" >' +
'      <td > Run         </td>' +
'      <td > End of Run  </td>' +
'      <td > File        </td>' +
'      <td > Size        </td>' +
'      <td > Status      </td>' +
'      <td > Changed     </td>' +
'      <td > Log         </td>' +
'      <td > Priority    </td>' +
'      <td > Actions     </td>' +
'      <td > Comments    </td>' +
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
            for (var i in this._last_request.requests) {
                var request = this._last_request.requests[i] ;
                var state = request.state ;
                summary[state.status]++ ;
                var files_xtc = { type: 'xtc',  files: hide_xtc ? [] : request.xtc  } ;
                var files_hdf = { type: 'hdf5', files: request.hdf5 } ;
                var run_url = '<a class="link" href="javascript:global_elog_search_run_by_num('+state.run_number+',true)" title="click to see a LogBook record for this run" >'+state.run_number+'</a>' ;
                var log_url = state.log_available ? '<a class="link" href="translate/'+state.id+'/'+state.id+'.log" target="_blank" title="click to see the log file for the last translation attempt">log</a>' : '' ;
                var status_color = 'black' ;
                switch (state.status) {
                    case 'FAILED'         : status_color = 'red'   ; break ;
                    case 'NOT-TRANSLATED' : status_color = state.actions ? 'green' : '#b0b0b0' ; break ;
                }
                var decorated_status = '<span style="font-weight:bold; color:'+status_color+';">'+state.status+'</span>' ;
                html +=
'  <tr class="run-header" id="'+state.run_number+'" >'+
'    <td >'                 +run_url+         '</td>'+
'    <td >'                 +state.end_of_run+'</td>'+
'    <td >&nbsp;</td>'+
'    <td >&nbsp;</td>'+
'    <td >'                 +decorated_status   +'</td>' +
'    <td >'                 +state.changed      +'</td>' +
'    <td >'                 +log_url            +'</td>' +
'    <td class="priority" >'+state.priority     +'</td>' +
'    <td >&nbsp;'           +state.actions      +'</td>' +
'    <td class="comment"  >'+state.comments     +'&nbsp;</td>' +
'  </tr>' ;

                var first_of_a_kind = {} ;
                var collections = [files_hdf, files_xtc] ;
                for(var j in collections ) {
                    var type = collections[j].type ;
                    var files = collections[j].files ;
                    for (var k in files) {
                        var file = files[k] ;
                        var classes = hide_xtc ? '' : type ;
                        if (first_of_a_kind[type] === undefined) {
                            first_of_a_kind[type] = true ;
                            classes += ' first-in-class' ;
                        }
                        html +=
'  <tr class="'+classes+'" >' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td ><span '+(type==='hdf5'?'style="font-weight:bold;"':'')+'>'+file.name+'</span></td>' +
'    <td >'+file.size+'</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'  </tr>' ;
                    }
                }
            } ;
            html +=
'  </tbody>' +
'</table>' ;
            this._viewer.html(html) ;
            this._viewer.find('.control-button')
            .button()
            .button(!this.access_list.hdf5.manage ? 'disable' : 'enable')
            .click(function () {
                var val = parseInt($(this).attr('value')) ;
                switch (this.name) {
                    case 'translate' :
                        var runnum = val ;
                        _that._translate(runnum) ;
                        break ;
                    case 'escalate' :
                        var run_icws_id = val ;
                        _that._escalate(run_icws_id) ;
                        break ;
                    case 'stop' :
                        var run_icws_id = val ;
                        _that._stop(run_icws_id) ;
                        break ;
                }
            }) ;
            if (!this.access_list.hdf5.is_data_administrator) {
                this._viewer.find('.control-button.retranslate').button('disable') ;
            }
            var summary_html = '' ;
            for (var status in summary) {
                var counter = summary[status] ;
                if (counter) {
                    if (summary_html) summary_html += ', ' ;
                    summary_html += status+': <b>'+counter+'</b>' ;
                }
            }
            this._info.html(
'<b>'+this._last_request.requests.length+'</b> runs [ '+summary_html+' ]'
            ) ;
        } ;
    
        /**
         * Translate all eligible runs shown in the table.
         *
         * @returns {undefined}
         */
        this._translate_all = function () {

            var num_runs = 0;
            for (var i in this._last_request.requests)
                if (this._last_request.requests[i].state.ready4translation)
                    ++num_runs ;

            Fwk.ask_yes_no (
                'Confirm HDF5 Translation Request' ,
                'You are about to request HDF5 translaton of <b>'+num_runs+'</b> runs. ' +
                'This may take a while. Are you sure you want to proceed with this operation?' ,
                function() {
                    _that._viewer.find('button[name="translate"]').each(function () {
                        $(this).button('disable') ;
                        var runnum = parseInt($(this).val()) ;
                        _that._translate(runnum) ;
                    }) ;
                }
            );
        } ;
        this._translate = function (runnum) {

            var tr = this._viewer.find('tr.run-header#'+runnum) ;
            var comment  = tr.find('td.comment') ;

            comment.html('<span style="color:red;">Processing...</span>') ;

            Fwk.web_service_GET (
                '../portal/ws/NewRequest.php' ,
                {   exper_id: this.experiment.id ,
                    runnum: runnum
                } ,
                function (data) {
                    comment.html('<span style="color:green;">Translation request was queued</span>') ;
                } ,
                function (msg) {
                    comment.html('<span style="color:red;">Translation request was rejected</span>') ;
                    Fwk.report_error(msg) ;
                }
            );
        } ;

        /**
         * Withdraw all queued requests show in teh table.
         *
         * @returns {undefined}
         */
        this._stop_all = function () {

            var num_runs = 0;
            for (var i in this._last_request.requests)
                if (this._last_request.requests[i].state.status === 'QUEUED')
                    ++num_runs ;

            Fwk.ask_yes_no (
                'Confirm HDF5 Translation Request Withdrawal' ,
                'You are about to withdraw HDF5 translaton requests for <b>'+num_runs+'</b> sitting in the translation queue. ' +
                'This may take a while. Are you sure you want to proceed with this operation?' ,
                function() {
                    _that._viewer.find('button[name="stop"]').each(function () {
                        $(this).button('disable') ;
                        var run_icws_id = parseInt($(this).val()) ;
                        _that._stop(run_icws_id) ;
                    }) ;
                }
            );
        } ;
        this._stop = function (run_icws_id) {

            var runnum = 0 ;
            for (var i in this._last_request.requests) {
                var state = this._last_request.requests[i].state ;
                if (state.id === run_icws_id) {
                    runnum = state.run_number ;
                    break ;
                }
            }
            if (!runnum) {
                Fwk.report_error('internal error: no run number found for request id: '+run_icws_id) ;
                return ;
            }
            var tr = this._viewer.find('tr.run-header#'+runnum) ;
            var comment  = tr.find('td.comment') ;

            comment.html('<span style="color:red;">Processing...</span>') ;

            Fwk.web_service_GET (
                '../portal/ws/DeleteRequest.php' ,
                {   id: run_icws_id
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
        this._escalate = function (run_icws_id) {

            var runnum = 0 ;
            for (var i in this._last_request.requests) {
                var state = this._last_request.requests[i].state ;
                if (state.id === run_icws_id) {
                    runnum = state.run_number ;
                    break ;
                }
            }
            if (!runnum) {
                Fwk.report_error('internal error: no run number found for request id: '+run_icws_id) ;
                return ;
            }

            var tr = this._viewer.find('tr.run-header#'+runnum) ;
            var comment  = tr.find('td.comment') ;
            var priority = tr.find('td.priority') ;

            comment.html('<span style="color:red;">Processing...</span>') ;

            Fwk.web_service_GET (
                '../portal/ws/EscalateRequestPriority.php' ,
                {   exper_id: this.experiment.id ,
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
        this._auto_translation = function(on) {
            Fwk.ask_yes_no (
                'Confirmn HDF5 Translation Request' ,
                on ? 'You are about to request automatic HDF5 translaton of all (past and future) runs of the experiment. ' +
                     'This may take substantial resources (CPU and disk storage). Are you sure you want to proceed with this operation?'
                   : 'You are about to stop automatic HDF5 translaton of all (past and future) runs of the experiment. ' +
                     'Note that this may potentially affect all members of the experiment. Are you sure you want to proceed with this operation?' ,
                function () {
                    Fwk.web_service_POST (
                        '../regdb/ws/SetAutoTranslate2HDF5.php' ,
                        {   exper_id: this.experiment.id ,
                            autotranslate2hdf5: on ? 1 : 0 } ,
                        function ()     {} ,
                        function (msg)  {
                            _that._revert_auto_translation(on) ;
                            Fwk.report_error(msg) ;
                        }
                    ) ;
                } ,
                function () { _that._revert_auto_translation(on) ; }
            ) ;
        } ;
        this._revert_auto_translation = function (on) {
            if (on) this._auto.removeAttr('checked') ;
            else    this._auto.attr('checked', 'checked') ;
        } ;
    }
    define_class (HDF5_Manage, FwkApplication, {}, {}) ;

    return HDF5_Manage ;
}) ;
