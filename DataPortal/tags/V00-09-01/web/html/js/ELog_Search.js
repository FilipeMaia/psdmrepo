define ([
    'webfwk/CSSLoader',          'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk' ,
    'portal/ELog_MessageViewer', 'portal/ELog_Utils'] ,

function (
    cssloader,          Class, FwkApplication, Fwk ,
    ELog_MessageViewer, ELog_Utils) {

    cssloader.load('../portal/css/ELog_Search.css') ;

    /**
     * The application for searching messages in the experimental e-Log
     *
     * @returns {ELog_Search}
     */
    function ELog_Search (experiment, access_list) {

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

        this.on_update = function () {
            if (this.active) {
                this._init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment  = experiment ;
        this.access_list = access_list ;


        // Public services

        this.search_message_by_id = function (id, show_in_vicinity) {
            this._init() ;
            this._bymessage.find('input[name="message2search"]').val(id) ;
            var show_in_vicinity_elem = this._bymessage.find('input[name="show_in_vicinity"]') ;
            if (show_in_vicinity) show_in_vicinity_elem.attr('checked', 'checked') ;
            else                  show_in_vicinity_elem.removeAttr('checked') ;
            this._tabs.tabs('option', 'active', 4) ;
            this._search_bymessage() ;
        } ;

        this.search_run_by_num = function (num, show_in_vicinity) {
            this._init() ;
            this._byrun.find('input[name="run2search"]').val(num) ;
            var show_in_vicinity_elem = this._byrun.find('input[name="show_in_vicinity"]') ;
            if (show_in_vicinity) show_in_vicinity_elem.attr('checked', 'checked') ;
            else                  show_in_vicinity_elem.removeAttr('checked') ;
            this._tabs.tabs('option', 'active', 3) ;
            this._search_byrun() ;
        } ;
        this.search_message_by_text = function (text2search) {
            this._init() ;
            this._simple.find('input[name="text2search"]').val(text2search) ;
            this._simple.find('input[name="search_in_deleted"]').attr('checked', 'checked') ;
            this._tabs.tabs('option', 'active', 0) ;
            this._search_simple() ;
        } ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._wa = null ;
        this._tabs = null ;

        this._simple     = null ;
        this._advanced   = null ;
        this._byrunrange = null ;
        this._byrun      = null ;
        this._bymessage  = null ;

        this._info    = null ;
        this._updated = null ;

        this._viewer = null ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this.container.html('<div id="elog-search"></div>') ;
            this._wa = this.container.find('div#elog-search') ;

            if (!this.access_list.elog.read_messages) {
                this._wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var time_title = "Use either the full format: '2013-11-15 14:56:00' \nor the day only format: '2013-11-15'" ;
            var html =
'<div id="ctrl">' +
'  <div id="tabs">' +

'    <ul>' +
'      <li><a href="#simple">Simple</a></li>' +
'      <li><a href="#advanced">Advanced</a></li>' ;
            if (!this.experiment.is_facility) html +=
'      <li><a href="#byrunrange">Range of Runs</a></li>' +
'      <li><a href="#byrun">By Run #</a></li>' ;
            html +=
'      <li><a href="#bymessage">By Message ID</a></li>' +
'    </ul>' +

'    <div id="simple">' +
'      <div class="search-dialog">' +
'        <div class="group">' +
'          <span class="label">Text to search:</span>' +
'          <input class="update-trigger" type="text" name="text2search" value="" size=24 />' +
'        </div>' +
'        <div class="group">' +
'          <span class="label">Include:</span>' +
'          <div><input class="update-trigger" type="checkbox" name="search_in_deleted" checked="checked" /> deleted messages</div>' +
'        </div>' +
'        <div class="buttons" style="float:left;" >' +
'          <button class="control-button" name="simple:search" title="search and display results">Search</button>' +
'          <button class="control-button" name="simple:reset"  title="reset the form">Reset</button>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'    </div>' +

'    <div id="advanced">' +
'      <div class="search-dialog">' +
'        <div class="group">' +
'          <span class="label">Text to search:</span>' +
'          <input class="update-trigger" type="text" name="text2search" value="" size=24 />' +
'          <div style="float:left;">' +
'            <span class="label">Tag:</span>' +
'            <select class="update-trigger" name="tag"></select>' +
'          </div>' +
'          <div style="float:left;">' +
'            <span class="label">Author:</span>' +
'            <select class="update-trigger" name="author"></select>' +
'          </div>' +
'          <div style="clear:both;"></div>' +
'        </div>' +
'        <div class="group">' +
'          <span class="label">Posted at:</span>' +
'          <div><input class="update-trigger" type="checkbox" name="posted_at_instrument" '+(this.experiment.is_facility ? 'disabled' : '')+' /> instrument</div>' +
'          <div><input class="update-trigger" type="checkbox" name="posted_at_experiment" '+(this.experiment.is_facility ? 'disabled' : '')+' checked="checked" /> experiment</div>' +
'          <div><input class="update-trigger" type="checkbox" name="posted_at_shifts"     '+(this.experiment.is_facility ? 'disabled' : 'checked="checked"')+' /> shifts</div>' +
'          <div><input class="update-trigger" type="checkbox" name="posted_at_runs"       '+(this.experiment.is_facility ? 'disabled' : 'checked="checked"')+' /> runs</div>' +
'        </div>' +
'        <div class="group">' +
'          <div title="'+time_title+'">' +
'            <span class="label">Begin Time:</span>' +
'            <div><input class="update-trigger" type="text" name="begin" value="" size=24 /></div>' +
'          </div>' +
'          <div title="'+time_title+'">' +
'            <span class="label">End Time:</span>' +
'            <div><input class="update-trigger" type="text" name="end" value="" size=24 /></div>' +
'          </div>' +
'        </div>' +
'        <div class="group">' +
'          <span class="label">Include:</span>' +
'          <div><input class="update-trigger" type="checkbox" name="search_in_deleted" checked="checked" /> deleted messages</div>' +
'        </div>' +
'        <div class="buttons" style="float:left;" >' +
'          <button class="control-button" name="advanced:search" title="search and display results">Search</button>' +
'          <button class="control-button" name="advanced:reset"  title="reset the form">Reset</button>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'    </div>' ;
            if (!this.experiment.is_facility) html +=

'    <div id="byrunrange">' +
'      <div class="search-dialog">' +
'        <div  class="group" >' +
'          <span class="label">Range of Runs:</span>' +
'          <input class="update-trigger" type="text" name="runs2search" value="" size=24' +
'                 title="Enter a run number or a range of runs where to look for messages.' +
' For a single run put its number. For a range the correct syntax is: 12-35"' +
'          />' +
'        </div>' +
'        <div class="group">' +
'          <span class="label">Include:</span>' +
'          <div><input class="update-trigger" type="checkbox" name="search_in_deleted" checked="checked" /> deleted messages</div>' +
'        </div>' +
'        <div class="buttons" style="float:left;" >' +
'          <button class="control-button" name="byrunrange:search" title="search and display results">Search</button>' +
'          <button class="control-button" name="byrunrange:reset"  title="reset the form">Reset</button>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'    </div>' +

'    <div id="byrun">' +
'      <div class="search-dialog">' +
'        <div  class="group" >' +
'          <span class="label">Run #:</span>' +
'          <input class="update-trigger" type="text" name="run2search" value="" size=24 />' +
'        </div>' +
'        <div class="group" >' +
'          <span class="label">Include:</span>' +
'          <div><input class="update-trigger" type="checkbox" name="show_in_vicinity" /> surrounding messages</div>' +
'        </div>' +
'        <div class="buttons" style="float:left;" >' +
'          <button class="control-button" name="byrun:search" title="search and display results">Search</button>' +
'          <button class="control-button" name="byrun:reset"  title="reset the form">Reset</button>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'    </div>' ;

            html +=

'    <div id="bymessage">' +
'      <div class="search-dialog">' +
'        <div  class="group" >' +
'          <span class="label">Message ID:</span>' +
'          <input class="update-trigger" type="text" name="message2search" value="" size=24 />' +
'        </div>' +
'        <div class="group" >' +
'          <span class="label">Include:</span>' +
'          <div><input class="update-trigger" type="checkbox" name="show_in_vicinity" /> surrounding messages</div>' +
'        </div>' +
'        <div class="buttons" style="float:left;" >' +
'          <button class="control-button" name="bymessage:search" title="search and display results">Search</button>' +
'          <button class="control-button" name="bymessage:reset"  title="reset the form">Reset</button>' +
'        </div>' +
'        <div style="clear:both;"></div>' +
'      </div>' +
'    </div>' +

'  </div>' +
'</div>' +

'<div id="body">' +
'  <div class="info" id="info" style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div style="float:left;"></div>' +
'  <div id="viewer" class="elog-msg-viewer"></div>' +
'</div>' ;
            this._wa.html(html) ;

            this._tabs = this._wa.find('div#ctrl > div#tabs') ;
            this._tabs.tabs() ;

            this._simple     = this._tabs.children('div#simple') ;
            this._advanced   = this._tabs.children('div#advanced') ;
            this._byrunrange = this._tabs.children('div#byrunrange') ;
            this._byrun      = this._tabs.children('div#byrun') ;
            this._bymessage  = this._tabs.children('div#bymessage') ;

            this._tabs.find('button.control-button').button().click(function () {
                var s2 = this.name.split(':') ;
                var search_type = s2[0] ;
                var op = s2[1] ;
                switch (search_type) {
                    case 'simple'     : if (op === 'search') _that._search_simple() ;     else _that._search_simple_reset() ;     break ;
                    case 'advanced'   : if (op === 'search') _that._search_advanced() ;   else _that._search_advanced_reset() ;   break ;
                    case 'byrunrange' : if (op === 'search') _that._search_byrunrange() ; else _that._search_byrunrange_reset() ; break ;
                    case 'byrun'      : if (op === 'search') _that._search_byrun() ;      else _that._search_byrun_reset() ;      break ;
                    case 'bymessage'  : if (op === 'search') _that._search_bymessage() ;  else _that._search_bymessage_reset() ;  break ;
                }
            }) ;
            this._simple    .find('.update-trigger').change(function () { _that._search_simple() ;     }) ;
            this._advanced  .find('.update-trigger').change(function () { _that._search_advanced() ;   }) ;
            this._byrunrange.find('.update-trigger').change(function () { _that._search_byrunrange() ; }) ;
            this._byrun     .find('.update-trigger').change(function () { _that._search_byrun() ;      }) ;
            this._bymessage .find('.update-trigger').change(function () { _that._search_bymessage() ;  }) ;

            var body = this._wa.children('div#body') ;

            this._info    = body.find('div#info') ;
            this._updated = body.find('div#updated') ;

            this._viewer = new ELog_MessageViewer (
                this ,
                this._wa.find('#viewer') ,
                {
                    allow_groups: true ,
                    allow_runs:   !this.experiment.is_facility ,
                    allow_shifts: !this.experiment.is_facility
                }
            ) ;

            ELog_Utils.load_tags_and_authors (
                this.experiment.id ,
                null ,
                function (tags, authors) {

                    var html = '<option></option>' ;
                    for (var i in tags) html += '<option>'+tags[i]+'</option>' ;
                    _that._advanced.find('select[name="tag"]').html(html) ;

                    var html = '<option></option>' ;
                    for (var i in authors) html += '<option>'+authors[i]+'</option>' ;
                    _that._advanced.find('select[name="author"]').html(html) ;
                } ,
                function (msg)  {
                    Fwk.report_error(msg) ;
                }
            ) ;

            // Process global search options

            if ('message' in app_config.global_extra_params) {
                this.search_message_by_id(parseInt(app_config.global_extra_params['message']), true) ;
            } else if ('run' in app_config.global_extra_params) {
                this.search_run_by_num(parseInt(app_config.global_extra_params['run']), true) ;
            }
        } ;

        this._search_simple = function () {

            this._init() ;

            this._search_and_display({
                text2search: this._simple.find('input[name="text2search"]').val()} ,
                this._simple.find('input[name="search_in_deleted"]').attr('checked')
            ) ;
        } ;
        this._search_simple_reset = function () {
            this._simple.find('input[name="text2search"]').val('') ;
            this._simple.find('input[name="search_in_deleted"]').attr('checked', 'checked') ;
        } ;

        this._search_advanced = function () {

            this._init() ;

            this._search_and_display({
                text2search:          this._advanced.find('input[name="text2search"]').val() ,
                tag:                  this._advanced.find('select[name="tag"]')       .val() ,
                author:               this._advanced.find('select[name="author"]')    .val() ,
                posted_at_instrument: this._advanced.find('input[name="posted_at_instrument"]').attr('checked') ? 1 : 0 ,
                posted_at_experiment: this._advanced.find('input[name="posted_at_experiment"]').attr('checked') ? 1 : 0 ,
                posted_at_shifts:     this._advanced.find('input[name="posted_at_shifts"]')    .attr('checked') ? 1 : 0 ,
                posted_at_runs:       this._advanced.find('input[name="posted_at_runs"]')      .attr('checked') ? 1 : 0 ,
                begin:                this._advanced.find('input[name="begin"]').val() ,
                end:                  this._advanced.find('input[name="end"]')  .val()} ,
                this._advanced.find('input[name="search_in_deleted"]').attr('checked')
            ) ;
        } ;
        this._search_advanced_reset = function () {
            this._init() ;
            this._advanced.find('input[name="text2search"]').val('') ;
            this._advanced.find('select[name="tag"]')       .val('') ;
            this._advanced.find('select[name="author"]')    .val('') ;
            this._advanced.find('input[name="search_in_deleted"]')   .attr('checked', 'checked') ;
            this._advanced.find('input[name="posted_at_instrument"]').removeAttr('checked') ;
            this._advanced.find('input[name="posted_at_experiment"]').attr('checked', 'checked') ;
            this._advanced.find('input[name="posted_at_shifts"]')    .attr('checked', 'checked') ;
            this._advanced.find('input[name="posted_at_runs"]')      .attr('checked', 'checked') ;
            this._advanced.find('input[name="begin"]').val('') ;
            this._advanced.find('input[name="end"]')  .val('') ;
        } ;

        this._search_byrunrange = function () {

            this._init() ;

            this._search_and_display({
                range_of_runs : this._byrunrange.find('input[name="runs2search"]').val()} ,
                this._byrunrange.find('input[name="search_in_deleted"]').attr('checked')
            ) ;
        } ;
        this._search_byrunrange_reset = function () {
            this._init() ;
            this._byrunrange.find('input[name="runs2search"]').val('') ;
            this._byrunrange.find('input[name="search_in_deleted"]').attr('checked', 'checked') ;
        } ;

        this._search_byrun = function () {

            this._init() ;

            var run_num = parseInt(this._byrun.find('input[name="run2search"]').val()) ;
            if (run_num) {

                var show_in_vicinity = this._byrun.find('input[name="show_in_vicinity"]').attr('checked') ? 1 : 0 ;

                this._updated.html('Loading messages...') ;

                var params = {
                    exper_id         : this.experiment.id ,
                    run_num          : run_num ,
                    show_in_vicinity : show_in_vicinity
                } ;
                Fwk.web_service_GET (
                    '../logbook/ws/message_search_one.php' ,
                    params ,
                    function (data) {
                        _that._viewer.load(data.ResultSet.Result) ;
                        if (show_in_vicinity) _that._viewer.focus_at_run(run_num) ;
                        _that._info.html('<b>'+_that._viewer.num_rows()+'</b> messages') ;
                        _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    } ,
                    function (msg) {
                        Fwk.report_error(msg) ; 
                    }
                ) ;
            }
        } ;
        this._search_byrun_reset = function () {
            this._init() ;
            this._byrun.find('input[name="run2search"]').val('') ;
            this._byrun.find('input[name="show_in_vicinity"]').removeAttr('checked') ;
        } ;

        this._search_bymessage = function () {

            this._init() ;

            var message_id = parseInt(this._bymessage.find('input[name="message2search"]').val()) ;
            if (message_id) {

                var show_in_vicinity = this._bymessage.find('input[name="show_in_vicinity"]').attr('checked') ? 1 : 0 ;

                this._updated.html('Loading messages...') ;

                var params = {
                    id               : message_id ,
                    show_in_vicinity : show_in_vicinity
                } ;
                Fwk.web_service_GET (
                    '../logbook/ws/message_search_one.php' ,
                    params ,
                    function (data) {
                        _that._viewer.load(data.ResultSet.Result) ;
                        if (show_in_vicinity) _that._viewer.focus_at_message(message_id) ;
                        _that._info.html('<b>'+_that._viewer.num_rows()+'</b> messages') ;
                        _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    } ,
                    function (msg) {
                        Fwk.report_error(msg) ; 
                    }
                ) ;
            }
        } ;
        this._search_bymessage_reset = function () {
            this._init() ;
            this._bymessage.find('input[name="message2search"]').val('') ;
            this._bymessage.find('input[name="show_in_vicinity"]').removeAttr('checked') ;
        } ;

        this._default_params = {
            id:                   this.experiment.id ,
            text2search:          '' ,
            tag:                  '' ,
            author:               '' ,
            search_in_messages :  1 ,
            search_in_tags :      0 ,
            search_in_values :    0 ,
            posted_at_instrument: 0,
            posted_at_experiment: 1 ,
            posted_at_shifts:     1 ,
            posted_at_runs:       1 ,
            begin:                '' ,
            end:                  '' ,
            format :              'detailed'
        } ;

        this._search_and_display = function (params, inject_deleted_messages) {

            for (var p in this._default_params)
                if (typeof params[p] === 'undefined')
                    params[p] = this._default_params[p] ;

            if (inject_deleted_messages) params.inject_deleted_messages = '' ;

            this._updated.html('Loading messages...') ;

            Fwk.web_service_GET (
                '../logbook/ws/message_search.php' ,
                params ,
                function (data) {
                    _that._viewer.load(data.ResultSet.Result) ;
                    _that._info.html('<b>'+_that._viewer.num_rows()+'</b> messages') ;
                    _that._updated.html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;
    }
    Class.define_class (ELog_Search, FwkApplication, {}, {}) ;

    return ELog_Search ;
}) ;
