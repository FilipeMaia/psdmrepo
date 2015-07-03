define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/Widget', 'webfwk/StackOfRows', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, Widget, StackOfRows, FwkApplication, Fwk) {

    cssloader.load('../portal/css/ELog_Shifts.css') ;

    /**
     * The application for viewing and managing  shifts in the experimental e-Log
     *
     * @returns {ELog_Shifts}
     */
    function ELog_Shifts (experiment, access_list) {

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

        this._prev_refresh_sec     = 0 ;
        this._refresh_interval_sec = 10 ;

        this.on_update = function () {
            this._init() ;
            if (this.active && this._refresh_interval_sec) {
                var now_sec = Fwk.now().sec ;
                if (Math.abs(now_sec - this._prev_refresh_sec) > this._refresh_interval_sec && !this._num_edit_sessions) {
                    this._prev_refresh_sec = now_sec ;
                    var update_mode = true ;
                    this._search(update_mode) ;
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

        this._num_edit_sessions = 0 ;

        this.editor_open = function () {
            this._init() ;
            this._num_edit_sessions++ ;
            if (this._num_edit_sessions === 1)
                this._ctrl().find('button.control-button').button('disable') ;
        } ;

        this.editor_close = function () {
            this._init() ;
            this._num_edit_sessions-- ;
            if (this._num_edit_sessions === 0)
                this._ctrl().find('button.control-button').button('enable') ;
        } ;


        this._last_request = null ;
        this._max_seconds = 0 ;

        // ---------------------------------------
        //   BEGIN INITIALIZING THE UI FROM HERE
        // ---------------------------------------

        this._is_initialized = false ;

        this.wa = null ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            // -- no further initialization beyond this point if not authorized

            if (!this.access_list.elog.manage_shifts) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }

            // -- set up event handlers

            this._ctrl().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'search':
                        _that._search() ;
                        break ;
                    case 'create' :
                        _that._create() ;
                        break ;
                }
            }) ;

            // -- initiate the loading

            this._search() ;
        } ;

        /**
         * Initialize the work area and return an element. Use an html document from 
         * an optional parameter if the one passed into the function. Otherwise use
         * the standard initialization.
         *
         *   ()     - the standard initialization
         *   (html) - initialization with the specified content
         *
         * NOTE: if the parameter is present the method would always go
         * for the forced (re-)initialization regardless of any prior
         * initialization attempts.
         * 
         * @param   {string} the new content
         * @returns {string} the JQuery element
         */
        this._wa = function (html) {
            if (this._wa_elem) {
                if (html !== undefined) {
                    this._wa_elem.html(html) ;
                }
            } else {
                this.container.html('<div id="elog-shifts"></div>') ;
                this._wa_elem = this.container.find('div#elog-shifts') ;
                if (html === undefined) {
                    html =
'<div id="ctrl">' +
'  <button class="control-button"' +
'          name="create"' +
'          style="color:red;"' +
'          title="stop the current shift and begin the new one" >NEW SHIFT</button>' +
'  <button class="control-button"' +
'          name="search"' +
'          title="search and display results" >Search</button>' +
'</div>' +
'<div id="body">' +
'  <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div id="viewer" class="elog-msg-viewer"></div>' +
'</div>' ;
                }
                this._wa_elem.html(html) ;
            }
            return this._wa_elem ;
        } ;

        this._ctrl = function () {
            if (!this._ctrl_elem) {
                this._ctrl_elem = this._wa().children('#ctrl') ;
            }
            return this._ctrl_elem ;
        } ;


        this._body = function () {
            if (!this._body_elem) {
                this._body_elem = this._wa().children('#body') ;
            }
            return this._body_elem ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) {
                this._info_elem = this._body().children('#info') ;
            }
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) {
                this._updated_elem = this._body().children('#updated') ;
            }
            this._updated_elem.html(html) ;
        } ;

        this._shift_table  = function () {
            if (!this._shift_table_obj) {
                this._viewer_elem = this._body().children('#viewer') ;

                var hdr = [
                    {id: 'begin',         title: 'Begin time',   width:  160} ,
                    {id: 'end',           title: 'End Time',     width:  160} ,
                    {id: 'duration_days', title: 'days',         width:   40, align: 'right'} ,
                    {id: 'duration_hms',  title: 'hh:mm:ss',     width:   65, align: 'right'} ,
                    {id: 'duration_bar',  title: '&nbsp;',       width:  200} ,
                    {id: 'num_runs',      title: '# runs',       width:   50, align: 'right'} ,
                    {id: '_',                                    width:   10} ,
                    {id: 'goals',         title: 'Goals',        width:  320, style: 'color:maroon;'}
                ] ;
                this._shift_table_obj = new StackOfRows.StackOfRows (
                    hdr ,
                    [] ,
                    {
                        theme: 'stack-theme-brown'
                    }
                ) ;
                this._shift_table_obj.display(this._viewer_elem) ;
            }
            return this._shift_table_obj ;
        }

        this._create = function () {
            this._set_updated('Creating...') ;
            var params = {
                exper_id: this.experiment.id
            } ;
            Fwk.web_service_POST (
                '../logbook/ws/shift_create.php' ,
                params ,
                function (data) {
                    var update_mode = false ;
                    _that._search (update_mode, function () {
                        console.log('ELog_Shifts::_create() open the tab of the new shift in the editing mode') ;
                    }) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;

        /**
         * Search for shifts
         *
         * @returns {undefined}
         */
        this._search = function (update_mode, on_success) {
            this._set_updated('Loading...') ;
            var params = {
                exper_id: this.experiment.id
            } ;
            Fwk.web_service_GET (
                '../logbook/ws/RequestAllShifts.php' ,
                params ,
                function (data) {
                    _that._set_info('<b>'+data.Shifts.length+'</b> shifts') ;
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(update_mode, data.Shifts, data.MaxSeconds) ;
                    if (on_success) on_success(data.Shifts, data.MaxSeconds) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;

        /**
         * Display a table of shifts
         *
         * @returns {undefined}
         */    
        this._display = function (update_mode, shifts, max_seconds) {

            var table = this._shift_table() ;

            // Redisplay the table from scratch if:
            // 
            //   - this is the very first request
            //   - the maxumu shift duration has changed (and we need to rescale the shift duration bars)
            //   - forced to do so by input parameter 'update_mode === false'
            //
            // Only apply updates to the table if:
            //
            //   - changed status of the last shift from the previous request
            //   - new shifts appeared since the previous request

            if (!this._last_request) {

                this._last_request = shifts ;
                this._max_seconds  = max_seconds ;

                for (var i in this._last_request) {
                    var s = this._last_request[i] ;
                    s.row_id = table.append(this._shift2row(s, max_seconds)) ;
                }

            } else if ((this._max_seconds !== max_seconds) || !update_mode) {

                table.reset() ;

                // TODO: consider rescaling the shift duration bars w/o redisplaying
                //       the table from scratch

                this._last_request = shifts ;
                this._max_seconds  = max_seconds ;

                for (var i in this._last_request) {
                    var s = this._last_request[i] ;
                    s.row_id = table.append(this._shift2row(s, max_seconds)) ;
                }

            } else {

                // -- update the first row if there were any changes in the shift status

                var s_last_old = this._last_request[0] ;
                var s_last_new = shifts[shifts.length - this._last_request.length] ;    // in case if there are more new shifts
                if (s_last_old.sec !== s_last_new.sec) {
                    table.update_row(s_last_old.row_id, this._shift2row(s_last_new, max_seconds)) ;
                    this._last_request[0] = s_last_new ;
                    this._last_request[0].row_id = s_last_old.row_id ;
                }

                // -- check if more shifts should be added to the front

                if (shifts.length > this._last_request.length) {

                    // -- insert new shifts at the begining of the list

                    for (var i = shifts.length - this._last_request.length - 1; i >= 0; i--) {
                        var s = shifts[i] ;
                        s.row_id = table.insert_front(this._shift2row(s, max_seconds)) ;
                    }

                    // -- and do NOT bother to carry over identifiers of rows from
                    //    the previous request

                    ;

                    this._last_request = shifts ;
                    this._max_seconds  = max_seconds ;
                }
            }
        } ;

        this._shift2row = function (s, max_seconds) {
            var max_seconds = max_seconds ? max_seconds : 1 ;
            var duration_days = s.durat_days ? s.durat_days : '&nbsp;' ;
            var duration_hms  = s.durat_hms  ? s.durat_hms  : '&nbsp;' ;
            var duration_bar_width = Math.floor(185.0 * (Math.min(s.sec, max_seconds) / max_seconds)) ;
            var duration_bar_color = '#5C5C33' ;
            if (s.is_open) {
                duration_days = '<span style="color:red; font-weight:bold;">'+duration_days+'</span>' ;
                duration_hms  = '<span style="color:red;">'+duration_hms+'</span>' ;
                duration_bar_color = 'red' ;
            } else {
                duration_days = '<span style="font-weight:bold;">'+duration_days+'</span>' ;
            }
            var row = {
                title: {
                    begin:         '<b>'+s.begin_ymd+'</b>&nbsp;&nbsp;'+s.begin_hms ,
                    end:           '<b>'+s.end_ymd  +'</b>&nbsp;&nbsp;'+s.end_hms ,
                    duration_days: duration_days ,
                    duration_hms:  duration_hms ,
                    duration_bar:  '<div style="margin-left:15px; width:'+duration_bar_width+'px; background:'+duration_bar_color+';">&nbsp;</div>' ,
                    num_runs:      s.num_runs ? s.num_runs : '&nbsp;' ,
                    goals:         s.goals !== '' ? s.goals.substr(0, 64) : '&nbsp;'
                } ,
                body: new ELog_Shifts_ShiftBody(this, s) ,
                block_common_expand: false
            } ;
            return row ;
        } ;
    }
    Class.define_class (ELog_Shifts, FwkApplication, {}, {}) ;


    function ELog_Shifts_ShiftBody (parent, shift) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        Widget.Widget.call(this) ;

        // -- parameters

        this.parent = parent ;
        this.experiment = parent.experiment ;
        this.access_list = parent.access_list ;

        this.shift = shift ;

        this._shift_url = function () {
            var idx = window.location.href.indexOf('?') ;
            var url = (idx < 0 ? window.location.href : window.location.href.substr(0, idx))+'?exper_id='+this.experiment.id+'&app=elog:search&params=shift:'+this.shift.id;
            var html = '<a href="'+url+'" target="_blank" title="Click to open in a separate tab, or cut and paste to incorporate into another document as a link."><img src="../portal/img/link.png"></img></a>' ;
            return html ;
        }

        this._cont = function () {
            if (!this._cont_elem) {
                var html =
'<div class="shift-cont" shift_id="'+this.shift.id+'">' +
'  <div id="ctrl">' +
'    <div style="float:left;" >'+this._shift_url()+'</div>' +
'    <div style="float:right;" >' +
'      <button class="control-button"' +
'              name="edit"' +
'              style="color:red;"' +
'              title="edit the shift" >Edit</button>' +
'      <button class="control-button"' +
'              name="save"' +
'              title="finish the editing session and save modifications to the database" >Save</button>' +
'      <button class="control-button"' +
'              name="cancel"' +
'              title="cancel " >Cancel</button>' +
'      <button class="control-button"' +
'              name="delete"' +
'              style="color:red;"' +
'              title="delete the shift" >DELETE SHIFT</button>' +
'    </div>' +
'    <div style="clear:both;"></div>' +
'  </div>' +
'  <div id="body">' +
'    <div id="viewer" ></div>' +
'    <div id="editor" class="elog-shifts-disabled" ></div>' +
'  </div>' +
'</div>' ;
                this.container.html(html) ;        
                this._cont_elem = this.container.children('.shift-cont') ;
            }
            return this._cont_elem ;
        } ;
        this._ctrl = function () {
            if (!this._ctrl_elem) {
                this._ctrl_elem = this._cont().children('#ctrl') ;
            }
            return this._ctrl_elem ;
        } ;
        this._ctrl_button = function (name) {
            if (!this._ctrl_button_elem) {
                this._ctrl_button_elem = {} ;
            }
            if (!this._ctrl_button_elem[name]) {
                this._ctrl_button_elem[name] = this._ctrl().find('button[name="'+name+'"]').button() ;
            }
            return this._ctrl_button_elem[name] ;
        } ;
        this._edit_button   = function () { return this._ctrl_button('edit') ;  } ;
        this._save_button   = function () { return this._ctrl_button('save') ;  } ;
        this._cancel_button = function () { return this._ctrl_button('cancel') ;  } ;
        this._delete_button = function () { return this._ctrl_button('delete') ;  } ;

        this._body = function () {
            if (!this._body_elem) {
                this._body_elem = this._cont().children('#body') ;
            }
            return this._body_elem ;
        } ;
        this._viewer = function () {
            if (!this._viewer_elem) {
                this._viewer_elem = this._body().children('#viewer') ;
            }
            return this._viewer_elem ;
        } ;
        this._editor = function () {
            if (!this._editor_elem) {
                this._editor_elem = this._body().children('#editor') ;
                this._editor_elem.html(
'<textarea id="goals" rows="6" cols="72" ></textarea>' +
'<div class="shift-interval" id="end" >' +
  '<div class="label">End time:</div>' +
  '<input class="shift-day" id="end-day" type="text" size=10 />' +
  '<input class="shift-hms" id="end-h"   type="text" size=1 />:' +
  '<input class="shift-hms" id="end-m"   type="text" size=1 />:' +
  '<input class="shift-hms" id="end-s"   type="text" size=1 />' +
'</div>' +
'<div class="shift-interval" id="begin" >' +
  '<div class="label">Begin time:</div>' +
  '<input class="shift-day" id="begin-day" type="text" size=10 />' +
  '<input class="shift-hms" id="begin-h"   type="text" size=1 />:' +
  '<input class="shift-hms" id="begin-m"   type="text" size=1 />:' +
  '<input class="shift-hms" id="begin-s"   type="text" size=1 />' +
'</div>'
                ) ;
            }
            return this._editor_elem ;
        } ;
        this._editor_field = function (id) {
            if (!this._editor_field_elem) {
                this._editor_field_elem = {} ;
            }
            if (!this._editor_field_elem[id]) {
                this._editor_field_elem[id] = this._editor().find('#'+id) ;
            }
            return this._editor_field_elem[id] ;
        } ;

        // ------------------------------------------------
        // Override event handler defined in thw base class
        // ------------------------------------------------

        this._is_rendered = false ;

        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this._edit_button()  .button('enable') .click(function () { _that._edit() ; }) ;
            this._save_button()  .button('disable').click(function () { _that._edit_save() ; }) ;
            this._cancel_button().button('disable').click(function () { _that._edit_cancel() ; }) ;
            this._delete_button().button('enable') .click(function () {}) ;

            this._viewer().html('<pre>'+this.shift.goals+'</pre>') ;
        } ;

        function init_date (parent, cxt, ymd, hms2array) {
            var h = hms2array[0] ;
            var m = hms2array[1] ;
            var s = hms2array[2] ;
            parent._editor_field(cxt+'-day').datepicker({ dateFormat: "yy-mm-dd" }).val(ymd) ;
            parent._editor_field(cxt+'-h'  ).val(h) ;
            parent._editor_field(cxt+'-m'  ).val(m) ;
            parent._editor_field(cxt+'-s'  ).val(s) ;
        }
        this._edit = function () {

            this.parent.editor_open() ;

            this._edit_button  ().button('disable') ;
            this._save_button  ().button('enable')
            this._cancel_button().button('enable') ;
            this._delete_button().button('disable') ;

            this._viewer().addClass('elog-shifts-disabled') ;
            this._editor().removeClass('elog-shifts-disabled') ;

            this._editor_field('goals').text(this.shift.goals) ;

            if (this.shift.is_open) {
                this._editor_field('end-day').attr('disabled', 'disabled');
                this._editor_field('end-h')  .attr('disabled', 'disabled');
                this._editor_field('end-m')  .attr('disabled', 'disabled');
                this._editor_field('end-s')  .attr('disabled', 'disabled');
            } else {
                init_date (
                    this ,
                    'end' ,
                    this.shift.end_ymd ,
                    this.shift.end_hms.split(':')) ;
            }
            init_date (
                this ,
                'begin' ,
                this.shift.begin_ymd ,
                this.shift.begin_hms.split(':')) ;

        } ;
        this._editor_get_time = function (cxt) {
            if (this.shift.is_open && cxt === 'end') return '' ;
            var day = this._editor_field(cxt+'-day').val() ;
            var h = parseInt(this._editor_field(cxt+'-h').val()) ;
            var m = parseInt(this._editor_field(cxt+'-m').val()) ;
            var s = parseInt(this._editor_field(cxt+'-s').val()) ;
            var time = day + ' ' + (h < 10 ? '0' + h : h) + ':' + (m < 10 ? '0' + m : m) + ':' + (s < 10 ? '0' + s : s) ;
            return time ;
        } ;
        this._edit_save = function () {

            this.parent._set_updated('Saving...') ;
            var params = {
                id:         this.shift.id ,
                goals:      this._editor().children('textarea').val() ,
                begin_time: this._editor_get_time('begin') ,
                end_time:   this._editor_get_time('end')
            } ;
            Fwk.web_service_POST (
                '../logbook/ws/shift_save.php' ,
                params ,
                function (data) {

                    _that._edit_button  ().button('enable') ;
                    _that._save_button  ().button('disable')
                    _that._cancel_button().button('disable') ;
                    _that._delete_button().button('enable') ;

                    _that.shift.goals = _that._editor().children('textarea').val() ;

                    _that._viewer().removeClass('elog-shifts-disabled') ;
                    _that._editor().addClass('elog-shifts-disabled') ;

                    _that._viewer().html('<pre>'+_that.shift.goals+'</pre>') ;

                    // Notify the parent to make sure that eveything is taken cared of,
                    // such as: updating the header of this shift as well as any neighboring
                    // shifts (due to changes in the shift's begin/end times).

                    _that.parent.editor_close() ;
                    _that.parent._search() ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                }
            ) ;
        } ;
        this._edit_cancel = function () {

            this.parent.editor_close() ;

            this._edit_button  ().button('enable') ;
            this._save_button  ().button('disable')
            this._cancel_button().button('disable') ;
            this._delete_button().button('enable') ;

            this._viewer().removeClass('elog-shifts-disabled') ;
            this._editor().addClass('elog-shifts-disabled') ;

            this._viewer().html('<pre>'+this.shift.goals+'</pre>') ;
        } ;
    }
    Class.define_class (ELog_Shifts_ShiftBody, Widget.Widget, {}, {}) ;

    return ELog_Shifts ;
}) ;
