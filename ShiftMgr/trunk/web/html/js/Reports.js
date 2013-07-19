function Reports (instr_name) {
    FwkDispatcherBase.call(this) ;
    this.instr_name = instr_name ;
}
define_class (Reports, FwkDispatcherBase, {}, {

    on_activate : function() {
        this.on_update() ;
        if (this.active) this.search() ;
    } ,

    on_deactivate : function() {
        this.init() ;
    } ,

    on_update : function (sec) {
        this.init() ;
        //if (this.active) this.search() ;
    } ,

    is_initialized : false ,

    init : function () {

        var that = this ;

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        var ctrl_elem = this.container.find('#shifts-search-controls') ;

        this.ctrl_range_elem = ctrl_elem.find('select[name="range"]') ;
        this.ctrl_begin_elem = ctrl_elem.find('input[name="begin"]') ;
        this.ctrl_end_elem   = ctrl_elem.find('input[name="end"]') ;

        this.ctrl_begin_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;
        this.ctrl_end_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;

        ctrl_elem.find('button[name="reset"]').button().click(function () {
            var range = 'week' ;
            that.ctrl_range_elem.val(range) ;
            that.ctrl_begin_elem.val('') ;
            that.ctrl_end_elem.val('') ;
            that.ctrl_begin_elem.attr('disabled', 'disabled') ;
            that.ctrl_end_elem  .attr('disabled', 'disabled') ;
            that.search(range) ;
        }) ;
        
        this.ctrl_range_elem.change(function () {
            var range = that.ctrl_range_elem.val() ;
            switch (range) {
                case 'week'  :
                case 'month' :
                    that.ctrl_begin_elem.attr('disabled', 'disabled') ;
                    that.ctrl_end_elem  .attr('disabled', 'disabled') ;
                    that.search() ;
                    break ;
                case 'range' :
                    that.ctrl_begin_elem.removeAttr('disabled') ;
                    that.ctrl_end_elem  .removeAttr('disabled') ;
                    if (that.ctrl_begin_elem.val() || that.ctrl_end_elem.val()) that.search() ;
                    break ;
            }
        }) ;
        this.ctrl_begin_elem.change(function () {
            if (that.ctrl_begin_elem.val() || that.ctrl_end_elem.val()) that.search() ;
        }) ;
        this.ctrl_end_elem.change(function () {
            if (that.ctrl_begin_elem.val() || that.ctrl_end_elem.val()) that.search() ;
        }) ;
 
        this.search_info_elem = this.container.find('#shifts-search-info') ;
        this.search_list_elem = this.container.find('#shifts-search-list') ;
    } ,

    shifts : [] ,

    search : function () {
        var that = this ;
        var range = this.ctrl_range_elem.val() ;
        var params = {
            instr_name: this.instr_name ,
            range: range
        } ;
        if (range === 'range') {
            params.begin = that.ctrl_begin_elem.val() ;
            params.end   = that.ctrl_end_elem.val() ;
        }
        this.search_info_elem.html('Searching') ;
        this.shifts_service (
            '../shiftmgr/ws/shifts_get.php', 'GET', params ,
            function (shifts) {
                that.shifts = shifts ;
                that.display() ;
            }
        ) ;
    } ,

    shifts_service: function (url, type, params, when_done) {

        var that = this ;

        $.ajax ({
            type: type ,
            url:  url ,
            data: params ,
            success: function (result) {
                if(result.status !== 'success') {
                    Fwk.report_error(result.message) ;
                    return ;
                }
                if (when_done) when_done(result.shifts) ;
            } ,
            error: function () {
                Fwk.report_error('shift service is not available for instrument: '+that.inst_name) ;
            } ,
            dataType: 'json'
        }) ;
    } ,

    display : function () {
        var that = this ;
        var total = 0;
        var html = '';
        for (var idx in this.shifts) {
            var shift = this.shifts[idx];
            shift.is_initialized = false ;
            html += this.shift2html(idx);
            total++;
        }
        var info_html = '<b>'+total+'</b> shift'+(total==1?'':'s');
        this.search_info_elem.html(info_html) ;
        this.search_list_elem.html(html);

        this.search_list_elem.find('div.shift-hdr').click(function () {
            var idx = this.id ;
            that.shift_toggle(idx) ;
        }) ;
    } ,

    area_names : [
        {key: 'FEL' , name: 'FEL'},
        {key: 'BMLN', name: 'Beamline'} ,
        {key: 'CTRL', name: 'Controls'} ,
        {key: 'DAQ' , name: 'DAQ'} ,
        {key: 'LASR', name: 'Laser'} ,
        {key: 'HALL', name: 'Hutch/Hall'} ,
        {key: 'OTHR', name: 'Other'}
    ] ,

    activity_names : [
        {key: 'tuning'   , name: 'Tuning'} ,
        {key: 'alignment', name: 'Alignment'} ,
        {key: 'daq',       name: 'Data Taking'} ,
        {key: 'access',    name: 'Hutch Access'} ,
        {key: 'other',     name: 'Other'}
    ] ,

    shift2html : function (idx) {
        var s = this.shifts[idx] ;
        var html =
'<div class="shift-hdr" id="'+idx+'">' +
'  <div class="shift-toggler" ><span class="toggler ui-icon ui-icon-triangle-1-e" ></span></div>' +
'  <div class="shift-day"     >'+s.begin.day +'</div>' +
'  <div class="shift-begin"   >'+s.begin.hm  +'</div>' +
'  <div class="shift-end"     >'+s.end.hm    +'</div>' +
'  <div class="shift-duration">'+s.duration  +'</div>' +
'  <div class="shift-stopper" >'+(s.stopper?s.stopper+'%':'&nbsp;')+'</div>' +
'  <div class="shift-door"    >'+(s.door   ?s.door   +'%':'&nbsp;')+'</div>' ;
        for (var i in this.area_names) {
            var area_name = this.area_names[i].key ;
            var classes = 'shift-area' ;
            if (area_name === 'FEL') classes += '-first' ;
            else if (area_name === 'OTHR') classes += '-last' ;
            classes += ' '+area_name ;
            html +=
'  <div class="'+classes+'" ><div class="status_'+(s.area[area_name].problem ?'red':'neutral')+'"></div></div>' ;
        }
        html +=
'  <div class="shift-editor"   >'+s.editor  +'</div>' +
'  <div class="shift-modified" >'+s.modified+'</div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div class="shift-con shift-hdn" id="'+idx+'">Loading...</div>' ;
        return html ;
    } ,

    shift_toggle : function (idx) {
        var that = this ;
        var shift = this.shifts[idx] ;
        var tgl = this.search_list_elem.find('div.shift-hdr#'+idx+' span.toggler') ;
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;
        if (con.hasClass('shift-hdn')) {
            if (!shift.is_initialized) {
                shift.is_initialized = true ;
                shift.editing = false ;
                var html =
'<div>' +

'  <div style="float:left;">' +
'    <table style="font-size:90%;"><tbody>' +
'      <tr>' +
'        <td class="shift-grid-hdr " valign="top" >Begin</td>' +
'        <td class="shift-grid-val " valign="top" >' +
'          <div class="viewer" >' +
'            <span name="begin_day" ></span>' +
'            <span name="begin_hm" style="font-weight: bold; padding-left:10px;" ></span>' +
'          </div>' +
'          <div class="editor" >' +
'            <input name="begin_day" type="text" size=8 title="specify the begin date of the shift" />' +
'            <input name="begin_h"   type="text" size=2 title="hour: 0..23" />' +
'            <input name="begin_m"   type="text" size=2 title="minute: 0..59" />' +
'          </div></td>' +
'        <td class="shift-grid-hdr " valign="top" >Stopper Out</td>' +
'        <td class="shift-grid-val " valign="top" ><span name="stopper_hm">02:23</span></td>' +
'        <td class="shift-grid-val " valign="top" ><span name="stopper_percent">( 12 % )</span></td>' +
'      </tr>' +
'      <tr>' +
'        <td class="shift-grid-hdr " valign="top" >End</td>' +
'        <td class="shift-grid-val " valign="top" >' +
'          <div class="viewer" >' +
'            <span name="end_day" ></span>' +
'            <span name="end_hm" style="font-weight: bold; padding-left:10px;" ></span>' +
'          </div>' +
'          <div class="editor" >' +
'            <input name="end_day" type="text" size=8 title="specify the end date of the shift" />' +
'            <input name="end_h"   type="text" size=2 title="hour: 0..23" />' +
'            <input name="end_m"   type="text" size=2 title="minute: 0..59" />' +
'          </div></td>' +
'        <td class="shift-grid-hdr " valign="top" >Door Open</td>' +
'        <td class="shift-grid-val " valign="top" ><span name="door_hm">04:34</span></td>' +
'        <td class="shift-grid-val " valign="top" ><span name="door_percent">( 25 % )</span></td>' +
'      </tr>' +
'      <tr>' +
'        <td class="shift-grid-hdr " valign="top" >Notes</td>' +
'        <td class="shift-grid-val " valign="top" colspan="4" >'+
'          <div      class="viewer" name="notes" style="width:520px; height:68px; overflow:auto;" ></div>' +
'          <textarea class="editor" name="notes" rows="4" cols=62 title="general notes on the shift (if any)" ></textarea></td>' +
'      </tr>' +
'    </tbody></table>' +
'  </div>' +
'  <div style="float:left; margin-left:40px; padding-top:5px;">' +
'    <button name="edit"   title="switch to the editing mode" style="color:red; font-weight:bold;" >Edit</button>' +
'    <button name="save"   title="submit modifications and switch back to the vieweing mode to edit">Save</button>' +
'    <button name="cancel" title="discard modifications and revert back to the viewing mode">Cancel</button>' +
'  </div>' +
'  <div style="clear:both;"></div>' +

'  <div id="tabs" style="font-size:12px; margin-top:20px;">' +
'    <ul>' +
'      <li><a href="#area_evaluation_'+idx+'">Area Evaluation</a></li>' +
'      <li><a href="#time_allocation_'+idx+'">Time Use Allocation</a></li>' +
'    </ul>' +
'    <div id="area_evaluation_'+idx+'" >' +
'      <div style="font-size:11px; border:solid 1px #b0b0b0; padding:15px; padding-left:20px; padding-bottom:20px;" >' +
'        <table><tbody>' +
'          <tr>' +
'            <td class="shift-table-hdr " >Area</td>' +
'            <td class="shift-table-hdr " >Issues?</td>' +
'            <td class="shift-table-hdr " >Downtime</td>' +
'            <td class="shift-table-hdr " >Comments</td>' +
'          </tr>' ;
                for (var i in this.area_names) {
                    var area_name = this.area_names[i].key ;
                    html +=
'          <tr>' +
'            <td class="shift-table-val " valign="top">'+this.area_names[i].name+'</td>' +
'            <td class="shift-table-val " valign="top">' +
'              <div   class="viewer flag" name="'+area_name+'" /></div>' +
'              <input class="editor flag" name="'+area_name+'" type="checkbox" /></td>' +
'            <td class="shift-table-val " valign="top">'+
'              <div class="viewer">' +
'                <span class="hour_minute" name="'+area_name+'" ></span>' +
'              </div>' +
'              <div class="editor">'+
'                <input class="hour"   name="'+area_name+'" type="text" size=1 title="hours: 0.." />' +
'                <input class="minute" name="'+area_name+'" type="text" size=1 title="minutes: 0..59" />' +
'              </div></td>' +
'            <td class="shift-table-val ">' +
'              <div      class="viewer comment" name="'+area_name+'" ></div>' +
'              <textarea class="editor comment" name="'+area_name+'" rows="2" cols=48 title="explain the problem" ></textarea></td>' +
'          </tr>' ;
                }
                html +=
'          <tr>' +
'            <td class="shift-table-tot "                valign="top">Total</td>' +
'            <td class="shift-table-tot "                valign="top">&nbsp;</td>' +
'            <td class="shift-table-tot "                valign="top"><span class="hour_minute" name="area"></td>' +
'            <td class="shift-table-tot "                valign="top">&nbsp;</td>' +
'          </tr>' +
'        </tbody></table>' +
'      </div>' +
'    </div>' +
'    <div id="time_allocation_'+idx+'" >' +
'      <div style="font-size:11px; border:solid 1px #b0b0b0; padding:15px; padding-left:20px; padding-bottom:20px;" >' +
'        <table><tbody>' +
'          <tr>' +
'            <td class="shift-table-hdr "               >Activity</td>' +
'            <td class="shift-table-hdr "               >Time spent</td>' +
'            <td class="shift-table-hdr " align="right" >%</td>' +
'            <td class="shift-table-hdr "               >Comments</td>' +
'          </tr>' ;
                for (var i in this.activity_names) {
                    var activity_name = this.activity_names[i].key ;
                    html +=
'          <tr>' +
'            <td class="shift-table-val " valign="top">'+this.activity_names[i].name+'</td>' +
'            <td class="shift-table-val " valign="top">' ;
                    if (activity_name === 'other') {
                        html +=
'              <div>' +
'                <span class="hour_minute" name="'+activity_name+'" ></span>' +
'              </div></td>' ;
                    } else {
                        html +=
'              <div class="viewer">' +
'                <span class="hour_minute" name="'+activity_name+'" ></span>' +
'              </div>' +
'              <div class="editor">' +
'                <input class="hour"   name="'+activity_name+'" type="text" size=1 title="hours: 0.." />' +
'                <input class="minute" name="'+activity_name+'" type="text" size=1 title="minutes: 0..59" />' +
'              </div></td>' ;
                    }
                    html +=
'            <td class="shift-table-val " align="right" valign="top"><span class="percent" name="'+activity_name+'"></span></td>' +
'            <td class="shift-table-val ">' +
'              <div      class="viewer comment" name="'+activity_name+'" ></div>' +
'              <textarea class="editor comment" name="'+activity_name+'" rows="3" cols=48 title="explain the activity" ></textarea></td>' +
'          </tr>' ;
                }
                html +=
'          <tr>' +
'            <td class="shift-table-val " >&nbsp;</td>' +
'            <td class="shift-table-val " >&nbsp;</td>' +
'            <td class="shift-table-val " >&nbsp;</td>' +
'            <td class="shift-table-val " >&nbsp;</td>' +
'          </tr>' +
'        </tbody></table>' +
'      </div>' +
'    </div>' +
'  </div>' +
'</div>' ;
                con.html(html) ;
                con.find('div#tabs').tabs() ;
                this.shift_register_handlers(idx) ;
                this.shift_view(idx) ;
            }
            tgl.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
            con.removeClass('shift-hdn').addClass('shift-vis' ) ;
        } else {
            if (shift.editing) {
                Fwk.ask_yes_no (
                    'Warning: editing in progress' ,
                    'Would you like to abort the editing and discard all modifications?' ,
                    function () {
                        that.shift_edit_cancel(idx) ;
                        tgl.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
                        con.removeClass('shift-vis').addClass('shift-hdn') ;
                    }
                ) ; 
            } else {
                tgl.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
                con.removeClass('shift-vis').addClass('shift-hdn') ;
            }
        }
    } ,

    shift_area_total_update : function(idx, shift_duration_min) {
        var shift = this.shifts[idx] ;
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;
        var total_min = 0 ;
        for (var i in this.area_names) {
            var area_name = this.area_names[i].key ;
            var h = parseInt(con.find('input.hour[name="'+area_name+'"]').val() || 0);
            if (h <  0) h = 0 ;
            total_min += h * 60 ;
            var m = parseInt(con.find('input.minute[name="'+area_name+'"]').val() || 0);
            if (m > 59) m = 59 ;
            if (m <  0) m = 0 ;
            total_min += m ;
        }
        if (total_min > shift_duration_min) {
            alert('The total amount of time reported in all areas can not exceeds the duration of the shift.') ;
            return false ;
        }
        con.find('span.hour_minute[name="area"]').text(Fwk.zeroPad(Math.floor(total_min / 60), 2)+':'+Fwk.zeroPad(total_min % 60, 2)) ;
        return true ;
    } ,

    shift_activity_total_update : function(idx, shift_duration_min) {
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;
        var total_min = 0 ;
        for (var i in this.activity_names) {
            var activity_name = this.activity_names[i].key ;
            if (activity_name !== 'other') {
                var h = parseInt(con.find('input.hour[name="'+activity_name+'"]').val() || 0);
                if (h <  0) h = 0 ;
                var m = parseInt(con.find('input.minute[name="'+activity_name+'"]').val() || 0);
                if (m > 59) m = 59 ;
                if (m <  0) m = 0 ;
                var duration_min = h * 60 + m ;
                total_min += duration_min ;
                var percent = shift_duration_min ? Math.floor(100 * duration_min / shift_duration_min) : 0;
                con.find('span.percent[name="'+activity_name+'"]').text(percent) ;
            }
        }
        if (total_min > shift_duration_min) {
            alert('The total amount of time reported for all activities can not exceed the duration of the shift.') ;
            return false ;
        }
        var percent = shift_duration_min ? Math.floor(100 * total_min / shift_duration_min) : 0;
        var other_min = shift_duration_min - total_min ;
        var other_percent = 100 - percent;
        con.find('span.hour_minute[name="other"]').text(Fwk.zeroPad(Math.floor(other_min / 60), 2)+':'+Fwk.zeroPad(other_min % 60, 2)) ;
        con.find('span.percent[name="other"]').text(other_percent) ;
        return true ;
    } ,

    shift_duration_min : function (idx) {
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;

        var begin_day_elem = con.find('input[name="begin_day"]').datepicker() ;
        var begin_h_elem   = con.find('input[name="begin_h"]') ;
        var begin_m_elem   = con.find('input[name="begin_m"]') ;
        var begin_sec      = Math.floor(Date.parse(begin_day_elem.val()) / 1000) + 3600 * parseInt(begin_h_elem.val()) + 60 * parseInt(begin_m_elem.val()) ;

        var end_day_elem  = con.find('input[name="end_day"]').datepicker() ;
        var end_h_elem    = con.find('input[name="end_h"]') ;
        var end_m_elem    = con.find('input[name="end_m"]') ;
        var end_sec       = Math.floor(Date.parse(end_day_elem.val()) / 1000) + 3600 * parseInt(end_h_elem.val()) + 60 * parseInt(end_m_elem.val()) ;
        if (end_sec <= begin_sec) {
            alert('End time must be strictly bigger than the begin time. Please, correct the issue!') ;
            return 0 ;
        }
        var duration_min = Math.floor((end_sec - begin_sec) / 60) ;
        this.shift_area_total_update    (idx, duration_min) ;
        this.shift_activity_total_update(idx, duration_min) ;
        return duration_min ;
    } ,

    shift_register_handlers : function(idx) {
        var that = this ;
        var shift = this.shifts[idx] ;
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;

        var begin_day_elem = con.find('input[name="begin_day"]').datepicker() ;
        var begin_h_elem   = con.find('input[name="begin_h"]') ;
        var begin_m_elem   = con.find('input[name="begin_m"]') ;

        var end_day_elem  = con.find('input[name="end_day"]').datepicker() ;
        var end_h_elem    = con.find('input[name="end_h"]') ;
        var end_m_elem    = con.find('input[name="end_m"]') ;

        function begin_changed(idx) {
            if (!that.shift_duration_min(idx)) {
                begin_day_elem.val(end_day_elem.val()) ;
                begin_h_elem  .val(end_h_elem  .val()) ;
                begin_m_elem  .val(end_m_elem  .val()) ;
            }
        }
        begin_day_elem.change(function () {
            begin_changed(idx) ;
        }) ;
        begin_h_elem.change(function () {
            var h = parseInt($(this).val()) || 0;
            if (h > 23) h = 23 ;
            if (h <  0) h = 0 ;
            $(this).val(h) ;
            begin_changed(idx) ;
        }) ;
        begin_m_elem.change(function () {
            var m = parseInt($(this).val()) || 0;
            if (m > 59) m = 59 ;
            if (m <  0) m = 0 ;
            $(this).val(m) ;
            begin_changed(idx) ;
        }) ;

        function end_changed(idx) {
            if (!that.shift_duration_min(idx)) {
                end_day_elem.val(begin_day_elem.val()) ;
                end_h_elem  .val(begin_h_elem  .val()) ;
                end_m_elem  .val(begin_m_elem  .val()) ;
            }
        }
        end_day_elem.change(function () {
            end_changed(idx) ;
        }) ;
        end_h_elem.change(function () {
            var h = parseInt($(this).val()) || 0;
            if (h > 23) h = 23 ;
            if (h <  0) h = 0 ;
            $(this).val(h) ;
            end_changed(idx) ;
        }) ;
        end_m_elem.change(function () {
            var m = parseInt($(this).val()) || 0;
            if (m > 59) m = 59 ;
            if (m <  0) m = 0 ;
            $(this).val(m) ;
            end_changed(idx) ;
        }) ;

        for (var i in this.area_names) {
            var area_name = this.area_names[i].key ;
            con.find('input.flag[name="'+area_name+'"]').change(function () {
                var area_name = this.name ; // Get area name directly from the element because the one
                                            // defined in the upper scope is wrong (it will always be
                                            // from the last iteration of the loop).
                var h_elem = con.find('input.hour[name="'+area_name+'"]') ;
                var m_elem = con.find('input.minute[name="'+area_name+'"]') ;
                var h = '' ;
                var m = '' ;
                if ($(this).attr('checked')) {
                    var time_down_min = shift.area[area_name].time_down_min;
                    h = Fwk.zeroPad(Math.floor(time_down_min / 60), 2) ;
                    m = Fwk.zeroPad(time_down_min % 60, 2) ;
                    h_elem.removeAttr('disabled') ;
                    m_elem.removeAttr('disabled') ;
                } else {
                    h_elem.attr('disabled', 'disabled') ;
                    m_elem.attr('disabled', 'disabled') ;
                }
                h_elem.val(h) ;
                m_elem.val(m) ;
                con.find('textarea.comment[name="'+area_name+'"]').val(shift.area[area_name].comments) ;
            }) ;
            con.find('input.hour[name="'+area_name+'"]').change(function () {
                var h = parseInt($(this).val()) || 0;
                if (h <  0) h = 0 ;
                if (!that.shift_area_total_update(idx, that.shift_duration_min(idx))) h = 0;
                $(this).val(Fwk.zeroPad(h, 2)) ;
            }) ;
            con.find('input.minute[name="'+area_name+'"]').change(function () {
                var m = parseInt($(this).val()) || 0;
                if (m > 59) m = 59 ;
                if (m <  0) m = 0 ;
                $(this).val(m) ;
                if (!that.shift_area_total_update(idx, that.shift_duration_min(idx))) m = 0;
                $(this).val(Fwk.zeroPad(m, 2)) ;
            }) ;
        }
        for (var i in this.activity_names) {
            var activity_name = this.activity_names[i].key ;
            if (activity_name !== 'other') {
                con.find('input.hour[name="'+activity_name+'"]').change(function () {
                    var h = parseInt($(this).val()) || 0;
                    if (h <  0) h = 0 ;
                    $(this).val(h) ;
                    if (!that.shift_activity_total_update(idx, that.shift_duration_min(idx))) h = 0 ;
                    $(this).val(Fwk.zeroPad(h, 2)) ;
                }) ;
                con.find('input.minute[name="'+activity_name+'"]').change(function () {
                    var m = parseInt($(this).val()) || 0;
                    if (m > 59) m = 59 ;
                    if (m <  0) m = 0 ;
                    if (!that.shift_activity_total_update(idx, that.shift_duration_min(idx))) m = 0 ;
                    $(this).val(Fwk.zeroPad(m, 2)) ;
                }) ;
            }
        }
        con.find('button[name="edit"]').
            button().
            click(function () { that.shift_edit(idx) ; }) ;
        con.find('button[name="save"]').
            button().
            button('disable').
            click(function () { that.shift_edit_save(idx) ; }) ;
        con.find('button[name="cancel"]').
            button().
            button('disable').
            click(function () { that.shift_edit_cancel(idx) ; }) ;
    } ,

    shift_edit : function (idx) {
        var shift = this.shifts[idx] ;
        shift.editing = true ;
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;

        con.find('input[name="begin_day"]').datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val(shift.begin.day) ;
        con.find('input[name="begin_h"]').val(shift.begin.hour) ;
        con.find('input[name="begin_m"]').val(shift.begin.minute) ;
        con.find('input[name="end_day"]').datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val(shift.end.day) ;
        con.find('input[name="end_h"]').val(shift.end.hour) ;
        con.find('input[name="end_m"]').val(shift.end.minute) ;
        con.find('textarea[name="notes"]').val(shift.notes) ;

        for (var i in this.area_names) {
            var area_name = this.area_names[i].key ;
            var h_elem = con.find('input.hour[name="'+area_name+'"]') ;
            var m_elem = con.find('input.minute[name="'+area_name+'"]') ;
            var h = '' ;
            var m = '' ;
            if (shift.area[area_name].problem) {
                con.find('input.flag[name="'+area_name+'"]').attr('checked','checked') ;
                var time_down_min = shift.area[area_name].time_down_min;
                h = Fwk.zeroPad(Math.floor(time_down_min / 60), 2) ;
                m = Fwk.zeroPad(time_down_min % 60, 2) ;
                h_elem.removeAttr('disabled') ;
                m_elem.removeAttr('disabled') ;
            } else {
                con.find('input.flag[name="'+area_name+'"]').removeAttr('checked') ;
                h_elem.attr('disabled', 'disabled') ;
                m_elem.attr('disabled', 'disabled') ;
            }
            con.find('input.hour[name="'+area_name+'"]').val(h) ;
            con.find('input.minute[name="'+area_name+'"]').val(m) ;
            con.find('textarea.comment[name="'+area_name+'"]').val(shift.area[area_name].comments) ;
        }
        for (var i in this.activity_names) {
            var activity_name = this.activity_names[i].key ;
            var duration_min = shift.activity[activity_name].duration_min ;
            var h = Math.floor(duration_min / 60) ;
            var m = duration_min % 60 ;
            var percent = shift.duration_min ? Math.floor(100 * duration_min / shift.duration_min) : 0;
            if (activity_name !== 'other') {
                con.find('input.hour[name="'+activity_name+'"]').val(Fwk.zeroPad(h, 2)) ;
                con.find('input.minute[name="'+activity_name+'"]').val(Fwk.zeroPad(m, 2)) ;
            }
            con.find('span.percent[name="'+activity_name+'"]').text(percent) ;
            con.find('textarea.comment[name="'+activity_name+'"]').val(shift.activity[activity_name].comments) ;
        }

        con.find('button[name="edit"]')  .button('disable') ;
        con.find('button[name="save"]')  .button('enable') ;
        con.find('button[name="cancel"]').button('enable') ;

        con.find('.viewer').removeClass('shift-vis').addClass('shift-hdn') ;
        con.find('.editor').removeClass('shift-hdn').addClass('shift-vis') ;
    } ,

    shift_edit_save : function (idx) {
        var that = this ;
        var shift = this.shifts[idx] ;
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;
        
        var params = {
            shift_id : shift.id ,
            notes    : con.find('textarea[name="notes"]').val() 
        } ;

        var begin_day_elem = con.find('input[name="begin_day"]').datepicker() ;
        var begin_h_elem   = con.find('input[name="begin_h"]') ;
        var begin_m_elem   = con.find('input[name="begin_m"]') ;
        params.begin = begin_day_elem.val() + ' ' + Fwk.zeroPad(parseInt(begin_h_elem.val()), 2) + ':' + Fwk.zeroPad(parseInt(begin_m_elem.val()), 2) + ':00' ;

        var end_day_elem  = con.find('input[name="end_day"]').datepicker() ;
        var end_h_elem    = con.find('input[name="end_h"]') ;
        var end_m_elem    = con.find('input[name="end_m"]') ;
        params.end = end_day_elem.val() + ' ' + Fwk.zeroPad(parseInt(end_h_elem.val()), 2) + ':' + Fwk.zeroPad(parseInt(end_m_elem.val()), 2) + ':00' ;

        var area = {} ;
        for (var i in this.area_names) {
            var area_name = this.area_names[i].key ;
            var h        = parseInt(con.find('input.hour[name="'+area_name+'"]')  .val()) || 0 ;
            var m        = parseInt(con.find('input.minute[name="'+area_name+'"]').val()) || 0 ;
            area[area_name] = {
                problem       : con.find('input.flag[name="'+area_name+'"]').attr('checked') === undefined ? 0 : 1,
                time_down_min : 60 * h + m ,
                comments      : con.find('textarea.comment[name="'+area_name+'"]').val()
            } ;
        }
        params.area = JSON.stringify(area) ;

        var activity = {
            'other' : {
                duration_min : 0 ,
                comments     : con.find('textarea.comment[name="other"]').val()
            }
        } ;
        for (var i in this.activity_names) {
            var activity_name = this.activity_names[i].key ;
            var h             = parseInt(con.find('input.hour[name="'+activity_name+'"]')  .val()) || 0 ;
            var m             = parseInt(con.find('input.minute[name="'+activity_name+'"]').val()) || 0 ;
            if (activity_name !== 'other') {
                activity[activity_name] = {
                    duration_min : 60 * h + m ,
                    comments     : con.find('textarea.comment[name="'+activity_name+'"]').val()                
                } ;
            }
        }
        params.activity = JSON.stringify(activity ) ;

        this.shifts_service (
            '../shiftmgr/ws/shift_save.php', 'POST', params ,
            function (shifts) {
                that.shifts[idx] = shifts[0] ;
                that.display() ;
                that.shift_toggle(idx) ;
            }
        ) ;
    } ,

    shift_edit_cancel : function (idx) {
        this.shift_view(idx) ;
    } ,

    shift_view : function (idx) {
        var shift = this.shifts[idx] ;
        shift.editing = false ;
        var con = this.search_list_elem.find('div.shift-con#'+idx) ;

        con.find('span[name="begin_day"]').text(shift.begin.day) ;
        con.find('span[name="begin_hm"]').text(Fwk.zeroPad(shift.begin.hour, 2)+':'+Fwk.zeroPad(shift.begin.minute, 2)) ;
        con.find('span[name="end_day"]').text(shift.begin.day) ;
        con.find('span[name="end_hm"]').text(Fwk.zeroPad(shift.end.hour, 2)+':'+Fwk.zeroPad(shift.end.minute, 2)) ;
        con.find('div[name="notes"]').html('<pre style="margin:0; padding:2px;">'+shift.notes+'</pre>') ;

        var area_total_min = 0 ;
        for (var i in this.area_names) {
            var area_name = this.area_names[i].key ;
            con.find('div.flag[name="'+area_name+'"]').addClass(shift.area[area_name].problem?'status_red':'status_neutral') ;
            var hm = '' ;
            if (shift.area[area_name].problem) {
                var time_down_min = shift.area[area_name].time_down_min;
                hm = Fwk.zeroPad(Math.floor(time_down_min / 60), 2) + ':' + Fwk.zeroPad(time_down_min % 60, 2) ;
                area_total_min += time_down_min ;
            }
            con.find('span.hour_minute[name="' +area_name+'"]').text(hm) ;
            con.find('div.comment[name="'+area_name+'"]').html('<pre style="margin:0; padding:2px;">'+shift.area[area_name].comments+'</pre>') ;
        }
        con.find('span.hour_minute[name="area"]').text(Fwk.zeroPad(Math.floor(area_total_min / 60), 2)+':'+Fwk.zeroPad(area_total_min % 60, 2)) ;

        var activity_total_min = 0 ;
        for (var i in this.activity_names) {
            var activity_name = this.activity_names[i].key ;
            var duration_min = shift.activity[activity_name].duration_min ;
            var hm = Fwk.zeroPad(Math.floor(duration_min / 60), 2) + ':' + Fwk.zeroPad(duration_min % 60, 2) ;
            var percent = shift.duration_min ? Math.floor(100 * duration_min / shift.duration_min) : 0;
            activity_total_min += duration_min ;
            con.find('span.hour_minute[name="' +activity_name+'"]').text(hm) ;
            con.find('span.percent[name="'+activity_name+'"]').text(percent) ;
            con.find('div.comment[name="'+activity_name+'"]').html('<pre style="margin:0; padding:2px;">'+shift.activity[activity_name].comments+'</pre>') ;
        }

        con.find('button[name="edit"]')  .button('enable') ;
        con.find('button[name="save"]')  .button('disable') ;
        con.find('button[name="cancel"]').button('disable') ;

        con.find('.viewer').removeClass('shift-hdn').addClass('shift-vis') ;
        con.find('.editor').removeClass('shift-vis').addClass('shift-hdn') ;
    }
});
