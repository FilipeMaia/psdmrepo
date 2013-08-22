/**
 * This class representes a title of a row
 * 
 * @see StackOfRows
 * @see ShiftRow
 *
 * @param object data
 * @returns {ShiftTitle}
 */
function ShiftTitle(data) {
    this.data = data ;
    this.html = function (id) {
        var html = '' ;
        switch(id) {
            case 'shift'    : html = '<div class="shift-day">'      + this.data.begin.day + '</div>' ; break ;
            case 'begin'    : html = '<div class="shift-begin">'    + this.data.begin.hm  + '</div>' ; break ;
            case 'end'      : html = '<div class="shift-end">'      + this.data.end.hm    + '</div>' ; break ;
            case 'duration' : html = '<div class="shift-duration">' + this.data.duration  + '</div>' ; break ;

            case 'stopper'  :
                var stopper_percent = this.data.duration_min ? Math.floor(100 * this.data.stopper_min / this.data.duration_min) : 0 ;
                html = '<div class="shift-stopper" >' + (stopper_percent ? stopper_percent+'%' : '&nbsp;') + '</div>' ; break ;

            case 'door'     :
                var door_open_percent = this.data.duration_min ? Math.floor(100 * (this.data.duration_min - this.data.door_min) / this.data.duration_min) : 0 ;
                if (door_open_percent < 0) door_open_percent = 0 ;
                html = '  <div class="shift-door"    >' + (door_open_percent ? door_open_percent+'%' : '&nbsp;') + '</div>' ; break ;

            case 'FEL'      :
            case 'BMLN'     :
            case 'CTRL'     :
            case 'DAQ'      :
            case 'LASR'     :
            case 'TIME'     :
            case 'HALL'     :
            case 'OTHR'     : html = '<div class="shift-area'+id+'" ><div class="status-'+(this.data.area[id].problems ?'red':'neutral')+'"></div></div>' ; break ;

            case 'editor'   : html = '<div class="shift-editor"   >&nbsp;' + this.data.editor   + '</div>' ; break ;
            case 'modified' : html = '<div class="shift-modified" >&nbsp;' + this.data.modified + '</div>' ; break ;
        }
        return html ;
    } ;
}

/**
 * This class representes a bowdy of a row
 * 
 * @see StackOfRows
 * @see ShiftRow
 *
 * @param object parent
 * @param object data
 * @param boolean can_edit
 * @returns {ShiftBody}
 */
function ShiftBody (parent, data, can_edit) {

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    Widget.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent = parent ;
    this.data = data ;
    this.can_edit = can_edit ;

    // ----------------------------
    // Static variables & functions
    // ----------------------------

    var area_names = [
        {key: 'FEL' , name: 'FEL',        description: 'To report problems with the machine, operations, FEE, etc.'} ,
        {key: 'BMLN', name: 'Beamline',   description: 'To report problems with the photon beamline instrument\n including HPS and PPS problems'} ,
        {key: 'CTRL', name: 'Controls',   description: 'To report problems specific to controls like motors, cameras,\n MPS, laser controls, etc. '} ,
        {key: 'DAQ' , name: 'DAQ',        description: 'To report DAQ computer, data transfer and device problems'} ,
        {key: 'LASR', name: 'Laser',      description: 'To report problems with the laser, (not laser controls)'} ,
        {key: 'TIME', name: 'Timing',     description: 'To report problems with the timing system including: EVR triggers,\n RF, LBL Timing System, fstiming interface (laser related problems\n should be reported under laser)'} ,
        {key: 'HALL', name: 'Hutch/Hall', description: 'To report problem with the hutch like: PCW, temperature, setup space,\n common stock, etc.  Note that this could be confused with the overall\n name of this section of the form.'} ,
        {key: 'OTHR', name: 'Other',      description: 'Any other areas that might have problems can be addressed'}
    ] ;

    var allocation_names = [
        {key: 'tuning'   , name: 'Tuning',       description: 'This is machine tuning time'} ,
        {key: 'alignment', name: 'Alignment',    description: 'This is time spent aligning, calibrating,  turning the photon\n instrumentation'} ,
        {key: 'daq',       name: 'Data Taking',  description: 'This is time spent taking data that can be used in publication'} ,
        {key: 'access',    name: 'Hutch Access', description: 'This is time spent in the hutch for sample changes, laser tuning,\n trouble shooting and the like'} ,
        {key: 'other',     name: 'Other',        description: 'This is any other circumstance: machine downtime, extended\n specific activities, etc. This will be atomatically calculated\n to absorb the remaining time left of the shift.'}
    ] ;

    // ------------------------------------------------
    // Override event handler defined in thw base class
    // ------------------------------------------------

    this.is_rendered = false ;

    this.render = function () {

        if (this.is_rendered) return ;
        this.is_rendered = true ;

        var that = this ;
        var shift = this.data ;

        var stopper_h = Math.floor(shift.stopper_min / 60) ;
        var stopper_m = shift.stopper_min % 60 ;
        var stopper_hm = Fwk.zeroPad(stopper_h, 2)+':'+Fwk.zeroPad(stopper_m, 2) ;
        var stopper_percent = shift.duration_min ? Math.floor(100 * (shift.stopper_min / shift.duration_min)) : 0 ;

        var door_open_min = shift.duration_min - shift.door_min ;
        if (door_open_min < 0) door_open_min = 0 ;
        var door_open_h = Math.floor(door_open_min / 60) ;
        var door_open_m = door_open_min % 60 ;
        var door_open_hm = Fwk.zeroPad(door_open_h, 2)+':'+Fwk.zeroPad(door_open_m, 2) ;
        var door_open_percent = shift.duration_min ? Math.floor(100 * door_open_min / shift.duration_min) : 0 ;

        var html =
'<div>' +

'  <div style="float:left;">' +
'    <table style="font-size:90%;"><tbody>' +
'      <tr class="shift-active-row" >' +
'        <td class="annotated shift-grid-hdr " valign="center" data="This is the nominal start data & time of the shift" >Begin:</td>' +
'        <td class="shift-grid-val " valign="center" >' +
'          <div class="viewer" >' +
'            <span name="begin-day" ></span>' +
'            <span name="begin-hm" style="font-weight: bold; padding-left:10px;" ></span>' +
'          </div>' +
'          <div class="editor" >' +
'            <input name="begin-day" type="text" size=10 />' +
'            <input name="begin-h"   type="text" size=2 />' +
'            <input name="begin-m"   type="text" size=2 />' +
'          </div></td>' +
'        <td class="annotated shift-grid-hdr " valign="center" data="This is the integrated time that the stopper is out\n during the shift" >Stopper Out:</td>' +
'        <td class="shift-grid-val " valign="center" ><span name="stopper-hm">'+stopper_hm+'</span></td>' +
'        <td class="shift-grid-val " valign="center" ><span name="stopper-percent">( '+stopper_percent+' % )</span></td>' +
'      </tr>' +
'      <tr class="shift-active-row">' +
'        <td class="annotated shift-grid-hdr " valign="center" data="This should default to 12 hours after the Start Shift time,\n but it should also be editable for special circumstances.\n This will be used to determine the overall shift duration\n which is an important number in the shift usage section." >End:</td>' +
'        <td class="shift-grid-val " valign="center" >' +
'          <div class="viewer" >' +
'            <span name="end-day" ></span>' +
'            <span name="end-hm" style="font-weight: bold; padding-left:10px;" ></span>' +
'          </div>' +
'          <div class="editor" >' +
'            <input name="end-day" type="text" size=10 />' +
'            <input name="end-h"   type="text" size=2  />' +
'            <input name="end-m"   type="text" size=2  />' +
'          </div></td>' +
'        <td class="annotated shift-grid-hdr " valign="center" data="This is the integrated time that the hutch door is open" >Door Open:</td>' +
'        <td class="shift-grid-val " valign="center" ><span name="door-hm">'+door_open_hm+'</span></td>' +
'        <td class="shift-grid-val " valign="center" ><span name="door-percent">( '+door_open_percent+' % )</span></td>' +
'      </tr>' +
'      <tr class="shift-active-row">' +
'        <td class="annotated shift-grid-hdr " valign="center" data="general notes on the shift" >Notes:</td>' +
'        <td class="shift-grid-val " valign="center" colspan="4" >'+
'          <div      class="viewer notes"      name="notes" ></div>' +
'          <textarea class="annotated editor"  name="notes" rows="4" cols=62 data="general notes on the shift (if any)" ></textarea></td>' +
'      </tr>' +
'    </tbody></table>' +
'  </div>' +
'  <div style="float:left; margin-left:40px; padding-top:5px;">' ;
        var area_title_data =
'The Area Evaluation section is intended to identify problem areas.\n' +
'These may be major problems like a controls motor failure that prevented\n' +
'data from being taken and required immediate attention or a recurring but\n' +
'non critical problem like occasional DAQ crashes. In all cases specific problems\n' +
'listed will not be resolved with this form, but treated only in agitate with other input.' ;
        var allocation_title_data =
'Time Use Allocation is intended to understand how experiments are\n' +
'performed.  It is not assumed that experiment spending the most time taking\n' +
'data will generate the best science.   For example, if it is found that extensive\n' +
'time is spent on alignment it may be decided that time and effort should be\n' +
'directed toward improved alignment software and feedback systems.';
        var history_title_data =
'This panel is for information purposes only. It will display the known\n' +
'history of modifications made to the shift.' ;
        if (this.can_edit)
            html +=
'    <button class="annotated" name="edit"   data="Switch to the editing mode" style="color:red; font-weight:bold;" >Edit</button>' +
'    <button class="annotated" name="save"   data="Submit modifications and switch back to the vieweing mode to edit">Save</button>' +
'    <button class="annotated" name="cancel" data="Discard modifications and revert back to the viewing mode">Cancel</button>' +
'    <button class="annotated" name="delete" data="Delete the shift.\n Note that an additional confirmation will be requested\n before deliting the shift." style="margin-left:40px; color:red; font-weight:bold;" >Delete Shift</button>' ;
        var area_downtime_title =
'This field only appears when there is a problem. If the problem is minor (taking\n' + 
'less than 30 min of down time) it does not need to be completed. If the problem\n' +
'accounts for over 30min of down time over the entire shift a value for the time (in hours)\n' +
'should be entered. The comment field remains optional.\n' +
'Note the correct format for entering the time: HH:MM' ;
        html +=
'  </div>' +
'  <div style="clear:both;"></div>' +

'  <div id="tabs" style="font-size:12px; margin-top:20px;">' +
'    <ul>' +
'      <li><a class="tab-hdr" href="#area-evaluation" data="' + area_title_data       + '" >Area Evaluation</a></li>' +
'      <li><a class="tab-hdr" href="#time-allocation" data="' + allocation_title_data + '" >Time Use Allocation</a></li>' +
'      <li><a class="tab-hdr" href="#history"         data="' + history_title_data    + '" >Shift History</a></li>' +
'    </ul>' +
'    <div id="area-evaluation" >' +
'      <div class="shift-area-evaluation-con" >' +
'        <table><tbody>' +
'          <tr>' +
'            <td class="shift-table-hdr " >Area</td>' +
'            <td class="shift-table-hdr " >Issues?</td>' +
'            <td class="shift-table-hdr annotated" data="'+area_downtime_title+'" >Downtime</td>' +
'            <td class="shift-table-hdr " >Comments</td>' +
'          </tr>' ;
        var first = true ;
        for (var i in area_names) {
            var classes = 'annotated shift-table-val'+(first?'-first':'') ;
            first = false ;
            var area_name = area_names[i].key ;
            html +=
'          <tr class="shift-active-row" >' +
'            <td class="'+classes+' shift-table-val-hdr " valign="top" data="'+area_names[i].description+'" >'+area_names[i].name+'</td>' +
'            <td class="'+classes+' " valign="top">' +
'              <div   class="viewer flag" name="'+area_name+'" /></div>' +
'              <input class="editor flag" name="'+area_name+'" type="checkbox" /></td>' +
'            <td class="'+classes+' " valign="top">'+
'              <div class="viewer">' +
'                <span class="hour-minute" name="'+area_name+'" ></span>' +
'              </div>' +
'              <div class="editor">'+
'                <input class="annotated hour"   name="'+area_name+'" type="text" size=1 data="hours: 0.." />' +
'                <input class="annotated minute" name="'+area_name+'" type="text" size=1 data="minutes: 0..59" />' +
'              </div></td>' +
'            <td class="'+classes+' ">' +
'              <div      class="viewer comment" name="'+area_name+'" ></div>' +
'              <textarea class="annotated editor comment" name="'+area_name+'" rows="2" cols=48 data="explain the problem" ></textarea></td>' +
'          </tr>' ;
        }
        var time_allocation_title =
'Time should be accounted for on a 0.5 hour increment or less.\n' +
'Note the correct format for entering the time: HH:MM' ;
        html +=
'          <tr>' +
'            <td class="shift-table-tot " valign="top">Total</td>' +
'            <td class="shift-table-tot " valign="top">&nbsp;</td>' +
'            <td class="shift-table-tot " valign="top"><span class="hour-minute" name="area"></td>' +
'            <td class="shift-table-tot " valign="top">&nbsp;</td>' +
'          </tr>' +
'        </tbody></table>' +
'      </div>' +
'    </div>' +
'    <div id="time-allocation" >' +
'      <div class="shift-time-allocation-con" >' +
'        <table><tbody>' +
'          <tr>' +
'            <td class="shift-table-hdr " >Activity</td>' +
'            <td class="shift-table-hdr annotated" data="'+time_allocation_title+'" >Time spent</td>' +
'            <td class="shift-table-hdr " align="right" >%</td>' +
'            <td class="shift-table-hdr " >Comments</td>' +
'          </tr>' ;
        first = true ;
        for (var i in this.allocation_names) {
            var classes = 'annotated shift-table-val'+(first?'-first':'') ;
            first = false ;
            var allocation_name = this.allocation_names[i].key ;
            html +=
'          <tr class="shift-active-row" >' +
'            <td class="'+classes+' shift-table-val-hdr " valign="top" data="'+this.allocation_names[i].description+'" >'+this.allocation_names[i].name+'</td>' +
'            <td class="'+classes+' " valign="top">' ;
            if (allocation_name === 'other') {
                html +=
'              <div>' +
'                <span class="hour-minute" name="'+allocation_name+'" ></span>' +
'              </div></td>' ;
            } else {
                html +=
'              <div class="viewer">' +
'                <span class="hour-minute" name="'+allocation_name+'" ></span>' +
'              </div>' +
'              <div class="editor">' +
'                <input class="annotated hour"   name="'+allocation_name+'" type="text" size=1 data="hours: 0.." />' +
'                <input class="annotated minute" name="'+allocation_name+'" type="text" size=1 data="minutes: 0..59" />' +
'              </div></td>' ;
            }
            html +=
'            <td class="'+classes+' " align="right" valign="top"><span class="percent" name="'+allocation_name+'"></span></td>' +
'            <td class="'+classes+' ">' +
'              <div      class="viewer comment" name="'+allocation_name+'" ></div>' +
'              <textarea class="annotated editor comment" name="'+allocation_name+'" rows="3" cols=48 data="explain the time allocation" ></textarea></td>' +
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
'    <div id="history" >' +
'      <div class="shift-history-con" >' +
'        Loading...' +
'      </div>' +
'    </div>' +
'  </div>' +
'</div>' ;
        this.container.html(html) ;
        this.container.find('div#tabs').tabs({
            activate: function( event, ui ) {
                if (ui.newPanel.attr('id') === 'history') {
                    that.shift_load_history() ;
                }
            }
        }) ;
        this.shift_register_handlers() ;
        this.shift_view() ;
    } ;

    // --------------------
    // Own data and methods
    // --------------------

    this.shift_register_handlers = function() {
        var that = this ;
        var shift = this.data ;
        var con = this.container ;

        var begin_day_elem = con.find('input[name="begin-ay"]').datepicker() ;
        var begin_h_elem   = con.find('input[name="begin-h"]') ;
        var begin_m_elem   = con.find('input[name="begin-m"]') ;

        var end_day_elem  = con.find('input[name="end-day"]').datepicker() ;
        var end_h_elem    = con.find('input[name="end-h"]') ;
        var end_m_elem    = con.find('input[name="end-m"]') ;

        function begin_changed() {
            if (!that.shift_duration_min()) {
                begin_day_elem.val(end_day_elem.val()) ;
                begin_h_elem  .val(end_h_elem  .val()) ;
                begin_m_elem  .val(end_m_elem  .val()) ;
            }
        }
        begin_day_elem.change(function () {
            begin_changed() ;
        }) ;
        begin_h_elem.change(function () {
            var h = parseInt($(this).val()) || 0;
            if (h > 23) h = 23 ;
            if (h <  0) h = 0 ;
            $(this).val(h) ;
            begin_changed() ;
        }) ;
        begin_m_elem.change(function () {
            var m = parseInt($(this).val()) || 0;
            if (m > 59) m = 59 ;
            if (m <  0) m = 0 ;
            $(this).val(m) ;
            begin_changed() ;
        }) ;

        function end_changed() {
            if (!that.shift_duration_min()) {
                end_day_elem.val(begin_day_elem.val()) ;
                end_h_elem  .val(begin_h_elem  .val()) ;
                end_m_elem  .val(begin_m_elem  .val()) ;
            }
        }
        end_day_elem.change(function () {
            end_changed() ;
        }) ;
        end_h_elem.change(function () {
            var h = parseInt($(this).val()) || 0;
            if (h > 23) h = 23 ;
            if (h <  0) h = 0 ;
            $(this).val(h) ;
            end_changed() ;
        }) ;
        end_m_elem.change(function () {
            var m = parseInt($(this).val()) || 0;
            if (m > 59) m = 59 ;
            if (m <  0) m = 0 ;
            $(this).val(m) ;
            end_changed() ;
        }) ;

        for (var i in area_names) {
            var area_name = area_names[i].key ;
            con.find('input.flag[name="'+area_name+'"]').change(function () {
                var area_name = this.name ; // Get area name directly from the element because the one
                                            // defined in the upper scope is wrong (it will always be
                                            // from the last iteration of the loop).
                var h_elem = con.find('input.hour[name="'+area_name+'"]') ;
                var m_elem = con.find('input.minute[name="'+area_name+'"]') ;
                var h = '' ;
                var m = '' ;
                if ($(this).attr('checked')) {
                    var downtime_min = shift.area[area_name].downtime_min;
                    h = Fwk.zeroPad(Math.floor(downtime_min / 60), 2) ;
                    m = Fwk.zeroPad(downtime_min % 60, 2) ;
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
                if (!that.shift_area_total_update(that.shift_duration_min())) h = 0;
                $(this).val(Fwk.zeroPad(h, 2)) ;
            }) ;
            con.find('input.minute[name="'+area_name+'"]').change(function () {
                var m = parseInt($(this).val()) || 0;
                if (m > 59) m = 59 ;
                if (m <  0) m = 0 ;
                $(this).val(m) ;
                if (!that.shift_area_total_update(that.shift_duration_min())) m = 0;
                $(this).val(Fwk.zeroPad(m, 2)) ;
            }) ;
        }
        for (var i in this.allocation_names) {
            var allocation_name = this.allocation_names[i].key ;
            if (allocation_name !== 'other') {
                con.find('input.hour[name="'+allocation_name+'"]').change(function () {
                    var h = parseInt($(this).val()) || 0;
                    if (h <  0) h = 0 ;
                    $(this).val(h) ;
                    if (!that.shift_allocation_total_update(that.shift_duration_min())) h = 0 ;
                    $(this).val(Fwk.zeroPad(h, 2)) ;
                }) ;
                con.find('input.minute[name="'+allocation_name+'"]').change(function () {
                    var m = parseInt($(this).val()) || 0;
                    if (m > 59) m = 59 ;
                    if (m <  0) m = 0 ;
                    if (!that.shift_allocation_total_update(that.shift_duration_min())) m = 0 ;
                    $(this).val(Fwk.zeroPad(m, 2)) ;
                }) ;
            }
        }
        con.find('button[name="edit"]').
            button().
            click(function () { that.shift_edit() ; }) ;
        con.find('button[name="save"]').
            button().
            button('disable').
            click(function () { that.shift_edit_save() ; }) ;
        con.find('button[name="cancel"]').
            button().
            button('disable').
            click(function () { that.shift_edit_cancel() ; }) ;
        con.find('button[name="delete"]').
            button().
            click(function () {
                Fwk.ask_yes_no (
                    "Confirm Shift Delete" ,
                    'Are you really going to delete that shift?' ,
                    function () { that.parent.shift_delete(that.data.id) ; }) ; }) ;
    } ;

    this.shift_area_total_update = function(shift_duration_min) {
        var con = this.container ;
        var total_min = 0 ;
        for (var i in area_names) {
            var area_name = area_names[i].key ;
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
        con.find('span.hour-minute[name="area"]').text(Fwk.zeroPad(Math.floor(total_min / 60), 2)+':'+Fwk.zeroPad(total_min % 60, 2)) ;
        return true ;
    } ;

    this.shift_allocation_total_update = function(shift_duration_min) {
        var con = this.container ;
        var total_min = 0 ;
        for (var i in this.allocation_names) {
            var allocation_name = this.allocation_names[i].key ;
            if (allocation_name !== 'other') {
                var h = parseInt(con.find('input.hour[name="'+allocation_name+'"]').val() || 0);
                if (h <  0) h = 0 ;
                var m = parseInt(con.find('input.minute[name="'+allocation_name+'"]').val() || 0);
                if (m > 59) m = 59 ;
                if (m <  0) m = 0 ;
                var duration_min = h * 60 + m ;
                total_min += duration_min ;
                var percent = shift_duration_min ? Math.floor(100 * duration_min / shift_duration_min) : 0;
                con.find('span.percent[name="'+allocation_name+'"]').text(percent) ;
            }
        }
        if (total_min > shift_duration_min) {
            alert('The total amount of time reported for all activities can not exceed the duration of the shift.') ;
            return false ;
        }
        var percent = shift_duration_min ? Math.floor(100 * total_min / shift_duration_min) : 0;
        var other_min = shift_duration_min - total_min ;
        var other_percent = 100 - percent;
        con.find('span.hour-minute[name="other"]').text(Fwk.zeroPad(Math.floor(other_min / 60), 2)+':'+Fwk.zeroPad(other_min % 60, 2)) ;
        con.find('span.percent[name="other"]').text(other_percent) ;
        return true ;
    } ;

    this.shift_duration_min = function () {
        var con = this.container ;

        var begin_day_elem = con.find('input[name="begin-day"]').datepicker() ;
        var begin_h_elem   = con.find('input[name="begin-h"]') ;
        var begin_m_elem   = con.find('input[name="begin-m"]') ;
        var begin_sec      = Math.floor(Date.parse(begin_day_elem.val()) / 1000) + 3600 * parseInt(begin_h_elem.val()) + 60 * parseInt(begin_m_elem.val()) ;

        var end_day_elem  = con.find('input[name="end-day"]').datepicker() ;
        var end_h_elem    = con.find('input[name="end-h"]') ;
        var end_m_elem    = con.find('input[name="end-m"]') ;
        var end_sec       = Math.floor(Date.parse(end_day_elem.val()) / 1000) + 3600 * parseInt(end_h_elem.val()) + 60 * parseInt(end_m_elem.val()) ;
        if (end_sec <= begin_sec) {
            alert('End time must be strictly bigger than the begin time. Please, correct the issue!') ;
            return 0 ;
        }
        var duration_min = Math.floor((end_sec - begin_sec) / 60) ;
        this.shift_area_total_update    (duration_min) ;
        this.shift_allocation_total_update(duration_min) ;
        return duration_min ;
    } ;

    this.shift_edit = function () {

        var shift = this.data ;
        shift.editing = true ;
        var con = this.container ;

        con.find('input[name="begin-day"]').datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val(shift.begin.day) ;
        con.find('input[name="begin-h"]').val(shift.begin.hour) ;
        con.find('input[name="begin-m"]').val(shift.begin.minute) ;
        con.find('input[name="end-day"]').datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val(shift.end.day) ;
        con.find('input[name="end-h"]').val(shift.end.hour) ;
        con.find('input[name="end-m"]').val(shift.end.minute) ;
        con.find('textarea[name="notes"]').val(shift.notes) ;

        for (var i in area_names) {
            var area_name = area_names[i].key ;
            var h_elem = con.find('input.hour[name="'+area_name+'"]') ;
            var m_elem = con.find('input.minute[name="'+area_name+'"]') ;
            var h = '' ;
            var m = '' ;
            if (shift.area[area_name].problems) {
                con.find('input.flag[name="'+area_name+'"]').attr('checked','checked') ;
                var downtime_min = shift.area[area_name].downtime_min;
                h = Fwk.zeroPad(Math.floor(downtime_min / 60), 2) ;
                m = Fwk.zeroPad(downtime_min % 60, 2) ;
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
        for (var i in this.allocation_names) {
            var allocation_name = this.allocation_names[i].key ;
            var duration_min = shift.allocation[allocation_name].duration_min ;
            var h = Math.floor(duration_min / 60) ;
            var m = duration_min % 60 ;
            var percent = shift.duration_min ? Math.floor(100 * duration_min / shift.duration_min) : 0;
            if (allocation_name !== 'other') {
                con.find('input.hour[name="'+allocation_name+'"]').val(Fwk.zeroPad(h, 2)) ;
                con.find('input.minute[name="'+allocation_name+'"]').val(Fwk.zeroPad(m, 2)) ;
            }
            con.find('span.percent[name="'+allocation_name+'"]').text(percent) ;
            con.find('textarea.comment[name="'+allocation_name+'"]').val(shift.allocation[allocation_name].comments) ;
        }

        con.find('button[name="edit"]')  .button('disable') ;
        con.find('button[name="save"]')  .button('enable') ;
        con.find('button[name="cancel"]').button('enable') ;
        con.find('button[name="delete"]').button('enable') ;

        con.find('.viewer').removeClass('shift-vis').addClass('shift-hdn') ;
        con.find('.editor').removeClass('shift-hdn').addClass('shift-vis') ;
    } ;

    this.shift_edit_save = function () {
        var that = this ;
        var shift = this.data ;
        var con = this.container ;
        
        var params = {
            shift_id : shift.id ,
            notes    : con.find('textarea[name="notes"]').val() 
        } ;

        var begin_day_elem = con.find('input[name="begin-day"]').datepicker() ;
        var begin_h_elem   = con.find('input[name="begin-h"]') ;
        var begin_m_elem   = con.find('input[name="begin-m"]') ;
        params.begin = begin_day_elem.val() + ' ' + Fwk.zeroPad(parseInt(begin_h_elem.val()), 2) + ':' + Fwk.zeroPad(parseInt(begin_m_elem.val()), 2) + ':00' ;

        var end_day_elem  = con.find('input[name="end-day"]').datepicker() ;
        var end_h_elem    = con.find('input[name="end-h"]') ;
        var end_m_elem    = con.find('input[name="end-m"]') ;
        params.end = end_day_elem.val() + ' ' + Fwk.zeroPad(parseInt(end_h_elem.val()), 2) + ':' + Fwk.zeroPad(parseInt(end_m_elem.val()), 2) + ':00' ;

        var area = {} ;
        for (var i in area_names) {
            var area_name = area_names[i].key ;
            var h         = parseInt(con.find('input.hour[name="'+area_name+'"]')  .val()) || 0 ;
            var m         = parseInt(con.find('input.minute[name="'+area_name+'"]').val()) || 0 ;
            area[area_name] = {
                problems     : con.find('input.flag[name="'+area_name+'"]').attr('checked') === undefined ? 0 : 1,
                downtime_min : 60 * h + m ,
                comments     : con.find('textarea.comment[name="'+area_name+'"]').val()
            } ;
        }
        params.area = JSON.stringify(area) ;

        var allocation = {
            'other' : {
                duration_min : 0 ,
                comments     : con.find('textarea.comment[name="other"]').val()
            }
        } ;
        for (var i in this.allocation_names) {
            var allocation_name = this.allocation_names[i].key ;
            var h             = parseInt(con.find('input.hour[name="'+allocation_name+'"]')  .val()) || 0 ;
            var m             = parseInt(con.find('input.minute[name="'+allocation_name+'"]').val()) || 0 ;
            if (allocation_name !== 'other') {
                allocation[allocation_name] = {
                    duration_min : 60 * h + m ,
                    comments     : con.find('textarea.comment[name="'+allocation_name+'"]').val()                
                } ;
            }
        }
        params.allocation = JSON.stringify(allocation ) ;

        this.parent.shift_update (
            shift.id ,
            '../shiftmgr/ws/shift_save.php' ,
            'POST' ,
            params
        ) ;
    } ;

    this.shift_edit_cancel = function () {
        this.shift_view() ;
    } ,

    this.shift_view = function () {
        var shift = this.data ;
        shift.editing = false ;
        var con = this.container ;

        con.find('span[name="begin-day"]').text(shift.begin.day) ;
        con.find('span[name="begin-hm"]').text(Fwk.zeroPad(shift.begin.hour, 2)+':'+Fwk.zeroPad(shift.begin.minute, 2)) ;
        con.find('span[name="end-day"]').text(shift.end.day) ;
        con.find('span[name="end-hm"]').text(Fwk.zeroPad(shift.end.hour, 2)+':'+Fwk.zeroPad(shift.end.minute, 2)) ;
        con.find('div[name="notes"]').html('<pre>'+shift.notes+'</pre>') ;

        var area_total_min = 0 ;
        for (var i in area_names) {
            var area_name = area_names[i].key ;
            con.find('div.flag[name="'+area_name+'"]').addClass(shift.area[area_name].problems?'status-red':'status-neutral') ;
            var hm = '' ;
            if (shift.area[area_name].problems) {
                var downtime_min = shift.area[area_name].downtime_min;
                hm = Fwk.zeroPad(Math.floor(downtime_min / 60), 2) + ':' + Fwk.zeroPad(downtime_min % 60, 2) ;
                area_total_min += downtime_min ;
            }
            con.find('span.hour-minute[name="' +area_name+'"]').text(hm) ;
            con.find('div.comment[name="'+area_name+'"]').html('<pre>'+shift.area[area_name].comments+'</pre>') ;
        }
        con.find('span.hour-minute[name="area"]').text(Fwk.zeroPad(Math.floor(area_total_min / 60), 2)+':'+Fwk.zeroPad(area_total_min % 60, 2)) ;

        var allocation_total_min = 0 ;
        for (var i in this.allocation_names) {
            var allocation_name = this.allocation_names[i].key ;
            var duration_min = shift.allocation[allocation_name].duration_min ;
            var hm = Fwk.zeroPad(Math.floor(duration_min / 60), 2) + ':' + Fwk.zeroPad(duration_min % 60, 2) ;
            var percent = shift.duration_min ? Math.floor(100 * duration_min / shift.duration_min) : 0;
            allocation_total_min += duration_min ;
            con.find('span.hour-minute[name="' +allocation_name+'"]').text(hm) ;
            con.find('span.percent[name="'+allocation_name+'"]').text(percent) ;
            con.find('div.comment[name="'+allocation_name+'"]').html('<pre>'+shift.allocation[allocation_name].comments+'</pre>') ;
        }

        con.find('button[name="edit"]')  .button('enable') ;
        con.find('button[name="save"]')  .button('disable') ;
        con.find('button[name="cancel"]').button('disable') ;

        con.find('.viewer').removeClass('shift-hdn').addClass('shift-vis') ;
        con.find('.editor').removeClass('shift-vis').addClass('shift-hdn') ;
    } ;

    this.shift_load_history = function () {
        var shift = this.data ;
        if (!shift.history_table) {
            shift.history_table = new Table (
                this.container.find('div#history').find('div.shift-history-con') ,
                [
                    { name: 'modified'  },
                    { name: 'editor'  },
                    { name: 'event' },
                    { name: 'area/allocation' },
                    { name: 'parameter' },
                    { name: 'old value', sorted: false },
                    { name: 'new value', sorted: false }
                ] ,
                null ,                              // data will be loaded dynamically
                {   default_sort_column:  0 ,
                    default_sort_forward: false ,
                    text_when_empty:      'Loading...'
                } ,
                Fwk.config_handler('History', 'shifts')
            ) ;
            shift.history_table.display() ;

            var params = {
                shift_id : shift.id
            } ;
            var jqXHR = $.get('../shiftmgr/ws/history_get.php', params, function (data) {
                if (data.status !== 'success') {
                    Fwk.report_error(data.message, null) ;
                    return ;
                }
                var rows = [] ;
                for (var i in data.history) {
                    var event = data.history[i] ;
                    var row = [
                        event.modified.full ,
                        event.editor ,
                        event.operation+' '+event.scope ,
                        event.scope2 ,
                        event.parameter ,
                        '<div class="comment"><pre>'+event.old_value+'</pre></div>' ,
                        '<div class="comment"><pre>'+event.new_value+'</pre></div>'
                    ] ;
                    rows.push (row) ;
                }
                shift.history_table.load(rows) ;
            } ,
            'JSON').error(function () {
                Fwk.report_error('failed to obtain the cable history because of: '+jqXHR.statusText, null) ;
                return ;
            }) ;
        }
    }
}
define_class (ShiftBody, Widget, {}, {}) ;

/**
 * This class binds the data with the row interface as requited by the StackOfRows class
 *
 * @see StackOfRows
 *
 * @param object parent
 * @param object data
 * @param boolean can_edit
 * @returns {ShiftRow}
 */
function ShiftRow (parent, data, can_edit) {

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackRowData.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    if (!data) throw new Error('ShiftRow:constructor() data is not defined') ;
    this.data = data ;

    this.title = new ShiftTitle(this.data) ;
    this.body  = new ShiftBody (parent, this.data, can_edit) ;

    // --------------------------------------------------
    // Override methods handler defined in the base class
    // --------------------------------------------------

    this.is_locked = function () {
        return this.data.is_editing ;
    } ;
}
define_class(ShiftRow, StackRowData, {}, {}) ;

/**
 * The main class representing instrument-level reports.
 *
 * @param string instr_name
 * @param boolean can_edit
 * @returns {Reports}
 */
function Reports (instr_name, can_edit) {

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    FwkApplication.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.instr_name = instr_name ;
    this.can_edit = can_edit ? 1 : 0 ;

    // ------------------------------------------------
    // Override event handler defined in the base class
    // ------------------------------------------------

    this.on_activate = function() {
        this.on_update() ;
    } ;

    this.on_deactivate = function() {
        ;
    } ;

    this.on_update = function () {
        if (this.active) {
            this.init() ;
            if (this.update_allowed()) {
                this.search() ;
            }
        }
    } ;

    // --------------------
    // Own data and methods
    // --------------------

    this.shifts = [] ;
    this.shifts_stack = null ;

    this.update_allowed = function () {
        if (this.shifts_stack) return !this.shifts_stack.is_locked() ;
        return true ;
    } ;

    this.is_initialized = false ;

    this.init = function () {

        var that = this ;

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        var ctrl_elem = this.container.find('#shifts-search-controls') ;

        this.ctrl_range_elem       = ctrl_elem.find('select[name="range"]') ;
        this.ctrl_begin_elem       = ctrl_elem.find('input[name="begin"]') ;
        this.ctrl_end_elem         = ctrl_elem.find('input[name="end"]') ;
        this.ctrl_display_all_elem = ctrl_elem.find('input[name="display-all"]') ;

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
        this.ctrl_display_all_elem.change(function () {
            that.search() ;
        }) ;

        this.search_info_elem = this.container.find('#shifts-search-info') ;
        this.shifts_search_display = this.container.find('#shifts-search-display') ;

        this.shifts_stack = new StackOfRows ([
            {id: 'shift',    title: 'Shift',    width: 100} ,
            {id: 'begin',    title: 'begin',    width:  50} ,
            {id: 'end',      title: 'end',      width:  50} ,
            {id: 'duration', title: '&Delta;t', width:  50} ,
            {id: '|' } ,
            {id: 'stopper',  title: 'Stopper',  width:  60} ,
            {id: 'door',     title: 'Door',     width:  40} ,
            {id: '|' } ,
            {id: 'FEL',      title: 'FEL',      width:  50} ,
            {id: 'BMLN',     title: 'BMLN',     width:  50} ,
            {id: 'CTRL',     title: 'CTRL',     width:  50} ,
            {id: 'DAQ',      title: 'DAQ',      width:  50} ,
            {id: 'LASR',     title: 'LASR',     width:  50} ,
            {id: 'TIME',     title: 'TIME',     width:  50} ,
            {id: 'HALL',     title: 'HALL',     width:  50} ,
            {id: 'OTHR',     title: 'OTHR',     width:  50} ,
            {id: '|' } ,
            {id: 'editor',   title: 'Editor',   width:  80} ,
            {id: 'modified', title: 'Modified', width: 240}
        ] , null, {
            theme: 'stack-theme-large14 stack-theme-mustard'
        }) ;
        
        this.init_new_shift() ;
    } ;

    this.init_new_shift = function () {

        var that = this ;

        var new_shift_ctrl_elem = this.container.find('#new-shift-controls') ;
        var new_shift           = new_shift_ctrl_elem.find('button[name="new-shift"]').button() ;
        var new_shift_save      = new_shift_ctrl_elem.find('button[name="save"]').button() ;
        var new_shift_cancel    = new_shift_ctrl_elem.find('button[name="cancel"]').button() ;
        var new_shift_con       = new_shift_ctrl_elem.find('#new-shift-con') ;

        var new_shift_begin_day_elem = new_shift_con.find('input[name="begin-day"]') ;
        new_shift_begin_day_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val($.datepicker.formatDate('yy-mm-dd', new Date())) ;

        var new_shift_begin_h_elem = new_shift_con.find('input[name="begin-h"]') ;
        new_shift_begin_h_elem.val('09') ;

        var new_shift_begin_m_elem = new_shift_con.find('input[name="begin-m"]') ;
        new_shift_begin_m_elem.val('00') ;

        var new_shift_end_day_elem = new_shift_con.find('input[name="end-day"]') ;
        new_shift_end_day_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val($.datepicker.formatDate('yy-mm-dd', new Date())) ;

        var new_shift_end_h_elem = new_shift_con.find('input[name="end-h"]') ;
        new_shift_end_h_elem.val('21') ;

        var new_shift_end_m_elem = new_shift_con.find('input[name="end-m"]') ;
        new_shift_end_m_elem.val('00') ;

        new_shift.click(function () {
            new_shift       .button('disable') ;
            new_shift_save  .button('enable') ;
            new_shift_cancel.button('enable') ;
            new_shift_con.removeClass('new-shift-hdn').addClass('new-shift-vis') ;
        }) ;

        new_shift_save.click(function () {
            var begin_day  = new_shift_begin_day_elem.val() ;
            var begin_time = begin_day + ' ' + Fwk.zeroPad(parseInt(new_shift_begin_h_elem.val()), 2) + ':' + Fwk.zeroPad(parseInt(new_shift_begin_m_elem.val()), 2) + ':00' ;
            var end_day    = new_shift_end_day_elem.val() ;
            var end_time   = end_day + ' ' + Fwk.zeroPad(parseInt(new_shift_end_h_elem.val()), 2) + ':' + Fwk.zeroPad(parseInt(new_shift_end_m_elem.val()), 2) + ':00' ;
            new_shift_save  .button('disable') ;
            new_shift_cancel.button('disable') ;
            that.shifts_service (
                '../shiftmgr/ws/shift_create.php' ,
                'GET' ,
                {   instr_name : that.instr_name ,
                    begin_time : begin_time ,
                    end_time   : end_time ,
                } ,
                function (shifts_data) {
                    that.shifts = [] ;
                    for (var i in shifts_data) {
                        var data = shifts_data[i] ;
                        that.shifts.push(new ShiftRow(that, data, that.can_edit)) ;
                    }
                    that.display() ;

                    that.ctrl_range_elem.val('range') ;
                    that.ctrl_begin_elem.removeAttr('disabled').val(begin_day) ;
                    that.ctrl_end_elem  .removeAttr('disabled').val(end_day) ;
                    new_shift.button('enable') ;
                    new_shift_con.removeClass('new-shift-vis').addClass('new-shift-hdn') ;
                } ,
                function () {
                    new_shift_save  .button('enable') ;
                    new_shift_cancel.button('enable') ;
                }
            ) ;
        }) ;
        new_shift_cancel.click(function () {
            new_shift       .button('enable') ;
            new_shift_save  .button('disable') ;
            new_shift_cancel.button('disable') ;
            new_shift_con.removeClass('new-shift-vis').addClass('new-shift-hdn') ;
        }) ;
    } ;

    this.search = function () {

        this.init() ;

        var that = this ;
        var range = this.ctrl_range_elem.val() ;
        var params = {
            instr_name : this.instr_name ,
            all        : that.ctrl_display_all_elem.attr('checked') ? 1 : 0 ,
            range      : range
        } ;
        if (range === 'range') {
            params.begin = that.ctrl_begin_elem.val() ;
            params.end   = that.ctrl_end_elem.val() ;
        }
        this.shifts_service (
            '../shiftmgr/ws/shifts_get.php', 'GET', params ,
            function (shifts_data) {
                that.shifts = [] ;
                for (var i in shifts_data) {
                    var data = shifts_data[i] ;
                    that.shifts.push(new ShiftRow(that, data, that.can_edit)) ;
                }
                that.display() ;
            }
        ) ;
    } ;

    this.search_shift_by_id = function(id) {

        this.init() ;

        var that = this ;
        that.shifts_service (
            '../shiftmgr/ws/shifts_get.php' ,
            'GET' ,
            { shift_id : id } ,
            function (shifts_data) {
                that.shifts = [] ;
                for (var i in shifts_data) {
                    var data = shifts_data[i] ;
                    that.shifts.push(new ShiftRow(that, data, that.can_edit)) ;

                    // Re-adjust visible search criteria to be compatible with the found
                    // shift.

                    var begin_day = data.begin.day ;
                    var end_day   = data.end.day ;

                    that.ctrl_range_elem.val('range') ;
                    that.ctrl_begin_elem.removeAttr('disabled').val(begin_day) ;
                    that.ctrl_end_elem  .removeAttr('disabled').val(end_day) ;
                    that.ctrl_display_all_elem.attr('checked', 'checked') ;
                }
                that.display() ;
            } ,
            function () {
                alert('Shift search failed for shift ID '+id) ;
            }
        ) ;
    } ;

    this.shifts_service = function (url, type, params, when_done, on_error) {

        var that = this ;

        $.ajax ({
            type: type ,
            url:  url ,
            data: params ,
            success: function (result) {
                if(result.status !== 'success') {
                    Fwk.report_error(result.message) ;
                    if (on_error) on_error() ;
                    return ;
                }
                if (when_done) {
                    that.search_info_elem.html('[ Last update: '+result.updated+' ]') ;
                    when_done(result.shifts) ;
                }
            } ,
            error: function () {
                Fwk.report_error('shift service is not available for instrument: '+that.inst_name) ;
                if (on_error) on_error() ;
            } ,
            dataType: 'json'
        }) ;
    } ;

    this.display = function () {
        this.shifts_stack.set_rows(this.shifts) ;
        this.shifts_stack.display(this.shifts_search_display) ;
    } ;

    this.shift_delete = function (shift_id) {
        var that = this ;
        var params = {
            shift_id : shift_id
        } ;
        this.shifts_service (
            '../shiftmgr/ws/shift_delete.php', 'GET', params ,
            function () {
                that.search() ;
            }
        ) ;
    } ;

    this.shift_update = function (shift_id, url, request_type, params) {
        var that = this ;
        this.shifts_service (
            url, request_type, params ,
            function (shifts_data) {
                that.shifts = [] ;
                for (var i in shifts_data) {
                    var data = shifts_data[i] ;
                    that.shifts.push(new ShiftRow(that, data, that.can_edit)) ;
                }
                that.display() ;
            }
        ) ;
    } ;
}
define_class (Reports, FwkApplication, {}, {});
