define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/Widget', 'webfwk/StackOfRows', 'webfwk/FwkApplication', 'webfwk/Fwk' ,
    'shiftmgr/Definitions'] ,

function (
    cssloader ,
    Class, Widget, StackOfRows, FwkApplication, Fwk ,
    Definitions) {

    cssloader.load('../shiftmgr/css/shiftmgr.css') ;

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
                case 'type'     : html = '<div class="shift-type">'     + this.data.type      + '</div>' ; break ;
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
                case 'OTHR'     : html = '<div class="shift-area" ><div class="status-'+(this.data.area[id].problems ?'red':'neutral')+'"></div></div>' ; break ;

                case 'editor'   : html = '<div class="shift-editor"   >&nbsp;' + this.data.editor   + '</div>' ; break ;
                case 'modified' : html = '<div class="shift-modified" >&nbsp;' + this.data.modified + '</div>' ; break ;

                case 'alerts'   :
                    var alerts = '';
                    if (!this.data.editor) alerts = (alerts ? ', ' : '') + 'no input' ;
                    for (var i in Definitions.AreaNames) {
                        var area_name = Definitions.AreaNames[i].key ;
                        var area = this.data.area[area_name] ;
                        if (area.downtime_min && !area.comments) alerts += (alerts ? ', ' : '') + 'areas' ;
                    }
                    if ((this.data.allocation['other'].duration_min > Definitions.MinOther2Comment) && !this.data.allocation['other'].comments) alerts += (alerts ? ', ' : '') + 'allocations' ;
                    if (alerts) html =
'<div style="float:left;"><span class="ui-icon ui-icon-alert"></span></div>' +
'<div style="float:left;"><span style="color:maroon;">'+alerts+'</span></div>' +
'<div style="clear:both;"></div>' ;
                    break ;
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

        Widget.Widget.call(this) ;

        // ------------------------
        // Parameters of the object
        // ------------------------

        this.parent = parent ;
        this.data = data ;
        this.can_edit = can_edit ;

        // ----------------------------
        // Static variables & functions
        // ----------------------------

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

            var experiments = '' ;
            for (var i in shift.experiments) {
                var e = shift.experiments[i] ;
                experiments += '<a target="_blank" href="../portal/index.php?exper_id='+e.id+'" title="open Web Portal of the Experiment in new window/tab">'+e.name+'</a>&nbsp;&nbsp;&nbsp;' ;
            }

            var html =
'<div>' +

'  <div style="float:left;">' +
'    <table style="font-size:90%;"><tbody>' +
'      <tr class="shift-active-row" >' +
'        <td class="annotated shift-grid-hdr " valign="center" data="This is the type of the shift" >Type:</td>' +
'        <td class="shift-grid-val " valign="center" >' +
'          <div class="viewer" >' +
'            <span name="type" style="font-weight: bold;" ></span>' +
'          </div>' +
'          <div class="editor" >' +
'            <select name="type" >' +
'              <option value="USER"     >USER</option>' +
'              <option value="MD"       >MD</option>' +
'              <option value="IN-HOUSE" >IN-HOUSE</option>' +
'            </select>' +
'          </div></td>' +
'        <td class="annotated shift-grid-hdr " valign="center" data="experiments which were active during the shift" >Experiment(s):</td>' +
'        <td class="shift-grid-val " colspan="2" valign="center" >'+experiments+'</td>' +
'      </tr>' +
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
'    <button class="annotated" name="edit"   data="Switch to the editing mode" style="color:red; font-weight:bold;" >EDIT</button>' +
'    <button class="annotated" name="save"   data="Submit modifications and switch back to the vieweing mode to edit">SAVE</button>' +
'    <button class="annotated" name="cancel" data="Discard modifications and revert back to the viewing mode">CANCEL</button>' +
'    <button class="annotated" name="delete" data="Delete the shift.\n Note that an additional confirmation will be requested\n before deliting the shift." style="margin-left:40px; color:red; font-weight:bold;" >DELETE SHIFT</button>' ;
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
            for (var i in Definitions.AreaNames) {
                var classes = 'annotated shift-table-val'+(first?'-first':'') ;
                first = false ;
                var area_name = Definitions.AreaNames[i].key ;
                html +=
'          <tr class="shift-active-row" >' +
'            <td class="'+classes+' shift-table-val-hdr " valign="top" data="'+Definitions.AreaNames[i].description+'" >'+Definitions.AreaNames[i].name+'</td>' +
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
'            <td class="shift-table-val-nodecor" >' +
'              <div class="alerts" name="'+area_name+'" ></div>' +
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
            for (var i in Definitions.AllocationNames) {
                var classes = 'annotated shift-table-val'+(first?'-first':'') ;
                first = false ;
                var allocation_name = Definitions.AllocationNames[i].key ;
                html +=
'          <tr class="shift-active-row" >' +
'            <td class="'+classes+' shift-table-val-hdr " valign="top" data="'+Definitions.AllocationNames[i].description+'" >'+Definitions.AllocationNames[i].name+'</td>' +
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
'            <td class="shift-table-val-nodecor" >' +
'              <div class="alerts" name="'+allocation_name+'" ></div>' +
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
        this.alert_explain = function () {
            var html =
'<div style="float:left;"><span class="ui-icon ui-icon-alert"></span></div><div style="float:left;"><span style="color:maroon;">Please, explain</span></div><div style="clear:both;"></div>' ;
            return html ;
        } ;

        // --------------------
        // Own data and methods
        // --------------------

        this.shift_register_handlers = function() {
            var that = this ;
            var shift = this.data ;
            var con = this.container ;

            var type_elem = con.find('select[name="type"]') ;

            var begin_day_elem = con.find('input[name="begin-day"]').datepicker() ;
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

            for (var i in Definitions.AreaNames) {
                var area_name = Definitions.AreaNames[i].key ;
                con.find('input.flag[name="'+area_name+'"]').change(function () {
                    var area_name = this.name ;
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
                        var comment = con.find('textarea.comment[name="'+area_name+'"]').val() ;
                        con.find('div.alerts[name="'+area_name+'"]').html(downtime_min && !comment ? that.alert_explain() : '') ;
                    } else {
                        h_elem.attr('disabled', 'disabled') ;
                        m_elem.attr('disabled', 'disabled') ;
                        con.find('div.alerts[name="'+area_name+'"]').html('') ;
                    }
                    h_elem.val(h) ;
                    m_elem.val(m) ;
                    con.find('textarea.comment[name="'+area_name+'"]').val(shift.area[area_name].comments) ;
                }) ;
                con.find('input.hour[name="'+area_name+'"]').change(function () {
                    var area_name = this.name ;
                    var h = parseInt($(this).val()) || 0;
                    if (h <  0) h = 0 ;
                    if (!that.shift_area_total_update(that.shift_duration_min())) h = 0;
                    $(this).val(Fwk.zeroPad(h, 2)) ;
                    var m = parseInt(con.find('input.minute[name="'+area_name+'"]').val()) || 0 ;
                    var downtime_min = 60 * h + m ;
                    var comment = con.find('textarea.comment[name="'+area_name+'"]').val() ;
                    con.find('div.alerts[name="'+area_name+'"]').html(downtime_min && !comment ? that.alert_explain() : '') ;
                }) ;
                con.find('input.minute[name="'+area_name+'"]').change(function () {
                    var area_name = this.name ;
                    var m = parseInt($(this).val()) || 0;
                    if (m > 59) m = 59 ;
                    if (m <  0) m = 0 ;
                    $(this).val(m) ;
                    if (!that.shift_area_total_update(that.shift_duration_min())) m = 0;
                    $(this).val(Fwk.zeroPad(m, 2)) ;
                    var h = parseInt(con.find('input.hour[name="'+area_name+'"]').val()) || 0 ;
                    var downtime_min = 60 * h + m ;
                    var comment = con.find('textarea.comment[name="'+area_name+'"]').val() ;
                    con.find('div.alerts[name="'+area_name+'"]').html(downtime_min && !comment ? that.alert_explain() : '') ;
                }) ;
                con.find('textarea.comment[name="'+area_name+'"]').change(function () {
                    var area_name = this.name ;
                    var comment = $(this).val() ;
                    var h = parseInt(con.find('input.hour[name="'+area_name+'"]').val()) || 0 ;
                    var m = parseInt(con.find('input.minute[name="'+area_name+'"]').val()) || 0 ;
                    var downtime_min = 60 * h + m ;
                    con.find('div.alerts[name="'+area_name+'"]').html(downtime_min && !comment ? that.alert_explain() : '') ;
                }) ;
            }
            for (var i in Definitions.AllocationNames) {
                var allocation_name = Definitions.AllocationNames[i].key ;
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
                } else {
                    con.find('textarea.comment[name="'+allocation_name+'"]').change(function () {
                        var allocation_name = this.name ;
                        var comment = $(this).val() ;
                        var hm = con.find('span.hour-minute[name="'+allocation_name+'"]').text().split(':') ;
                        var m = 60 * parseInt(hm[0]) + parseInt(hm[1]) ;
                        con.find('div.alerts[name="'+allocation_name+'"]').html((m > Definitions.MinOther2Comment) && !comment ? that.alert_explain() : '') ;
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
            for (var i in Definitions.AreaNames) {
                var area_name = Definitions.AreaNames[i].key ;
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
            for (var i in Definitions.AllocationNames) {
                var allocation_name = Definitions.AllocationNames[i].key ;
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
            con.find('div.alerts[name="other"]').html((other_min > Definitions.MinOther2Comment) && !con.find('textarea.comment[name="other"]').val() ? this.alert_explain() : '') ;
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

            con.find('select[name="type"]').val(shift.type) ;
            con.find('input[name="begin-day"]').datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val(shift.begin.day) ;
            con.find('input[name="begin-h"]').val(shift.begin.hour) ;
            con.find('input[name="begin-m"]').val(shift.begin.minute) ;
            con.find('input[name="end-day"]').datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd').val(shift.end.day) ;
            con.find('input[name="end-h"]').val(shift.end.hour) ;
            con.find('input[name="end-m"]').val(shift.end.minute) ;
            con.find('textarea[name="notes"]').val(shift.notes) ;

            for (var i in Definitions.AreaNames) {
                var area_name = Definitions.AreaNames[i].key ;
                var h_elem = con.find('input.hour[name="'+area_name+'"]') ;
                var m_elem = con.find('input.minute[name="'+area_name+'"]') ;
                var h = '' ;
                var m = '' ;
                var comment = shift.area[area_name].comments ;
                if (shift.area[area_name].problems) {
                    con.find('input.flag[name="'+area_name+'"]').attr('checked','checked') ;
                    var downtime_min = shift.area[area_name].downtime_min;
                    h = Fwk.zeroPad(Math.floor(downtime_min / 60), 2) ;
                    m = Fwk.zeroPad(downtime_min % 60, 2) ;
                    h_elem.removeAttr('disabled') ;
                    m_elem.removeAttr('disabled') ;
                    con.find('div.alerts[name="'+area_name+'"]').html(downtime_min && !comment ? this.alert_explain() : '') ;
                } else {
                    con.find('input.flag[name="'+area_name+'"]').removeAttr('checked') ;
                    h_elem.attr('disabled', 'disabled') ;
                    m_elem.attr('disabled', 'disabled') ;
                    con.find('div.alerts[name="'+area_name+'"]').html('') ;
                }
                con.find('input.hour[name="'+area_name+'"]').val(h) ;
                con.find('input.minute[name="'+area_name+'"]').val(m) ;
                con.find('textarea.comment[name="'+area_name+'"]').val(comment) ;
            }
            for (var i in Definitions.AllocationNames) {
                var allocation_name = Definitions.AllocationNames[i].key ;
                var duration_min = shift.allocation[allocation_name].duration_min ;
                var h = Math.floor(duration_min / 60) ;
                var m = duration_min % 60 ;
                var percent = shift.duration_min ? Math.floor(100 * duration_min / shift.duration_min) : 0;
                var comment = shift.allocation[allocation_name].comments ;
                if (allocation_name !== 'other') {
                    con.find('input.hour[name="'+allocation_name+'"]').val(Fwk.zeroPad(h, 2)) ;
                    con.find('input.minute[name="'+allocation_name+'"]').val(Fwk.zeroPad(m, 2)) ;
                }
                con.find('span.percent[name="'+allocation_name+'"]').text(percent) ;
                con.find('textarea.comment[name="'+allocation_name+'"]').val(comment) ;
                if (allocation_name === 'other') {
                    con.find('div.alerts[name="'+allocation_name+'"]').html((duration_min > Definitions.MinOther2Comment) && !comment ? this.alert_explain() : '') ;
                }
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
                type     : con.find('select[name="type"]').val() ,
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
            for (var i in Definitions.AreaNames) {
                var area_name = Definitions.AreaNames[i].key ;
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
            for (var i in Definitions.AllocationNames) {
                var allocation_name = Definitions.AllocationNames[i].key ;
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

            con.find('span[name="type"]').html(shift.type) ;
            con.find('span[name="begin-day"]').text(shift.begin.day) ;
            con.find('span[name="begin-hm"]').text(Fwk.zeroPad(shift.begin.hour, 2)+':'+Fwk.zeroPad(shift.begin.minute, 2)) ;
            con.find('span[name="end-day"]').text(shift.end.day) ;
            con.find('span[name="end-hm"]').text(Fwk.zeroPad(shift.end.hour, 2)+':'+Fwk.zeroPad(shift.end.minute, 2)) ;
            con.find('div[name="notes"]').html('<pre>'+shift.notes+'</pre>') ;

            var area_total_min = 0 ;
            for (var i in Definitions.AreaNames) {
                var area_name = Definitions.AreaNames[i].key ;
                con.find('div.flag[name="'+area_name+'"]').addClass(shift.area[area_name].problems?'status-red':'status-neutral') ;
                var hm = '' ;
                var comment = shift.area[area_name].comments ;
                if (shift.area[area_name].problems) {
                    var downtime_min = shift.area[area_name].downtime_min;
                    hm = Fwk.zeroPad(Math.floor(downtime_min / 60), 2) + ':' + Fwk.zeroPad(downtime_min % 60, 2) ;
                    area_total_min += downtime_min ;
                    con.find('div.alerts[name="'+area_name+'"]').html(downtime_min && !comment ? this.alert_explain() : '') ;
                } else {
                    con.find('div.alerts[name="'+area_name+'"]').html('') ;
                }
                con.find('span.hour-minute[name="' +area_name+'"]').text(hm) ;
                con.find('div.comment[name="'+area_name+'"]').html('<pre>'+comment+'</pre>') ;
            }
            con.find('span.hour-minute[name="area"]').text(Fwk.zeroPad(Math.floor(area_total_min / 60), 2)+':'+Fwk.zeroPad(area_total_min % 60, 2)) ;

            var allocation_total_min = 0 ;
            for (var i in Definitions.AllocationNames) {
                var allocation_name = Definitions.AllocationNames[i].key ;
                if (typeof(shift.allocation[allocation_name]) === 'undefined' ) alert('allocation_name: '+allocation_name) ;
                var duration_min = shift.allocation[allocation_name].duration_min ;
                var hm = Fwk.zeroPad(Math.floor(duration_min / 60), 2) + ':' + Fwk.zeroPad(duration_min % 60, 2) ;
                var percent = shift.duration_min ? Math.floor(100 * duration_min / shift.duration_min) : 0;
                var comment = shift.allocation[allocation_name].comments ;
                allocation_total_min += duration_min ;
                con.find('span.hour-minute[name="' +allocation_name+'"]').text(hm) ;
                con.find('span.percent[name="'+allocation_name+'"]').text(percent) ;
                con.find('div.comment[name="'+allocation_name+'"]').html('<pre>'+comment+'</pre>') ;
                if (allocation_name === 'other') {
                    con.find('div.alerts[name="'+allocation_name+'"]').html((duration_min > Definitions.MinOther2Comment) && !comment ? this.alert_explain() : '') ;
                }
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
    Class.define_class (ShiftBody, Widget.Widget, {}, {}) ;

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

        StackOfRows.StackRowData.call(this) ;

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
    Class.define_class(ShiftRow, StackOfRows.StackRowData, {}, {}) ;


    function Reports_Export2Excel (parent) {
        this.parent   = parent ;
        this.icon     = function () { return '../webfwk/img/MS_Excel_1.png' ; } ;
        this.title    = function () { return 'Export into Microsoft Excel 2007 File' ; } ;
        this.on_click = function () { this.parent.export('excel') ; } ;
    }

    function Reports_Print (parent) {
        this.parent   = parent ;
        this.icon     = function () { return '../webfwk/img/Printer.png' ; } ;
        this.title    = function () { return 'Print the document' ; } ;
        this.on_click = function () { alert('Printing...') ; } ;
    }

    /**
     * The main class representing instrument-level reports.
     *
     * @param {string} instr_name
     * @param {boolean} can_edit
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

        this.tools = function () {
            if (!this.my_tools)
                this.my_tools = [
                    new Reports_Export2Excel(this) ,
                    new Reports_Print(this)] ;
            return this.my_tools ;
        } ;

        // --------------------
        // Own data and methods
        // --------------------

        this.shifts = [] ;
        this.shifts_stack = null ;

        this.seconds_since_last_check = 0 ;

        this.update_allowed = function () {
            var not_allowed =
                (this.seconds_since_last_check) ||
                this.web_service_is_loading ||
                (this.shifts_stack && this.shifts_stack.is_locked());
            this.seconds_since_last_check++ ;
            if (this.seconds_since_last_check > Definitions.ShiftsUpdateInterval_Sec) this.seconds_since_last_check = 0 ;

            return !not_allowed ;
        } ;

        this.is_initialized = false ;

        this.init = function () {

            var that = this ;

            if (this.is_initialized) return ;
            this.is_initialized = true ;

            this.container.html (
'<div class="shift-reports">' +
  '<div id="shifts-search-controls"  style="float:left;" >' +
    '<div class="shifts-search-filters" >' +
      '<div class="shifts-search-filter-group" >' +
        '<div class="header" >Time range</div>' +
        '<div class="cell-1" >' +
          '<select class="filter" name="range" style="padding:1px;">' +
            '<option value="week"  >Last 7 days</option>' +
            '<option value="month" >Last month</option>' +
            '<option value="range" >Specific range</option>' +
          '</select>' +
        '</div>' +
        '<div class="cell-2" >' +
          '<input class="filter" type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />' +
          '<input class="filter" type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" />' +
        '</div>' +
        '<div class="terminator" ></div>' +
      '</div>' +
      '<div class="shifts-search-filter-group" >' +
        '<div class="header" >Stopper out</div>' +
        '<div class="cell-2">' +
          '<select class="filter" name="stopper" style="padding:1px;">' +
            '<option value=""  ></option>' +
            '<option value="0" >&gt; 0 %</option>' +
            '<option value="1" >&gt; 1 %</option>' +
            '<option value="2" >&gt; 2 %</option>' +
            '<option value="3" >&gt; 3 %</option>' +
            '<option value="4" >&gt; 4 %</option>' +
            '<option value="5" >&gt; 5 %</option>' +
          '</select>' +
        '</div>' +
        '<div class="header-1">Door open</div>' +
        '<div class="cell-2" >' +
          '<select class="filter" name="door" style="padding:1px;">' +
            '<option value="" ></option>' +
            '<option value="100" >&lt; 100 %</option>' +
            '<option value="99"  >&lt; 99 %</option>' +
            '<option value="98"  >&lt; 98 %</option>' +
            '<option value="97"  >&lt; 97 %</option>' +
            '<option value="96"  >&lt; 96 %</option>' +
            '<option value="95"  >&lt; 95 %</option>' +
          '</select>' +
        '</div>' +
        '<div class="header-1" >LCLS beam</div>' +
        '<div class="cell-2">' +
          '<select class="filter"t name="lcls" style="padding:1px;">' +
            '<option value=""  ></option>' +
            '<option value="0" >&gt; 0 %</option>' +
            '<option value="1" >&gt; 1 %</option>' +
            '<option value="2" >&gt; 2 %</option>' +
            '<option value="3" >&gt; 3 %</option>' +
            '<option value="4" >&gt; 4 %</option>' +
            '<option value="5" >&gt; 5 %</option>' +
          '</select>' +
        '</div>' +
        '<div class="header-1">Data taking</div>' +
        '<div class="cell-2" >' +
          '<select class="filter" name="daq" style="padding:1px;">' +
            '<option value=""  ></option>' +
            '<option value="0" >&gt; 0 %</option>' +
            '<option value="1" >&gt; 1 %</option>' +
            '<option value="2" >&gt; 2 %</option>' +
            '<option value="3" >&gt; 3 %</option>' +
            '<option value="4" >&gt; 4 %</option>' +
            '<option value="5" >&gt; 5 %</option>' +
          '</select>' +
        '</div>' +
        '<div class="terminator" ></div>' +
      '</div>' +
      '<div class="shifts-search-filter-group" >' +
        '<div class="header" >Shift types</div>' +
        '<div class="cell-2">' +
          '<div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="USER"     title="if enabled it will include shifts of this type" /></div><div class="cell-4">USER</div>' +
          '<div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="MD"       title="if enabled it will include shifts of this type" /></div><div class="cell-4">MD</div>' +
          '<div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="IN-HOUSE" title="if enabled it will include shifts of this type" /></div><div class="cell-4">IN-HOUSE</div>' +
        '</div>' +
        '<div class="terminator" ></div>' +
      '</div>' +
    '</div>' +
    '<div class="shifts-search-buttons" >' +
      '<button name="reset"  title="reset the search form to the default state">RESET</button>' +
    '</div>' +
    '<div class="shifts-search-filter-terminator" ></div>' +
  '</div>' + (!this.can_edit ? '' :
  '<div id="new-shift-controls" style="float:left; margin-left:10px;">' +
    '<button name="new-shift" title="open a dialog for creating a new shift" >CREATE NEW SHIFT</button>' +
    '<div id="new-shift-con" class="new-shift-hdn" style="background-color:#f0f0f0; margin-top:5px; padding:1px 10px 5px 10px; border-radius:5px;" >' +
      '<div style="max-width:460px;">' +
        '<p>Note that shifts are usually created automatically based on rules defined' +
        '   in the Administrative section of this application. You may still want to create' +
        '   your own shift if that shift happens to be an exception from the rules.' +
        '   Possible cases would be: non-planned shift, very short shift, etc. In all' +
        '   other cases please see if there is a possibility to reuse an empty shift slot' +
        '   by checking "Display all shifts" checkbox on the left.</p>' +
      '</div>' +
      '<div style="float:left;">' +
        '<table style="font-size:90%;"><tbody>' +
          '<tr>' +
            '<td class="shift-grid-hdr " valign="center" >Type:</td>' +
            '<td class="shift-grid-val " valign="center" >' +
              '<select name="type" >' +
                '<option value="USER"     >USER</option>' +
                '<option value="MD"       >MD</option>' +
                '<option value="IN-HOUSE" >IN-HOUSE</option>' +
              '</select></td>' +
          '</tr>' +
          '<tr>' +
            '<td class="shift-grid-hdr " valign="center" >Begin:</td>' +
            '<td class="shift-grid-val " valign="center" >' +
              '<input name="begin-day" type="text" size=8 title="specify the begin date of the shift" />' +
              '<input name="begin-h"   type="text" size=1 title="hour: 0..23" />' +
              '<input name="begin-m"   type="text" size=1 title="minute: 0..59" /></td>' +
          '</tr>' +
          '<tr>' +
            '<td class="shift-grid-hdr " valign="center" >End:</td>' +
            '<td class="shift-grid-val " valign="center" >' +
              '<input name="end-day" type="text" size=8 title="specify the end date of the shift" />' +
              '<input name="end-h"   type="text" size=1 title="hour: 0..23" />' +
              '<input name="end-m"   type="text" size=1 title="minute: 0..59" /></td>' +
          '</tr>' +
        '</tbody></table>' +
      '</div>' +
      '<div style="float:left; margin-left:20px; margin-top:40px; padding-top:5px;">' +
        '<button name="save"   title="submit modifications and open the editing dialog for the new shift">SAVE</button>' +
        '<button name="cancel" title="discard modifications and close this dialog">CANCEL</button>' +
      '</div>' +
      '<div style="clear:both;"></div>' +
    '</div>' +
  '</div>') +
  '<div style="clear:both;"></div>' +
  '<div style="float:right;" id="shifts-search-info">Searching...</div>' +
  '<div style="clear:both;"></div>' +
  '<div id="shifts-search-display"> </div>' +
'</div>'
            ) ;
            this.ctrl_elem = this.container.find('#shifts-search-controls') ;

            this.ctrl_range_elem       = this.ctrl_elem.find('select[name="range"]') ;
            this.ctrl_begin_elem       = this.ctrl_elem.find('input[name="begin"]') ;
            this.ctrl_end_elem         = this.ctrl_elem.find('input[name="end"]') ;
            this.ctrl_stopper_elem     = this.ctrl_elem.find('select[name="stopper"]') ;
            this.ctrl_door_elem        = this.ctrl_elem.find('select[name="door"]') ;
            this.ctrl_lcls_elem        = this.ctrl_elem.find('select[name="lcls"]') ;
            this.ctrl_daq_elem         = this.ctrl_elem.find('select[name="daq"]') ;

            this.ctrl_begin_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;
            this.ctrl_end_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;

            this.ctrl_elem.find('button[name="reset"]').button().click(function () {
                that.reset_and_search() ;
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
            this.ctrl_elem.find('.filter').change(function () {
                that.search() ;
            }) ;

            this.search_info_elem = this.container.find('#shifts-search-info') ;
            this.shifts_search_display = this.container.find('#shifts-search-display') ;

            this.shifts_stack = new StackOfRows.StackOfRows ([
                {id: 'type',     title: 'Type',     width:  65} ,
                {id: 'shift',    title: 'Shift',    width:  90} ,
                {id: 'begin',    title: 'begin',    width:  45} ,
                {id: 'end',      title: 'end',      width:  45} ,
                {id: 'duration', title: '&Delta;t', width:  50} ,
                {id: '|' } ,
                {id: 'stopper',  title: 'Stopper',  width:  65} ,
                {id: 'door',     title: 'Door',     width:  45} ,
                {id: '|' } ,
                {id: 'FEL',      title: 'Fel',      width:  40} ,
                {id: 'BMLN',     title: 'Beam',     width:  45} ,
                {id: 'CTRL',     title: 'Ctrl',     width:  40} ,
                {id: 'DAQ',      title: 'Daq',      width:  40} ,
                {id: 'LASR',     title: 'Lasr',     width:  40} ,
                {id: 'TIME',     title: 'Time',     width:  45} ,
                {id: 'HALL',     title: 'Hall',     width:  40} ,
                {id: 'OTHR',     title: 'Othr',     width:  40} ,
                {id: '|' } ,
                {id: 'editor',   title: 'Editor',   width:  80} ,
                {id: 'modified', title: 'Modified', width: 180} ,
                {id: 'alerts',   title: 'Alerts',   width: 240}
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
                var type       = new_shift_con.find('select[name="type"]').val() ;
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
                        type       : type ,
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

        this.reset_and_search = function (range, begin_time, end_time) {
            var range = range ? range : 'week' ;
            this.ctrl_range_elem.val(range) ;
            if (range === 'range') {
                this.ctrl_begin_elem.removeAttr('disabled') ;
                this.ctrl_end_elem  .removeAttr('disabled') ;
                this.ctrl_begin_elem.val(begin_time ? begin_time : '') ;
                this.ctrl_end_elem.val  (end_time   ? end_time   : '') ;
            } else {
                this.ctrl_begin_elem.attr('disabled', 'disabled') ;
                this.ctrl_end_elem  .attr('disabled', 'disabled') ;
                this.ctrl_begin_elem.val('') ;
                this.ctrl_end_elem.val  ('') ;
            }
            this.ctrl_stopper_elem.val('') ;
            this.ctrl_door_elem.val('') ;
            this.ctrl_lcls_elem.val('') ;
            this.ctrl_daq_elem.val('') ;
            this.ctrl_elem.find('input.type').attr('checked', 'checked') ;
            this.search() ;
        } ;

        this.search = function (export_format) {

            this.init() ;

            var that = this ;

            var range = this.ctrl_range_elem.val() ;
            var types = '' ;
            that.ctrl_elem.find('input.type:checked').each(function () {
                if (types) types += ':' ;
                types += this.name ;
            }) ;
            var params = {
                range       : range ,
                stopper     : this.ctrl_stopper_elem.val() ,
                door        : this.ctrl_door_elem.val() ,
                lcls        : this.ctrl_lcls_elem.val() ,
                daq         : this.ctrl_daq_elem.val() ,
                instruments : this.instr_name ,
                types       : types
            } ;
            if (range === 'range') {
                params.begin = that.ctrl_begin_elem.val() ;
                params.end   = that.ctrl_end_elem.val() ;
            }
            if (export_format) {
                params.export = export_format
                var url = '../shiftmgr/ws/shifts_export.php?'+$.param(params, true) ;
                window.open(url) ;
            } else {
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
            }
        } ;

        this.search_shift_by_id = function(id) {

            this.init() ;

            var that = this ;
            that.shifts_service (
                '../shiftmgr/ws/shifts_get.php' ,
                'GET' ,
                { shift_id : id } ,
                function (shifts_data) {

                    // Re-adjust visible search criteria to be compatible with the found
                    // shift.

                    var data = shifts_data[0] ;
                    that.reset_and_search('range', data.begin.day, data.end.day) ;
                } ,
                function () {
                    alert('Shift search failed for shift ID '+id) ;
                }
            ) ;
        } ;

        this.web_service_is_loading = false ;

        this.shifts_service = function (url, type, params, when_done, on_error) {

            var that = this ;

            this.web_service_is_loading = true ;

            this.search_info_elem.html('Loading...') ;

            $.ajax ({
                type: type ,
                url:  url ,
                data: params ,
                success: function (result) {
                    that.web_service_is_loading = false ;
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
                    that.web_service_is_loading = false ;
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

        this.export = function(format) {
            this.search(format) ;
        } ;
    }
    Class.define_class (Reports, FwkApplication, {}, {}) ;

    return Reports ;
}) ;