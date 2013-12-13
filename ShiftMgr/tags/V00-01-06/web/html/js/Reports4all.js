/**
 * This class representes a title of a row
 * 
 * @see StackOfRows
 * @see DayRow
 *
 * @param object data
 * @returns {DayTitle}
 */
function DayTitle(data) {
    this.data = data ;
    this.html = function (id) {
        var html = '' ;
        switch(id) {
            case 'day'  : html = '<div class="all-shifts-day">' +  this.data.begin.day  + '</div>' ; break ;
            case 'num'  : html = '<div class="all-shifts-num">' + (this.data.num || '') + '</div>' ; break ;

            case 'FEL'  :
            case 'BMLN' :
            case 'CTRL' :
            case 'DAQ'  :
            case 'LASR' :
            case 'TIME' :
            case 'HALL' :
            case 'OTHR' : html = '<div class="all-shifts-area" ><div class="status-'+(this.data.area_problems[id] ?'red':'neutral')+'"></div></div>' ; break ;

        }
        return html ;
    } ;
}

/**
 * This class representes a body of a row
 * 
 * @see StackOfRows
 * @see DayRow
 *
 * @param object parent
 * @param object data
 * @returns {DayBody}
 */
function DayBody (parent, data) {

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    Widget.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    this.parent = parent ;
    this.data = data ;

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

        var hdr = [
            {   name: 'Type' } ,
            {   name: 'Instr' ,
                type: {
                    to_string:      function(a)   { return a.instr_name ; } ,
                    compare_values: function(a,b) { return a.instr_name.localeCompare(b.instr_name) ; }}} ,
            {   name: 'begin' } ,
            {   name: 'end', sorted: false } ,
            {   name: '&Delta;t', sorted: false } ,
            {   name: 'Stopper', sorted: false } ,
            {   name: 'Door open', sorted: false }
        ] ;
        for (var i in Definitions.AreaNames)
            hdr.push(
                { name: Definitions.AreaNames[i].key, sorted: false }) ;

        var rows = [] ;
        for (var i in this.data.shifts) {
            var shift = this.data.shifts[i] ;
            var stopper_percent = shift.duration_min ? Math.floor(100 * shift.stopper_min / shift.duration_min) : 0 ;
            var door_open_percent = shift.duration_min ? Math.floor(100 * (shift.duration_min - shift.door_min) / shift.duration_min) : 0 ;
            if (door_open_percent < 0) door_open_percent = 0 ;
            var row = [
                shift.type ,
                {instr_name: shift.instr_name, id: shift.id} ,
                shift.begin.hm ,
                shift.end.hm ,
                shift.duration ,
                (stopper_percent ? stopper_percent+'%' : '&nbsp;') ,
                (door_open_percent ? door_open_percent+'%' : '&nbsp;')
            ] ;
            for (var j in Definitions.AreaNames)
                row.push('<div class="status-'+(shift.area[Definitions.AreaNames[j].key].problems ?'red':'neutral')+'"></div>') ;

            rows.push(row) ;
        }

        this.container.html (
'<div id="shifts_table" style="padding:10px;"></div>'
        ) ;
        this.shifts_table = new Table (
            this.container.find('#shifts_table') ,
            hdr ,
            rows ,
            {   default_sort_column:  1 ,
                default_sort_forward: false ,
                row_select_action: function (row) {
                    var cell = row[1] ;
                    Fwk.activate(cell.instr_name,'Reports').search_shift_by_id(cell.id) ;
                }
            } ,
            Fwk.config_handler('Reports4all', 'shifts')
        ) ;
        this.shifts_table.display() ;
    } ;

    // --------------------
    // Own data and methods
    // --------------------
}
define_class (DayBody, Widget, {}, {}) ;



/**
 * This class binds the data with the row interface as requited by the StackOfRows class
 *
 * @see StackOfRows
 *
 * @param object parent
 * @param object data
 * @returns {DayRow}
 */
function DayRow (parent, data) {

    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    StackRowData.call(this) ;

    // ------------------------
    // Parameters of the object
    // ------------------------

    if (!data) throw new Error('DayRow:constructor() data is not defined') ;
    this.data = data ;

    this.title = new DayTitle(this.data) ;
    this.body  = new DayBody (parent, this.data) ;
}
define_class(DayRow, StackRowData, {}, {}) ;

function Reports_Export2Excel (parent) {
    this.parent   = parent ;
    this.icon     = function () { return '../webfwk/img/MS_Excel_1.png' ; } ;
    this.title    = function () { return 'Export into Microsoft Excel 2007 File' ; } ;
    this.on_click = function () { this.parent.export('excel') ; } ;
}

function Reports_Print (parent) {
    this.parent   = parent ;
    this.icon     = function () { return '../webfwk/img/Printer.png' ; } ;
    this.title    = function () { return 'Print the documente' ; } ;
    this.on_click = function () { alert('Printing...') ; } ;
}

/**
 * The application for making shift reports accross all instrument
 *
 * @returns {Reports4all}
 */
function Reports4all () {
 
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    FwkApplication.call(this) ;

    // ------------------------------------------------
    // Override event handler defined in the base class
    // ------------------------------------------------

    this.on_activate = function() {
        this.init() ;
        this.search() ;
    } ;

    this.on_deactivate = function() {
        ;
    } ;

    this.on_update = function () {
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

    this.days = [] ;
    this.days_stack = null ;

    this.is_initialized = false ;

    this.init = function () {

        var that = this ;

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        this.ctrl_elem = this.container.find('#shifts-search-controls') ;

        this.ctrl_range_elem   = this.ctrl_elem.find('select[name="range"]') ;
        this.ctrl_begin_elem   = this.ctrl_elem.find('input[name="begin"]') ;
        this.ctrl_end_elem     = this.ctrl_elem.find('input[name="end"]') ;
        this.ctrl_stopper_elem = this.ctrl_elem.find('select[name="stopper"]') ;
        this.ctrl_door_elem    = this.ctrl_elem.find('select[name="door"]') ;
        this.ctrl_lcls_elem    = this.ctrl_elem.find('select[name="lcls"]') ;
        this.ctrl_daq_elem     = this.ctrl_elem.find('select[name="daq"]') ;

        this.ctrl_begin_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;
        this.ctrl_end_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;

        this.ctrl_elem.find('button[name="reset"]').button().click(function () {
            var range = 'week' ;
            that.ctrl_range_elem.val(range) ;
            that.ctrl_begin_elem.val('') ;
            that.ctrl_end_elem.val('') ;
            that.ctrl_begin_elem.attr('disabled', 'disabled') ;
            that.ctrl_end_elem  .attr('disabled', 'disabled') ;
            that.ctrl_stopper_elem.val('') ;
            that.ctrl_door_elem.val('') ;
            that.ctrl_lcls_elem.val('') ;
            that.ctrl_daq_elem.val('') ;
            that.ctrl_elem.find('input.instrument').attr('checked', 'checked') ;
            that.ctrl_elem.find('input.type').attr('checked', 'checked') ;
            that.search() ;
        }) ;
        this.ctrl_elem.find('button[name="search"]').button().click(function () {
            that.search() ;
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
        this.ctrl_elem.find('.filter').change(function () {
            that.search() ;
        }) ;

        this.search_info_elem = this.container.find('#shifts-search-info') ;
        this.shifts_search_display = this.container.find('#shifts-search-display') ;

        this.days_stack = new StackOfRows ([
            {id: 'day',  title: 'Day',      width: 100} ,
            {id: 'num',  title: '# shifts', width:  60} ,
            {id: '|' } ,
            {id: 'FEL',  title: 'FEL',      width:  50} ,
            {id: 'BMLN', title: 'BMLN',     width:  50} ,
            {id: 'CTRL', title: 'CTRL',     width:  50} ,
            {id: 'DAQ',  title: 'DAQ',      width:  50} ,
            {id: 'LASR', title: 'LASR',     width:  50} ,
            {id: 'TIME', title: 'TIME',     width:  50} ,
            {id: 'HALL', title: 'HALL',     width:  50} ,
            {id: 'OTHR', title: 'OTHR',     width:  50}
        ] , null, {
            theme: 'stack-theme-large14 stack-theme-aliceblue'
        }) ;
    } ;
    
    this.search = function (export_format) {

        this.init() ;

        var that = this ;

        var range = this.ctrl_range_elem.val() ;

        var instruments = '' ;
        this.ctrl_elem.find('input.instrument:checked').each(function () {
            if (instruments) instruments += ':' ;
            instruments += this.name ;
        }) ;

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
            instruments : instruments ,
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
            params.group_by_day = 1 ;
            this.shifts_service (
                '../shiftmgr/ws/shifts_get.php', 'GET', params ,
                function (days_data) {
                    that.days = [] ;
                    for (var i in days_data) {
                        var data = days_data[i] ;
                        that.days.push(new DayRow(that, data)) ;
                    }
                    that.display() ;
                }
            ) ;
        }
    } ;

    this.shifts_service = function (url, type, params, when_done, on_error) {

        var that = this ;

        this.search_info_elem.html('Loading...') ;

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
                    when_done(result.days) ;
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
        this.days_stack.set_rows(this.days) ;
        this.days_stack.display(this.shifts_search_display) ;
    } ;

    this.export = function(format) {
        this.search(format) ;
    } ;
}
define_class (Reports4all, FwkApplication, {}, {});
