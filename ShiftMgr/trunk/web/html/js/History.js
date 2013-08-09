function History (instr_name) {
    FwkDispatcherBase.call(this) ;
    this.instr_name = instr_name || '';
}
define_class (History, FwkDispatcherBase, {}, {

    on_activate : function() {
        this.on_update() ;
    } ,

    on_deactivate : function() {
        this.init() ;
    } ,

    on_update : function () {
        this.init() ;
        if (this.active) this.load_history() ;
    } ,

    is_initialized : false ,

    init : function () {

        var that = this ;

        if (this.is_initialized) return ;
        this.is_initialized = true ;

        var ctrl_elem = this.container.find('#shifts-history-controls') ;

        this.ctrl_range_elem       = ctrl_elem.find('select[name="range"]') ;
        this.ctrl_begin_elem       = ctrl_elem.find('input[name="begin"]') ;
        this.ctrl_end_elem         = ctrl_elem.find('input[name="end"]') ;

        this.ctrl_display_create_shift_elem = ctrl_elem.find('input[name="display-create-shift"]') ;
        this.ctrl_display_modify_shift_elem = ctrl_elem.find('input[name="display-modify-shift"]') ;
        this.ctrl_display_modify_area_elem  = ctrl_elem.find('input[name="display-modify-area"]') ;
        this.ctrl_display_modify_time_elem  = ctrl_elem.find('input[name="display-modify-time"]') ;

        this.ctrl_begin_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;
        this.ctrl_end_elem.datepicker().datepicker('option', 'dateFormat', 'yy-mm-dd') ;

        ctrl_elem.find('button[name="reset"]').button().click(function () {
            var range = 'week' ;
            that.ctrl_range_elem.val(range) ;
            that.ctrl_begin_elem.val('') ;
            that.ctrl_end_elem.val('') ;
            that.ctrl_begin_elem.attr('disabled', 'disabled') ;
            that.ctrl_end_elem  .attr('disabled', 'disabled') ;
            that.load_history() ;
        }) ;
        this.ctrl_range_elem.change(function () {
            var range = that.ctrl_range_elem.val() ;
            switch (range) {
                case 'week'  :
                case 'month' :
                    that.ctrl_begin_elem.attr('disabled', 'disabled') ;
                    that.ctrl_end_elem  .attr('disabled', 'disabled') ;
                    that.load_history() ;
                    break ;
                case 'range' :
                    that.ctrl_begin_elem.removeAttr('disabled') ;
                    that.ctrl_end_elem  .removeAttr('disabled') ;
                    if (that.ctrl_begin_elem.val() || that.ctrl_end_elem.val()) that.load_history() ;
                    break ;
            }
        }) ;
        this.ctrl_begin_elem.change(function () {
            if (that.ctrl_begin_elem.val() || that.ctrl_end_elem.val()) that.load_history() ;
        }) ;
        this.ctrl_end_elem.change(function () {
            if (that.ctrl_begin_elem.val() || that.ctrl_end_elem.val()) that.load_history() ;
        }) ;
        this.ctrl_display_create_shift_elem.change(function () { that.show_history() ; }) ;
        this.ctrl_display_modify_shift_elem.change(function () { that.show_history() ; }) ;
        this.ctrl_display_modify_area_elem.change (function () { that.show_history() ; }) ;
        this.ctrl_display_modify_time_elem.change (function () { that.show_history() ; }) ;
        
        this.history_info_elem    = this.container.find('#shifts-history-info') ;
        this.history_display_elem = this.container.find('#shifts-history-display') ;
        
        var hdr = this.instr_name ? [
            { name: 'modified'  },
            { name: 'editor'  },
            { name: 'shift' },
            { name: 'event' },
            { name: 'area/allocation' },
            { name: 'parameter' },
            { name: 'old value', sorted: false },
            { name: 'new value', sorted: false }
        ] : [
            { name: 'modified'  },
            { name: 'editor'  },
            { name: 'instr' },
            { name: 'shift' },
            { name: 'event' },
            { name: 'area/allocation' },
            { name: 'parameter' },
            { name: 'old value', sorted: false },
            { name: 'new value', sorted: false }
        ] ;
        this.history_table = new Table (
            this.container.find('#shifts-history-display') ,
            hdr ,
            null ,                              // data will be loaded dynamically
            {   default_sort_column:  0 ,
                default_sort_forward: false ,
                text_when_empty:      'Loading...'
            } ,
            Fwk.config_handler('History', 'instrument='+(this.instr_name ? this.instr_name : 'all'))
        ) ;
        this.history_table.display() ;
    } ,

    history: null ,
    updated: null ,

    load_history : function() {
        var that = this ;
        var range = this.ctrl_range_elem.val() ;
        var begin = this.ctrl_begin_elem.val() ;
        var end   = this.ctrl_end_elem.val() ;
        var params = {
            range : range ,
            begin : begin ,
            end   : end
        } ;
        if (this.instr_name)
            params.instr_name = this.instr_name ;
        if (this.updated && !end)
            params.since = this.updated ;

        var jqXHR = $.get('../shiftmgr/ws/history_get.php', params, function (data) {
            if (data.status !== 'success') {
                Fwk.report_error(data.message, null) ;
                that.history_info_elem.html('Update has failed') ;
                return ;
            }
            that.updated = data.updated ;
            that.history_info_elem.html('[ Last update: '+that.updated+' ]') ;
            if (that.history) {
                if (params.since) {
                    var num = 0 ;
                    for (var i in data.history) {
                        that.history.push(data.history[i]) ;
                        num++ ;
                    }
                    if (num) {
                        that.show_history() ;
                    }
                } else {
                    that.history = data.history ;
                    that.show_history() ;
                }
            } else {
                that.history = data.history ;
                that.show_history() ;
            }
        } ,
        'JSON').error(function () {
            Fwk.report_error('failed to obtain the cable history because of: '+jqXHR.statusText, null) ;
            that.history_info_elem.html('Update has failed') ;
            return ;
        }) ;
    } ,

    show_history : function() {
        var display_create_shift = this.ctrl_display_create_shift_elem.attr('checked') ;
        var display_modify_shift = this.ctrl_display_modify_shift_elem.attr('checked') ;
        var display_modify_area  = this.ctrl_display_modify_area_elem.attr ('checked') ;
        var display_modify_time  = this.ctrl_display_modify_time_elem.attr ('checked') ;
        var rows = [] ;
        for (var i in this.history) {
            var event = this.history[i] ;
            if        (event.scope === 'SHIFT') {
                if      (event.operation === 'CREATE') { if (!display_create_shift) continue ; }
                else if (event.operation === 'MODIFY') { if (!display_modify_shift) continue ; }
            } else if (event.scope === 'AREA') {
                if      (event.operation === 'MODIFY') { if (!display_modify_area)  continue ; }
            } else if (event.scope === 'TIME') {
                if      (event.operation === 'MODIFY') { if (!display_modify_time)  continue ; }
            }
            var row = this.instr_name ? [
                event.modified.full ,
                event.editor ,
                "<a href=\"javascript:Fwk.activate('"+event.instr_name+"','Reports').search_shift_by_id("+event.shift_id+");\">"+event.shift_begin.day+'&nbsp;&nbsp;'+event.shift_begin.hm+'</a>' ,
                event.operation+' '+event.scope ,
                event.scope2 ,
                event.parameter ,
                event.old_value ,
                event.new_value
            ] : [
                event.modified.full ,
                event.editor ,
                "<a href=\"javascript:Fwk.activate('"+event.instr_name+"','Reports');\">"+event.instr_name+'</a>' ,
                "<a href=\"javascript:Fwk.activate('"+event.instr_name+"','Reports').search_shift_by_id("+event.shift_id+");\">"+event.shift_begin.day+'&nbsp;&nbsp;'+event.shift_begin.hm+'</a>' ,
                event.operation+' '+event.scope ,
                event.scope2 ,
                event.parameter ,
                '<div class="comment"><pre>'+event.old_value+'</pre></div>' ,
                '<div class="comment"><pre>'+event.new_value+'</pre></div>'
            ] ;
            rows.push (row) ;
        }
        this.history_table.load(rows) ;
    }
});
