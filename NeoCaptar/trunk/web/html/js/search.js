function p_appl_search() {

    var that = this;

    this.when_done = null;

    /* -------------------------------------------------------------------------
     * Data structures and methods to be used/called by external users:
     *
     *   select(context, when_done)
     *      select a specific context
     *
     *   select_default()
     *      select default context as implemented in the object
     *
     *   if_ready2giveup(handler2call)
     *      check if the object's state allows to be released, and if so then
     *      call the specified function. Otherwise just ignore it. Normally
     *      this operation is used as a safeguard preventing releasing
     *      an interface focus if there is on-going unfinished editing
     *      within one of the interfaces associated with the object.
     *
     *   simple_search(pattern)
     *      search cables by the specified pattern to be found in any cable
     *      attributes.
     *
     *   search_cable_by_cablenumber(cablenumber)
     *      search a specific cable by its cable number
     *
     *   search_cables_by_prefix(prefix)
     *      search cables whose numbers begin with the specified prefix
     *
     *   search_cables_by_jobnumber(jobnumber)
     *      search cables associated with the specified job number
     *
     * -------------------------------------------------------------------------
     */
    this.name      = 'search';
    this.full_name = 'Search';
    this.context   = '';
    this.default_context = 'cables';

    this.select = function(context, when_done) {
        this.context   = context;
        this.when_done = when_done;
        this.init();
    };
    this.select_default = function() {
        if( this.context == '' ) this.context = this.default_context;
        this.init();
    };
    this.if_ready2giveup = function(handler2call) {
        this.init();
        handler2call();
    };
    this.simple_search = function(pattern) {
        this.init();
        that.search_cables_reset();
        this.simple_search_impl(pattern);
    };
    this.search_cable_by_cablenumber = function(cablenumber) {
        this.init();
        that.search_cables_reset();
        $('#search-cables-form').find('input[name="cable"]').val(cablenumber);
        this.search_cables_impl({cablenumber:cablenumber});
    };
    this.search_cable_by_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({id:id});
    };
    this.search_cables_by_prefix = function(prefix) {
        this.init();
        that.search_cables_reset();
        $('#search-cables-form').find('input[name="cable"]').val(prefix);
        this.search_cables_impl({prefix:prefix});
    };
    this.search_cables_by_cablenumber_range = function(range_id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({cablenumber_range_id:range_id});
    };
    this.search_cables_by_jobnumber = function(jobnumber) {
        this.init();
        that.search_cables_reset();
        $('#search-cables-form').find('input[name="job"]').val(jobnumber);
        this.search_cables_impl({jobnumber:jobnumber});
    };
    this.search_cables_by_jobnumber_prefix = function(prefix) {
        this.init();
        that.search_cables_reset();
        $('#search-cables-form').find('input[name="job"]').val(prefix);
        this.search_cables_impl({partial_job:prefix});
    };
    this.search_cables_by_dict_cable_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_cable_id:id});
    };
    this.search_cables_by_dict_connector_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_connector_id:id});
    };
    this.search_cables_by_dict_pinlist_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_pinlist_id:id});
    };
    this.search_cables_by_dict_location_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_location_id:id});
    };
    this.search_cables_by_dict_rack_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_rack_id:id});
    };
    this.search_cables_by_dict_routing_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_routing_id:id});
    };
    this.search_cables_by_dict_instr_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_instr_id:id});
    };
    this.search_cables_by_dict_device_location_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_device_location_id:id});
    };
    this.search_cables_by_dict_device_region_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_device_region_id:id});
    };
    this.search_cables_by_dict_device_component_id = function(id) {
        this.init();
        that.search_cables_reset();
        this.search_cables_impl({dict_device_component_id:id});
    };

    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
    this.initialized = false;
    this.init = function() {
        if( this.initialized ) return;
        this.initialized = true;
        $('#search-cables-search').button().click(function() { that.search_cables(); });
        $('#search-cables-reset' ).button().click(function() { that.search_cables_reset(); });
        $('#search-cables-form').find('input')
        .keyup(function(e) {
            if( $(this).val() == '' ) { return; }
            if( e.keyCode == 13     ) { that.search_cables(); }
        });
        that.cols2display = {
            project:  true,
            job:      true,
            cable:    true,
            device:   true,
            func:     true,
            length:   true,
            routing:  true,
            sd:       false,
            modified: false
        };
        $('#search-cables-display').find('input').change( function() {
            that.cols2display[this.name] = this.checked;
            that.display_cables();
        });
        $('#search-cables-display').find('select[name="sort"]').change( function() {
            that.sort_cables();
            that.display_cables();
        });
        $('#search-cables-display').find('button[name="reverse"]').
            button().
            click(function() {
                that.cable.reverse();
                that.display_cables();
            }
        );
        $('#search-cables').find('.export:button').
            button().
            click(function() {
                global_export_cables(that.params, this.name);
            }
        );
        this.enable_export_tools_if(false);
    };
    this.enable_export_tools_if = function(is_set) {
        $('#search-cables').find('.export:button').
            button(is_set?'enable':'disable');
    };
    this.cable = null;
    this.proj_id2title = null;
    this.cols2display = null;
    this.cable2html = function(cidx) {
        var c = this.cable[cidx];
        var html =
'<tr id="search-cables-'+cidx+'-1" class="table_row ">'+
'  <td nowrap="nowrap" class="table_cell table_cell_left table_cell_bottom "><div class="status"><b>'+c.status+'</b></div></td>';
        html += this.cols2display.project ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom ">&nbsp;<a class="link" href="javascript:global_search_project_by_id('+c.proj.id+');">'+this.proj_id2title[c.proj.id]+'</a></td>' : '';
        html += this.cols2display.job ?
'  <td nowrap="nowrap" class="table_cell table_cell_left table_cell_bottom "><div class="job"            >&nbsp;'+c.job            +'</div></td>' : '';
        html += this.cols2display.cable ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="cable"          >&nbsp;'+c.cable          +'</div></td>' : '';
        html += this.cols2display.device ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="device"         >&nbsp;'+c.device         +'</div></td>' : '';
        html += this.cols2display.func ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="func"           >&nbsp;'+c.func           +'</div></td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="cable_type"     >&nbsp;'+c.cable_type     +'</div></td>';
        html += this.cols2display.length ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="length"         >&nbsp;'+c.length         +'</div></td>' : '';
        html += this.cols2display.routing ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="routing"        >&nbsp;'+c.routing        +'</div></td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_name"    >&nbsp;'+c.origin.name    +'</div></td>';
        html += this.cols2display.sd ?
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_loc"     >&nbsp;'+c.origin.loc     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_rack"    >&nbsp;'+c.origin.rack    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_ele"     >&nbsp;'+c.origin.ele     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_side"    >&nbsp;'+c.origin.side    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_slot"    >&nbsp;'+c.origin.slot    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_conn"    >&nbsp;'+c.origin.conn    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_station" >&nbsp;'+c.origin.station +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_conntype">&nbsp;'+c.origin.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_pinlist" >&nbsp;'+dict.pinlist2url(c.origin.pinlist)+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_instr"   >&nbsp;'+c.origin.instr   +'</div></td>' : '';
        html += this.cols2display.modified ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom                 "><div class="modified"       >&nbsp;'+c.modified.time  +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom table_cell_right"><div class="modified_uid"   >&nbsp;'+c.modified.uid   +'</div></td>' : '';        html +=
'</tr>'+
'<tr id="search-cables-'+cidx+'-2" class="table_row ">'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left " align="right">&nbsp;</td>';

        html += this.cols2display.project ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left ">&nbsp;</td>' : '';
        html += this.cols2display.job ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left ">&nbsp;</td>' : '';
        html += this.cols2display.cable ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html += this.cols2display.device ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html += this.cols2display.func ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>';
        html += this.cols2display.length ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html += this.cols2display.routing ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_name"    >&nbsp;'+c.destination.name    +'</div></td>';
        html += this.cols2display.sd ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_loc"     >&nbsp;'+c.destination.loc     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_rack"    >&nbsp;'+c.destination.rack    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_ele"     >&nbsp;'+c.destination.ele     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_side"    >&nbsp;'+c.destination.side    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_slot"    >&nbsp;'+c.destination.slot    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_conn"    >&nbsp;'+c.destination.conn    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_station" >&nbsp;'+c.destination.station +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_conntype">&nbsp;'+c.destination.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_pinlist" >&nbsp;'+dict.pinlist2url(c.destination.pinlist)+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_instr"   >&nbsp;'+c.destination.instr   +'</div></td>' : '';
        html += this.cols2display.modified ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom                  ">&nbsp;</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_right ">&nbsp;</td>' : '';
        html +=
'</tr>';
        return html;
    };
    this.display_cables = function() {
        if(this.cable == null) return;
        var html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr table_hdr_tight" >'+
'      <select name="status" onchange="search.select_cables_by_status()">'+
'        <option>- status -</option>'+
'        <option>Planned</option>'+
'        <option>Registered</option>'+
'        <option>Labeled</option>'+
'        <option>Fabrication</option>'+
'        <option>Ready</option>'+
'        <option>Installed</option>'+
'        <option>Commissioned</option>'+
'        <option>Damaged</option>'+
'        <option>Retired</option>'+
'      </select>'+
'    </td>';
        html += this.cols2display.project ?
'    <td nowrap="nowrap" class="table_hdr">project</td>' : '';
        html += this.cols2display.job ?
'    <td nowrap="nowrap" class="table_hdr">job #</td>' : '';
        html += this.cols2display.cable ?
'    <td nowrap="nowrap" class="table_hdr">cable #</td>' : '';
        html += this.cols2display.device ?
'    <td nowrap="nowrap" class="table_hdr">device</td>' : '';
        html += this.cols2display.func ?
'    <td nowrap="nowrap" class="table_hdr">function</td>' : '';
        html +=
'    <td nowrap="nowrap" class="table_hdr">cable type</td>';
        html += this.cols2display.length ?
'    <td nowrap="nowrap" class="table_hdr">length</td>' : '';
        html += this.cols2display.routing ?
'    <td nowrap="nowrap" class="table_hdr">routing</td>' : '';
        html +=
'    <td nowrap="nowrap" class="table_hdr">ORIGIN / DESTINATION</td>';
        html += this.cols2display.sd ?
'    <td nowrap="nowrap" class="table_hdr">loc</td>'+
'    <td nowrap="nowrap" class="table_hdr">rack</td>'+
'    <td nowrap="nowrap" class="table_hdr">ele</td>'+
'    <td nowrap="nowrap" class="table_hdr">side</td>'+
'    <td nowrap="nowrap" class="table_hdr">slot</td>'+
'    <td nowrap="nowrap" class="table_hdr">conn #</td>'+
'    <td nowrap="nowrap" class="table_hdr">station</td>'+
'    <td nowrap="nowrap" class="table_hdr">contype</td>'+
'    <td nowrap="nowrap" class="table_hdr">pinlist</td>'+
'    <td nowrap="nowrap" class="table_hdr">instr</td>' : '';
        html += this.cols2display.modified ?
'    <td nowrap="nowrap" class="table_hdr">modified</td>'+
'    <td nowrap="nowrap" class="table_hdr">by user</td>' : '';
        for( var cidx in this.cable ) html += this.cable2html(cidx);
        html +=
'  </tr>'+
'</tbody></table>';
        $('#search-cables-result').html(html);
    };
    this.sort_cables = function() {
        var sort_by = $('#search-cables-display').find('select[name="sort"]').val();
        var sorter  = null;
        switch(sort_by) {
            case "status"     : sorter = global_cable_sorter_by_status;      break;
            case "project"    : sorter = global_cable_sorter_by_project;     break;
            case "job"        : sorter = global_cable_sorter_by_job;         break;
            case "cable"      : sorter = global_cable_sorter_by_cable;       break;
            case "device"     : sorter = global_cable_sorter_by_device;      break;
            case "function"   : sorter = global_cable_sorter_by_function;    break;
            case "origin"     : sorter = global_cable_sorter_by_origin;      break;
            case "destination": sorter = global_cable_sorter_by_destination; break;
            case "modified"   : sorter = global_cable_sorter_by_modified;    break;
        }
        this.cable.sort(sorter);
    };
    this.select_cables_by_status = function() {
        var status = $('#search-cables-result').find('select[name="status"]').val();
        if( '- status -' == status ) {
            for( var cidx in this.cable ) {
                $('#search-cables-'+cidx+'-1').css('display','');
                $('#search-cables-'+cidx+'-2').css('display','');
            }
        } else {
            for( var cidx in this.cable ) {
                var style = this.cable[cidx].status == status ? '' : 'none';
                $('#search-cables-'+cidx+'-1').css('display',style);
                $('#search-cables-'+cidx+'-2').css('display',style);
            }
        }
    };
    this.params = null;
    this.search_cable_by_cablenumber_impl = function(cablenumber) {
        this.search_cables_impl({cablenumber:cablenumber});
    };
    this.search_cables = function() {
        var form = $('#search-cables-form');
        this.search_cables_impl({
            partial_cable          : form.find('input[name="cable"]').val(),
            partial_job            : form.find('input[name="job"]').val(),
            partial_cable_type     : form.find('input[name="cable_type"]').val(),
            partial_routing        : form.find('input[name="routing"]').val(),
            partial_device         : form.find('input[name="device"]').val(),
            partial_func           : form.find('input[name="func"]').val(),
            partial_origin_loc     : form.find('input[name="origin_loc"]').val(),
            partial_destination_loc: form.find('input[name="destination_loc"]').val()
        });
    };
    this.search_cables_reset = function() {
        $('#search-cables-form').find('input').each(function() {$(this).val('')});
        this.cable = [];
        this.display_cables();
        $('#search-cables-info').html('&nbsp;');
        this.enable_export_tools_if(false);
    };
    this.simple_search_impl = function(pattern) {
        this.search_cables_impl({
            partial_cable          : pattern,
            partial_job            : '',
            partial_cable_type     : '',
            partial_routing        : '',
            partial_device         : '',
            partial_func           : '',
            partial_origin_loc     : '',
            partial_destination_loc: ''
        });
    };
    this.search_cables_impl = function(params) {

        this.params = params;
        this.enable_export_tools_if(false);

        // TODO: This function needs to be reimplemented to deal with JSON object
        // returned by the Web service. Also restrict the number of cables rendered/returned
        // by the script.
        //
        $('#search-cables-search').button('disable');
        $('#search-cables-reset').button('disable');
        $('#search-cables-info').html('Searching...');
        var jqXHR = $.get(
            '../neocaptar/ws/cable_search.php', this.params,
            function(data) {
                if( data.status != 'success') {
                    report_error('failed to load cables because of: '+data.message);
                    return;
                }
                that.cable = data.cable;
                that.proj_id2title = data.proj_id2title;
                that.sort_cables();
                that.display_cables();
                $('#search-cables-info').html(data.cable.length+' cables');
                that.enable_export_tools_if(true);
            },
            'JSON'
        ).error(
            function () {
                $('#search-cables-info').html('operation failed');
                report_error('failed because of: '+jqXHR.statusText);
            }
        ).complete(
            function () {
                $('#search-cables-search').button('enable');
                $('#search-cables-reset').button('enable');
            }
        );
    };
    this.export_cables = function(outformat) {
        global_export_cables(this.params, outformat);
    };
    return this;
}
var search = new p_appl_search();
