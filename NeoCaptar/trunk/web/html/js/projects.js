
function p_appl_projects() {

    var that = this;

    this.when_done           = null;
	this.create_form_changed = false;

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
     */
	this.name      = 'projects';
	this.full_name = 'Projects';
	this.context   = '';
    this.default_context = 'search';

    this.select = function(context,when_done) {
		that.context   = context;
		this.when_done = when_done;
		this.init();
	};
	this.select_default = function() {
		if( this.context == '' ) this.context = this.default_context;
		this.init();
	};
	this.if_ready2giveup = function(handler2call) {
		if(( this.context == 'create' ) && this.create_form_changed ) {
			ask_yes_no(
				'Unsaved Data Warning',
				'You are about to leave the page while there are unsaved data in the form. Are you sure?',
				handler2call,
				null );
			return;
		}
		handler2call();
	};

    /* ----------------------------------------
     *   Internal methods and data structures
     * ----------------------------------------
     */
	this.initialized = false;
	this.init = function() {
		if( this.initialized ) return;
		this.initialized = true;

        var search_controls = $('#projects-search-controls');
		search_controls.find('input[name="title"]').change(function() { that.search() ;});
		search_controls.find('select[name="owner"]').change(function() { that.search() ;});
		search_controls.find('input[name="begin"]').datepicker().datepicker('option','dateFormat','yy-mm-dd').change(function() { that.search() ;});
		search_controls.find('input[name="end"]').datepicker().datepicker('option','dateFormat','yy-mm-dd').change(function() { that.search() ;});
        search_controls.find('button[name="search"]').button().click(function() { that.search() ;});
		search_controls.find('button[name="reset"]').button().click(function() { that.search_reset(); });

        $('#projects-create-form').find('input[name="due_time"]').
            each(function() {
                $(this).datepicker().datepicker('option','dateFormat','yy-mm-dd'); });
		$('#projects-create-save').
            button().
            button('disable').
            click(function() {
                that.create_project(); });
		$('#projects-create-reset').
            button().
            click(function() {
                $('#projects-create-save').button('disable'); });
		$('.projects-create-form-element').
            change(function() {
                that.create_form_changed = true;
                $('#projects-create-save').button('enable'); });
	};
	this.project = null;
    this.can_create_projects = function() {
        return global_current_user.is_administrator || global_current_user.can_manage_projects;
    };
    this.can_manage_project = function(pidx) {
        return global_current_user.is_administrator || (global_current_user.can_manage_projects && (global_current_user.uid == this.project[pidx].owner));
    }
    this.can_define_new_types = function() {
        return global_current_user.is_administrator;
    };
    this.can_manage_workflow = function() {
        return global_current_user.is_administrator;
    };
	this.toggle_project = function(pidx) {
		var toggler='#proj-tgl-'+pidx;
		var container='#proj-con-'+pidx;
		if( $(container).hasClass('proj-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('proj-hdn').addClass('proj-vis');
            var proj = this.project[pidx];
			if( !proj.is_loaded ) {
				proj.is_loaded = true;
				$('#proj-edit-'+pidx ).
                    button().
                    button(this.can_manage_project(pidx)?'enable':'disable').
                    click(function() { that.edit_attributes(pidx); });
				$('#proj-delete-'+pidx ).
                    button().
                    button(this.can_manage_project(pidx)?'enable':'disable').
                    click(function() { that.delete_project(pidx); });
				$('#proj-clone-'+pidx ).
                    button().
                    button(this.can_create_projects()?'enable':'disable').
                    click(function() { that.clone_project(pidx); });
				$('#proj-history-'+pidx ).
                    button().
                    click(function() { that.show_project_history(pidx); });
				$('#proj-add-'+pidx ).
                    button().
                    button(this.can_manage_project(pidx)?'enable':'disable').
                    click(function() { that.add_cable(pidx); });
				$('#proj-label-'+pidx).
                    button().
                    click(function() { that.show_cable_label(pidx); });
				this.load_cables(pidx);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('proj-vis').addClass('proj-hdn');
		}
	};
    this.edit_attributes = function(pidx) {
        var proj = this.project[pidx];
        var html =
'        <table id="proj-editor-'+pidx+'" style="font-size:95%;"><tbody>'+
'          <tr>'+
'            <td><b>Owner: </b></td>'+
'            <td><select name="owner" style="padding:1px;" '+(global_current_user.is_administrator?'':' disabled="disabled"')+'>';
        var users = global_get_projmanagers();
        for( var i in users ) {
            var user = users[i];
            html +=
'                  <option '+(proj.owner==user?'selected="selected"':'')+'>'+user+'</option>';
        }
        html +=
'                </select></td>'+
'            <td><b>Title: </b></td>'+
'            <td><input type="text" size="36" name="title" value="'+proj.title+'" style="padding:1px;"></td>'+
'            <td></td>'+
'            <td><b>Due by: </b></td>'+
'            <td><input type="text" size="10" name="due" value="'+proj.due+'" style="padding:1px;"></td>'+
'          </tr>'+
'          <tr>'+
'            <td><b>Descr: </b></td>'+
'            <td colspan="6"><textarea cols=64 rows=4 name="description" style="padding:4px;" title="Here be the project description">'+proj.description+'</textarea></td>'+
'          </tr>'+
'        </tbody></table>';
        edit_dialog(
            'Edit Project Attributes',
            html,
            function() { that.edit_attributes_save(pidx); },
            function() { return; } );
        $('#proj-editor-'+pidx).find('input[name="due"]').
            datepicker().
            datepicker('option','dateFormat','yy-mm-dd').
            datepicker('setDate',proj.due);
    }
    this.edit_attributes_save = function(pidx) {

        var proj = this.project[pidx];

        // First we need to submit the request to the database service in order
		// to get an approval for the proposed changes. If the operation succeeds
        // then we can also make proper adjustmenens to the current state of the GUI.
		//
        var elem = $('#proj-editor-'+pidx);
        $.ajax({
            type: 'POST',
            url: '../neocaptar/project_save.php',
            data: {
                id:          proj.id,
                owner:       elem.find('select[name="owner"]').val(),
                title:       elem.find('input[name="title"]').val(),
                due_time:    elem.find('input[name="due"]').val(),
                description: elem.find('textarea[name="description"]').val()},
            success: function(data) {
                if( data.status != 'success' ) { report_error(data.message); return; }
                var new_proj = data.project;
                proj.owner       = new_proj.owner;
                proj.title       = new_proj.title;
                proj.due         = new_proj.due;
                proj.description = new_proj.description;
                var hdr = $('#proj-hdr-'+pidx);
                hdr.find('.proj-owner').html(proj.owner);
                hdr.find('.proj-title').html(proj.title);
                hdr.find('.proj-due'  ).html(proj.due);
                $('#proj-con-'+pidx).find('.proj-description').text(proj.description);
            },
            error: function() {	report_error('The request can not go through due a failure to contact the server.'); },
            dataType: 'json'
        });
    };
	this.cable_action2html = function(pidx,cidx,reverse) {
		var c = this.project[pidx].cable[cidx];
		var html = '';
        if(reverse) {
            switch(c.status) {
                case 'Registered':   html = '<button class="proj-cable-tool" name="un_register"   onclick="projects.un_register_cable  ('+pidx+','+cidx+')" title="unregister by releasing cable number" ><b>UNREG</b></button>'; break;
                case 'Labeled':      html = '<button class="proj-cable-tool" name="un_label"      onclick="projects.un_label_cable     ('+pidx+','+cidx+')" title="unlock the label"                     ><b>UNLBL</b></button>'; break;
                case 'Fabrication':  html = '<button class="proj-cable-tool" name="un_fabricate"  onclick="projects.un_fabricate_cable ('+pidx+','+cidx+')" title="back to the locked label state"       ><b>UNFAB</b></button>'; break;
                case 'Ready':        html = '<button class="proj-cable-tool" name="un_ready"      onclick="projects.un_ready_cable     ('+pidx+','+cidx+')" title="back to fabrication"                  ><b>UNRDY</b></button>'; break;
                case 'Installed':    html = '<button class="proj-cable-tool" name="un_install"    onclick="projects.un_install_cable   ('+pidx+','+cidx+')" title="uninstall"                            ><b>UN-INS</b></button>'; break;
                case 'Commissioned': html = '<button class="proj-cable-tool" name="un_commission" onclick="projects.un_commission_cable('+pidx+','+cidx+')" title="decommission"                         ><b>UN-COM</b></button>'; break;
                case 'Damaged':      html = '<button class="proj-cable-tool" name="un_damage"     onclick="projects.un_damage_cable    ('+pidx+','+cidx+')" title="back to the commissioned state"       ><b>UN-DMG</b></button>'; break;
                case 'Retired':      html = '<button class="proj-cable-tool" name="un_retire"     onclick="projects.un_retire_cable    ('+pidx+','+cidx+')" title="back to the damaged state"            ><b>UN-RTR</b></button>'; break;
            }
        } else {
            switch(c.status) {
                case 'Planned':      html = '<button class="proj-cable-tool" name="register"   onclick="projects.register_cable  ('+pidx+','+cidx+')" title="proceed to the formal registration to obtain official JOB and CABLE numbers" ><b>REG</b></button>'; break;
                case 'Registered':   html = '<button class="proj-cable-tool" name="label"      onclick="projects.label_cable     ('+pidx+','+cidx+')" title="produce a standard label"                                                    ><b>LBL</b></button>'; break;
                case 'Labeled':      html = '<button class="proj-cable-tool" name="fabricate"  onclick="projects.fabricate_cable ('+pidx+','+cidx+')" title="order fabrication"                                                           ><b>FAB</b></button>'; break;
                case 'Fabrication':  html = '<button class="proj-cable-tool" name="ready"      onclick="projects.ready_cable     ('+pidx+','+cidx+')" title="get ready for installation"                                                  ><b>RDY</b></button>'; break;
                case 'Ready':        html = '<button class="proj-cable-tool" name="install"    onclick="projects.install_cable   ('+pidx+','+cidx+')" title="install"                                                                     ><b>INS</b></button>'; break;
                case 'Installed':    html = '<button class="proj-cable-tool" name="commission" onclick="projects.commission_cable('+pidx+','+cidx+')" title="test and commission"                                                         ><b>COM</b></button>'; break;
                case 'Commissioned': html = '<button class="proj-cable-tool" name="damage"     onclick="projects.damage_cable    ('+pidx+','+cidx+')" title="mark as damaged, needing replacement"                                        ><b>DMG</b></button>'; break;
                case 'Damaged':      html = '<button class="proj-cable-tool" name="retire"     onclick="projects.retire_cable    ('+pidx+','+cidx+')" title="mark as retired"                                                             ><b>RTR</b></button>'; break;
            }
        }
        return html;
    };
	this.cable2html = function(pidx,cidx) {
        var proj = this.project[pidx];
        var cols2display = proj.cols2display;
		var c = proj.cable[cidx];
		var html =
'<tr class="table_row " id="proj-cable-'+pidx+'-'+cidx+'-1">'+
'  <td nowrap="nowrap" class="table_cell table_cell_left table_cell_bottom "><div class="status"><b>'+c.status+'</b></div></td>';
        html += cols2display.tools ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom " id="proj-cable-tools-'          +pidx+'-'+cidx+'-1" >'+
'    <button class="proj-cable-tool" name="edit"        onclick="projects.edit_cable       ('+pidx+','+cidx+')" title="edit"                                 ><b>E</b></button>'+
'  </td>' : '';
        html += cols2display.project ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom ">&nbsp;<a class="link" href="javascript:global_search_project_by_id('+proj.id+');">'+proj.title+'</a></td>' : '';
        html += cols2display.cable ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="cable"          >&nbsp;'+c.cable          +'</div></td>' : '';
        html += cols2display.device ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="device"         >&nbsp;'+c.device         +'</div></td>' : '';
        html += cols2display.func ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="func"           >&nbsp;'+c.func           +'</div></td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="cable_type"     >&nbsp;'+c.cable_type     +'</div></td>';
        html += cols2display.length ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="length"         >&nbsp;'+c.length         +'</div></td>' : '';
        html += cols2display.routing ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "                ><div class="routing"        >&nbsp;'+c.routing        +'</div></td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_name"    >&nbsp;'+c.origin.name    +'</div></td>';
        html += cols2display.sd ?
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_loc"     >&nbsp;'+c.origin.loc     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_rack"    >&nbsp;'+c.origin.rack    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_ele"     >&nbsp;'+c.origin.ele     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_side"    >&nbsp;'+c.origin.side    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_slot"    >&nbsp;'+c.origin.slot    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_conn"    >&nbsp;'+c.origin.conn    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_station" >&nbsp;'+c.origin.station +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_conntype">&nbsp;'+c.origin.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_pinlist" >&nbsp;'+dict.pinlist2url(c.cable_type,c.origin.conntype,c.origin.pinlist)+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_instr"   >&nbsp;'+c.origin.instr   +'</div></td>' : '';
        html += cols2display.modified ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom                 "><div class="modified"       >&nbsp;'+c.modified.time  +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom table_cell_right"><div class="modified_uid"   >&nbsp;'+c.modified.uid   +'</div></td>' : '';
        html +=
'</tr>'+
'<tr class="table_row " id="proj-cable-'+pidx+'-'+cidx+'-2">';
        html += cols2display.tools ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left " align="right">'+
'    <div style="float:left; margin-right:3px;" id="proj-cable-action-'+pidx+'-'+cidx+'-reverse">'+this.cable_action2html(pidx,cidx,true )+'</div>'+
'    <div style="float:left;" id="proj-cable-action-'+pidx+'-'+cidx+'"        >'+this.cable_action2html(pidx,cidx,false)+'</div>'+
'    <div style="clear:both;"></div>'+
'  </td>' :
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left " align="right">&nbsp;</td>';
        html += cols2display.tools ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "id="proj-cable-tools-' +pidx+'-'+cidx+'-2" >'+
'    <button class="proj-cable-tool" name="clone"   onclick="projects.clone_cable       ('+pidx+','+cidx+')" title="clone"  ><b>C</b></button>'+
'    <button class="proj-cable-tool" name="delete"  onclick="projects.delete_cable      ('+pidx+','+cidx+')" title="delete" ><b>D</b></button>'+
'    <button class="proj-cable-tool" name="history" onclick="projects.show_cable_history('+pidx+','+cidx+')" title="history"><b>H</b></button>'+
'    <button class="proj-cable-tool" name="label"   onclick="projects.show_cable_label  ('+pidx+','+cidx+')" title="label"  ><b>L</b></button>'+
'  </td>' : '';
        html += cols2display.project ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left ">&nbsp;</td>' : '';
        html += cols2display.cable ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html += cols2display.device ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html += cols2display.func ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>';
        html += cols2display.length ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html += cols2display.routing ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom ">&nbsp;</td>' : '';
        html +=
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_name"    >&nbsp;'+c.destination.name    +'</div></td>';
        html += cols2display.sd ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_loc"     >&nbsp;'+c.destination.loc     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_rack"    >&nbsp;'+c.destination.rack    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_ele"     >&nbsp;'+c.destination.ele     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_side"    >&nbsp;'+c.destination.side    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_slot"    >&nbsp;'+c.destination.slot    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_conn"    >&nbsp;'+c.destination.conn    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_station" >&nbsp;'+c.destination.station +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_conntype">&nbsp;'+c.destination.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_pinlist" >&nbsp;'+dict.pinlist2url(c.cable_type,c.destination.conntype,c.destination.pinlist)+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_instr"   >&nbsp;'+c.destination.instr   +'</div></td>' : '';
        html += cols2display.modified ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom                  ">&nbsp;</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_right ">&nbsp;</td>' : '';
        html +=
'</tr>';
		return html;
	};
	this.display_cables = function(pidx) {
        var proj = this.project[pidx];
        var cols2display = proj.cols2display;
        var html =
'<table><tbody>'+
'  <tr class="table_first_header" id="proj-cables-hdr-'+pidx+'">'+
'    <td nowrap="nowrap" class="table_hdr table_hdr_tight" >'+
'      <select name="status" onchange="projects.select_cables_by_status('+pidx+')">'+
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
        html += cols2display.tools ?
'    <td nowrap="nowrap" class="table_hdr">OPERATIONS</td>' : '';
        html += cols2display.project ?
'    <td nowrap="nowrap" class="table_hdr">project</td>' : '';
        html += cols2display.cable ?
'    <td nowrap="nowrap" class="table_hdr">cable #</td>' : '';
        html += cols2display.device ?
'    <td nowrap="nowrap" class="table_hdr">device</td>' : '';
        html += cols2display.func ?
'    <td nowrap="nowrap" class="table_hdr">function</td>' : '';
        html +=
'    <td nowrap="nowrap" class="table_hdr">cable type</td>';
        html += cols2display.length ?
'    <td nowrap="nowrap" class="table_hdr">length</td>' : '';
        html += cols2display.routing ?
'    <td nowrap="nowrap" class="table_hdr">routing</td>' : '';
        html +=
'    <td nowrap="nowrap" class="table_hdr">ORIGIN / DESTINATION</td>';
        html += cols2display.sd ?
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
        html += cols2display.modified ?
'    <td nowrap="nowrap" class="table_hdr">modified</td>'+
'    <td nowrap="nowrap" class="table_hdr">by user</td>' : '';
        html +=
'  </tr>'+
'</tbody></table>';
        $('#proj-con-'+pidx).find('div.table').html(html);
		var prev = $('#proj-cables-hdr-'+pidx);
		prev.siblings().remove();
		for( var cidx in proj.cable ) {
			var html = this.cable2html(pidx, cidx);
			prev.after( html );
			prev = $('#proj-cable-'+pidx+'-'+cidx+'-2');
		}
		$('.proj-cable-tool').button();
		for( var cidx in proj.cable )
			this.update_cable_tools(pidx,cidx);
	};
	this.load_cables = function(pidx) {
		$('#proj-cables-load-'+pidx).html('Loading...');
		var params = {project_id:this.project[pidx].id};
		var jqXHR = $.get(
			'../neocaptar/cable_search.php', params,
			function(data) {
				if( data.status != 'success') {
					report_error('failed to load cables because of: '+data.message);
					return;
				}
				that.project[pidx].cable = data.cable;
                that.sort_cables(pidx);
				that.display_cables(pidx);
			},
			'JSON'
		).error(
			function () {
				report_error('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
				$('#proj-cables-load-'+pidx).html('');
			}
		);
	};
	this.update_cable_tools = function(pidx,cidx) {
		var cable = this.project[pidx].cable[cidx];
        $('#proj-cable-action-'+pidx+'-'+cidx+'-reverse button').button('disable');
        $('#proj-cable-action-'+pidx+'-'+cidx+' button'        ).button('disable');
        var tools_1 = $('#proj-cable-tools-'+pidx+'-'+cidx+'-1');
        var tools_2 = $('#proj-cable-tools-'+pidx+'-'+cidx+'-2');
		tools_2.find('button[name="history"]').button('enable');
        tools_2.find('button[name="label"]' ).button(
            global_cable_status2rank(cable.status) > global_cable_status2rank('Planned') ?
            'enable':
            'disable');
        if(!this.can_manage_project(pidx)) {
            tools_1.find('button[name="edit"]'  ).button('disable');
			tools_2.find('button[name="clone"]' ).button('disable');
            tools_2.find('button[name="delete"]').button('disable');
        } else {
            if(this.can_manage_workflow()) {
                $('#proj-cable-action-'+pidx+'-'+cidx+'-reverse button').button('enable');
                $('#proj-cable-action-'+pidx+'-'+cidx+' button'        ).button('enable');
            }
            tools_1.find('button[name="edit"]').button(
                global_cable_status2rank(cable.status) < global_cable_status2rank('Fabrication') ?
                'enable':
                'disable');
            tools_2.find('button[name="clone"]'  ).button('enable');
            tools_2.find('button[name="delete"]').button(
                global_cable_status2rank(cable.status) < global_cable_status2rank('Labeled') ?
                'enable':
                'disable');
        }
	};
    this.update_project_hdr = function(pidx) {
        // TODO: Scan over all cables and rebuild the project object and visible
        // header. For now just update the modification time.
        //
        var proj = this.project[pidx];

        proj.status.total = 0;
        for( var status in proj.status ) proj.status[status] = 0;

        for( var cidx in proj.cable ) {
            var cable = proj.cable[cidx];
            if( cable.modified_time_sec > proj.modified_sec ) {
                proj.modified_sec = cable.modified_time_sec;
                proj.modified     = cable.modified_time;
            }
            proj.status[''+cable.status+'']++;
            proj.status.total++;
        }
        $('#proj-hdr-'+pidx+' div.proj-modified').html(proj.modified);
        $('#proj-hdr-'+pidx+' div.proj-num-cables-total').html(''+proj.status.total+'');
        for( var status in proj.status )
            $('#proj-hdr-'+pidx+' div.'+status).html(proj.status[status]);
    };
	this.add_cable = function(pidx) {
		var proj = this.project[pidx];

        // First we need to submit the request to the database service in order
		// to get a unique cable identifier.
		//
        $.ajax({
            type: 'POST',
            url: '../neocaptar/cable_new.php',
            data: {project_id:proj.id},
            success: function(data) {
                if( data.status != 'success' ) { report_error(data.message); return; }
				var cidx_new = proj.cable.push(data.cable) - 1;
				$('#proj-cables-hdr-'+pidx).after( that.cable2html(pidx, cidx_new))
				$('#proj-cable-'+pidx+'-'+cidx_new+'-1').find('.proj-cable-tool').each( function() { $(this).button(); });
				$('#proj-cable-'+pidx+'-'+cidx_new+'-2').find('.proj-cable-tool').each( function() { $(this).button(); });
				that.edit_cable(pidx,cidx_new);
            },
            error: function() {	report_error('The request can not go through due a failure to contact the server.'); },
            dataType: 'json'
        });
	};
    this.set_project2clone = function(title) {
        $('#projects-create-form').find('input[name="project2clone"]').val(title);
    };
    this.create_project = function() {
        var form = $('#projects-create-form');
        var owner       = form.find('input[name="owner"]').val();
        var title       = form.find('input[name="title"]').val();
        var description = form.find('textarea[name="description"]').val();
        var due_time    = form.find('input[name="due_time"]').val();
        if( owner === '' || title === '' || due_time === '' ) {
            report_error('One of the required fields is empty. Please correct the form and submit again.', null);
            return;
        }
        $('#projects-create-save').button('disable');
        $('#projects-create-info').html('Saving...');
        var params = {
            project2clone: form.find('input[name="project2clone"]').val(),
            owner:         owner,
            title:         title,
            description:   description,
            due_time:      due_time};
        var jqXHR = $.get('../neocaptar/project_new.php',params,function(data) {
            $('#projects-create-info').html('&nbsp;');
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.create_form_changed = false;
            if( that.when_done ) {
                that.when_done.execute();
                that.when_done = null;
            }
            var proj = data.project;
            global_search_project_by_id(proj.id);
            form.find('input[name="project2clone"]').val('');
            form.find('input[name="owner"]').val(global_current_user.uid);
            form.find('input[name="title"]').val('');
            form.find('textarea[name="description"]').val('');
            form.find('input[name="due_time"]').val('');
        },
        'JSON').error(function () {
            $('#projects-create-save').button('enable');
            $('#projects-create-info').html('&nbsp;');
            report_error('saving failed because of: '+jqXHR.statusText, null);
        });
    };
	this.delete_project = function(pidx) {
		ask_yes_no(
			'Data Deletion Warning',
			'You are about to delete the project and all cables associated with it. Are you sure?',
			function() {
                var params = {project_id:that.project[pidx].id};
                var jqXHR = $.get('../neocaptar/project_delete.php',params,function(data) {
                    if(data.status != 'success') { report_error(data.message, null); return; }
                    delete that.project[pidx];
                    that.display();
                },
                'JSON').error(function () {
                    report_error('failed to delete the project because of: '+jqXHR.statusText, null);
                    return;
                });
			},
			null );
	};
	this.clone_project = function(pidx) {
        this.set_project2clone(this.project[pidx].title);
        global_switch_context('projects','create');
    };
    this.show_project_history = function(pidx) {
        var params = {id:this.project[pidx].id};
        var jqXHR = $.get('../neocaptar/project_history.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.show_history('Project History',data.event);
        },
        'JSON').error(function () {
            report_error('failed to obtain the project history because of: '+jqXHR.statusText, null);
            return;
        });
    }
	this.show_cable_history = function(pidx,cidx) {
        var params = {id:this.project[pidx].cable[cidx].id};
        var jqXHR = $.get('../neocaptar/cable_history.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.show_history('Cable History',data.event);
        },
        'JSON').error(function () {
            report_error('failed to obtain the cable history because of: '+jqXHR.statusText, null);
            return;
        });
    };
	this.show_history = function(title,events) {
        var rows = [];
        for( var i in events) {
            var event = events[i];
            var comments = '';
            for( var j in event.comments ) comments += '<div>'+event.comments[j]+'</div>';
            rows.push( [event.event_time, event.event_uid, event.event, comments] );
        }
        report_info_table(
            title,
            [ { name: 'time'  },
              { name: 'user'  },
              { name: 'event' },
              { name: 'comments', sorted: false }],
            rows
        );
    };
	this.clone_cable = function(pidx,cidx) {
		var proj  = this.project[pidx];
		var cable = proj.cable[cidx];

        // First we need to submit the request to the database service in order
		// to get a unique cable identifier.
		//
        $.ajax({
            type: 'POST',
            url: '../neocaptar/cable_new.php',
            data: {cable_id: cable.id},
            success: function(data) {
                if( data.status != 'success' ) { report_error(data.message); return; }
				var cidx_new = proj.cable.push(data.cable) - 1;
				$('#proj-cables-hdr-'+pidx).after( that.cable2html(pidx, cidx_new))
				$('#proj-cable-'+pidx+'-'+cidx_new+'-1').find('.proj-cable-tool').each( function() { $(this).button(); });
				$('#proj-cable-'+pidx+'-'+cidx_new+'-2').find('.proj-cable-tool').each( function() { $(this).button(); });
				that.edit_cable(pidx,cidx_new);
            },
            error: function() {	report_error('The request can not go through due a failure to contact the server.'); },
            dataType: 'json'
        });
	};
	this.delete_cable = function(pidx,cidx) {
		ask_yes_no(
			'Data Deletion Warning',
			'You are about to delete the cable and all information associated with it. Are you sure?',
			function() {
				var proj  = that.project[pidx];
				var cable = proj.cable[cidx];
                var params = {cable_id:cable.id};
                var jqXHR = $.get('../neocaptar/cable_delete.php',params,function(data) {
                    var result = eval(data);
                    if(result.status != 'success') { report_error(result.message, null); return; }
                    //
                    // Remove it from the list and from the table
                    //
                    $('#proj-cable-'+pidx+'-'+cidx+'-1').remove();
                    $('#proj-cable-'+pidx+'-'+cidx+'-2').remove();
                    proj.status.total--;
                    switch(cable.status) {
                    case 'Planned':      proj.status.Planned--;      break;
                    case 'Registered':   proj.status.Registered--;   break;
                    case 'Labeled':      proj.status.Labeled--;      break;
                    case 'Fabrication':  proj.status.Fabrication--;  break;
                    case 'Ready':        proj.status.Ready--;        break;
                    case 'Installed':    proj.status.Installed--;    break;
                    case 'Commissioned': proj.status.Commissioned--; break;
                    case 'Damaged':      proj.status.Damaged--;      break;
                    case 'Retired':      proj.status.Retired--;      break;
                    }
                    delete proj.cable[cidx];
                    //
                    // Update the project header
                    //
                    that.update_project_hdr(pidx);
                },
                'JSON').error(function () {
                    report_error('saving failed because of: '+jqXHR.statusText, null);
                });
			},
			null );
	};
	this.cable_property_edit = function(pidx,cidx,prop,is_new,is_disabled) {

        var cable  = this.project[pidx].cable[cidx];
        var is_disabled_attr = is_disabled ? 'disabled="disabled"' : '';

        cable.editing = true;

        var base = $('#proj-con-'+pidx+'-shadow').find('.'+prop);

        var read_input  = function() { return base.find('input' ).val(); };
        var read_select = function() { return base.find('select').val(); };

        var html  = '';

		switch(prop) {

		case 'device':

            // Thsi field can't be edited directly. INstead it's composed
            // of its components which are edited via separate inputs (see below).
            // 
            cable.read_device = function() {
                var location  = global_truncate_device_location (cable.read_device_location ());
                var region    = global_truncate_device_region   (cable.read_device_region   ());
                var component = global_truncate_device_component(cable.read_device_component());
                var counter   = global_truncate_device_counter  (cable.read_device_counter  ());
                var suffix    = global_truncate_device_suffix   (cable.read_device_suffix   ());
                var device    = ( location  == '' ? '???'  :     location  ) +
                                ( region    == '' ? '-???' : '-'+region    ) +
                                ( component == '' ? '-???' : '-'+component ) +
                                ( counter   == '' ? '-??'  : '-'+counter   ) +
                                ( suffix    == '' ? ''     : '-'+suffix    );
                return device;
            };
            break;

		case 'device_location':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.read_device_location = read_input;

			} else if( dict.device_location_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.device_location+'" size="4" '+is_disabled_attr+' />');
				cable.read_device_location = read_input;

			} else {

				html = '<select '+is_disabled_attr+' >';
				if( cable.device_location == '' ) {
					for( var t in dict.device_locations()) {
						html += '<option';
						if( t == cable.device_location ) html += ' selected="selected"';
						html += ' value="'+t+'">'+t+'</option>';
					}
				} else {

					if( dict.device_location_is_not_known( cable.device_location )) {

						html += '<option selected="selected" value="'+cable.device_location+'">'+cable.device_location+'</option>';
						for( var t in dict.device_locations())
							html += '<option value="'+t+'">'+t+'</option>';

					} else {

						for( var t in dict.device_locations()) {
							html += '<option';
							if( t == cable.device_location ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
					}
				}
                if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new device location" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
						that.cable_property_edit(pidx,cidx,'device_region',true,false);
						that.cable_property_edit(pidx,cidx,'device_component',true,false);
					} else {
						that.cable_property_edit(pidx,cidx,'device_region',false,false);
						that.cable_property_edit(pidx,cidx,'device_component',false,false);
					}
				});
				cable.read_device_location = read_select;
			}
			break;

		case 'device_region':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.read_device_region = read_input;

			} else if( dict.device_location_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.device_region+'" size="4" '+is_disabled_attr+' />');
				cable.read_device_region = read_input;

			} else {

				if(	dict.device_location_is_not_known( cable.read_device_location())) {

					base.html('<input type="text" value="'+cable.device_region+'" size="4" '+is_disabled_attr+' />');
					cable.read_device_region = read_input;

				} else {

					if( dict.device_region_dict_is_empty( cable.read_device_location())) {

						base.html('<input type="text" value="'+cable.device_region+'" size="4" '+is_disabled_attr+' />');
						cable.read_device_region = read_input;

					} else {

						html = '<select '+is_disabled_attr+' >';
						for( var t in dict.device_regions( cable.read_device_location())) {
							html += '<option';
							if( t == cable.device_region ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
						if(this.can_define_new_types()) html += '<option value=""  style="color:maroon;" title="register new device region for the given device location" >New...</option>'
						html += '</select>';
						base.html(html);
						base.find('select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property_edit(pidx,cidx,prop,true,false);
								that.cable_property_edit(pidx,cidx,'device_component',true,false);
							} else {
								that.cable_property_edit(pidx,cidx,'device_component',false,false);
							}
						});
						cable.read_device_region = read_select;
					}
				}
			}
            break;

		case 'device_component':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.read_device_component = read_input;

			} else if( dict.device_location_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.device_component+'" size="4" '+is_disabled_attr+' />');
				cable.read_device_component = read_input;

			} else if( dict.device_location_is_not_known( cable.read_device_location())) {

				base.html('<input type="text" value="'+cable.device_component+'" size="4" '+is_disabled_attr+' />');
				cable.read_device_component = read_input;

			} else if( dict.device_region_is_not_known( cable.read_device_location(), cable.read_device_region())) {

				base.html('<input type="text" value="'+cable.device_component+'" size="4" '+is_disabled_attr+' />');
				cable.read_device_component = read_input;

			} else {

				if( dict.device_component_dict_is_empty( cable.read_device_location(), cable.read_device_region())) {

					base.html('<input type="text" value="'+cable.device_component+'" size="4" '+is_disabled_attr+' />');
					cable.read_device_component = read_input;

				} else {

					html = '<select '+is_disabled_attr+' >';
					for( var device_component in dict.device_components( cable.read_device_location(), cable.read_device_region())) {
						html += '<option';
						if( device_component == cable.device_component ) html += ' selected="selected"';
						html += ' value="'+device_component+'">'+device_component+'</option>';
					}
					if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new device component for the given device region" >New...</option>'
					html += '</select>';
					base.html(html);
					base.find('select').change(function() {
						if( $(this).val() == '') {
							that.cable_property_edit(pidx,cidx,prop,true,false);
						}
					});
					cable.read_device_component = read_select;
				}
			}
			break;

        case 'device_counter':
			base.html('<input type="text" value="'+cable.device_counter+'" size="2" '+is_disabled_attr+' />');
            cable.read_device_counter = read_input;
			break;

        case 'device_suffix':
			base.html('<input type="text" value="'+cable.device_suffix+'" size="2" '+is_disabled_attr+' />');
            cable.read_device_suffix = read_input;
			break;

        case 'func':
			base.html('<input type="text" value="'+cable.func+'" size="24" '+is_disabled_attr+' />');
            cable.read_func = read_input;
			break;

		case 'cable_type':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.read_cable_type = read_input;

			} else if( dict.cable_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.cable_type+'" size="4" '+is_disabled_attr+' />');
				cable.read_cable_type = read_input;

			} else {

				html = '<select '+is_disabled_attr+' >';
				if( cable.cable_type == '' ) {
					for( var t in dict.cables()) {
						html += '<option';
						if( t == cable.cable_type ) html += ' selected="selected"';
						html += ' value="'+t+'">'+t+'</option>';
					}
				} else {

					if( dict.cable_is_not_known( cable.cable_type )) {

						html += '<option selected="selected" value="'+cable.cable_type+'">'+cable.cable_type+'</option>';
						for( var t in dict.cables())
							html += '<option value="'+t+'">'+t+'</option>';

					} else {

						for( var t in dict.cables()) {
							html += '<option';
							if( t == cable.cable_type ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
					}
				}
                if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new cable type" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
						that.cable_property_edit(pidx,cidx,'origin_conntype',true,false);
						that.cable_property_edit(pidx,cidx,'origin_pinlist',true,false);
						that.cable_property_edit(pidx,cidx,'destination_conntype',true,false);
						that.cable_property_edit(pidx,cidx,'destination_pinlist',true,false);
					} else {
						that.cable_property_edit(pidx,cidx,'origin_conntype',false,false);
						that.cable_property_edit(pidx,cidx,'origin_pinlist',false,false);
						that.cable_property_edit(pidx,cidx,'destination_conntype',false,false);
						that.cable_property_edit(pidx,cidx,'destination_pinlist',false,false);
					}
				});
				cable.read_cable_type = read_select;
			}
			break;

		case 'length':
            base.html('<input type="text" value="'+cable.length+ '" size="1" '+is_disabled_attr+' />');
            cable.read_length = read_input;
			break;

		case 'routing':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' />');
				cable.read_routing = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select '+is_disabled_attr+' >';

				if( dict.routing_is_not_known( cable.routing ))
					html += '<option  selected="selected" value="'+cable.routing+'">'+cable.routing+'</option>';
				for( var routing in dict.routings()) {
					html += '<option';
					if( routing == cable.routing ) html += ' selected="selected"';
					html += ' value="'+routing+'">'+routing+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new routing" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
					}
				});
				cable.read_routing = read_select;
			}
			break;

		case 'origin_name':
            base.html('<input type="text" value="'+cable.origin.name+ '" size="16" '+is_disabled_attr+' />');
            cable.origin.read_name = read_input;
			break;

		case 'origin_conntype':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.origin.read_conntype = read_input;

			} else if( dict.cable_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.origin.conntype+'" size="4" '+is_disabled_attr+' />');
				cable.origin.read_conntype = read_input;

			} else {

				if(	dict.cable_is_not_known( cable.read_cable_type())) {

					base.html('<input type="text" value="'+cable.origin.conntype+'" size="4" '+is_disabled_attr+' />');
					cable.origin.read_conntype = read_input;

				} else {

					if( dict.connector_dict_is_empty( cable.read_cable_type())) {

						base.html('<input type="text" value="'+cable.origin.conntype+'" size="4" '+is_disabled_attr+' />');
						cable.origin.read_conntype = read_input;

					} else {

						html = '<select '+is_disabled_attr+' >';
						for( var t in dict.connectors( cable.read_cable_type())) {
							html += '<option';
							if( t == cable.origin.conntype ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
						if(this.can_define_new_types()) html += '<option value=""  style="color:maroon;" title="register new connector type for the given cable type" >New...</option>'
						html += '</select>';
						base.html(html);
						base.find('select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property_edit(pidx,cidx,prop,true,false);
								that.cable_property_edit(pidx,cidx,'origin_pinlist',true,false);
							} else {
								that.cable_property_edit(pidx,cidx,'origin_pinlist',false,false);
							}
						});
						cable.origin.read_conntype = read_select;
					}
				}
			}
			break;

		case 'origin_pinlist':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.origin.read_pinlist = read_input;

			} else if( dict.cable_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.origin.pinlist+'" size="4" '+is_disabled_attr+' />');
				cable.origin.read_pinlist = read_input;

			} else if( dict.cable_is_not_known( cable.read_cable_type())) {

				base.html('<input type="text" value="'+cable.origin.pinlist+'" size="4" '+is_disabled_attr+' />');
				cable.origin.read_pinlist = read_input;

			} else if( dict.connector_is_not_known( cable.read_cable_type(), cable.origin.read_conntype())) {

				base.html('<input type="text" value="'+cable.origin.pinlist+'" size="4" '+is_disabled_attr+' />');
				cable.origin.read_pinlist = read_input;

			} else {

				if( dict.pinlist_dict_is_empty( cable.read_cable_type(), cable.origin.read_conntype())) {

					base.html('<input type="text" value="'+cable.origin.pinlist+'" size="4" '+is_disabled_attr+' />');
					cable.origin.read_pinlist = read_input;

				} else {

					html = '<select '+is_disabled_attr+' >';
					for( var pinlist in dict.pinlists( cable.read_cable_type(), cable.origin.read_conntype())) {
						html += '<option';
						if( pinlist == cable.origin.pinlist ) html += ' selected="selected"';
						html += ' value="'+pinlist+'">'+pinlist+'</option>';
					}
					if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new pinlist for the given connector type" >New...</option>'
					html += '</select>';
					base.html(html);
					base.find('select').change(function() {
						if( $(this).val() == '') {
							that.cable_property_edit(pidx,cidx,prop,true,false);
						}
					});
					cable.origin.read_pinlist = read_select;
				}
			}
			break;
	
		case 'origin_loc':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' />');
				cable.origin.read_loc = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select '+is_disabled_attr+' >';

				if( dict.location_is_not_known( cable.origin.loc ))
					html += '<option  selected="selected" value="'+cable.origin.loc+'">'+cable.origin.loc+'</option>';
				for( var loc in dict.locations()) {
					html += '<option';
					if( loc == cable.origin.loc ) html += ' selected="selected"';
					html += ' value="'+loc+'">'+loc+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new location" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop, true,false);
						that.cable_property_edit(pidx,cidx,'origin_rack',true,false);
					} else {
						that.cable_property_edit(pidx,cidx,'origin_rack',false,false);
					}
				});
				cable.origin.read_loc = read_select;
			}
			break;

		case 'origin_rack':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' />');
				cable.origin.read_rack = read_input;

			} else if( dict.location_is_not_known( cable.origin.read_loc())) {

				// This might came from a pre-existing  database. So we have to respect
				// a choice of a rack because the current location is also not known to
				// the dictionary.
				//
				base.html('<input type="text" value="'+cable.origin.rack+'" size="1" '+is_disabled_attr+' />');
				cable.origin.read_rack = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				// IMPORTANT NOTE: Pay attention that a act that we're making the rack_is_not_known(()
				//                 against the persistent state of the location, not against the transinet
				//                 state which is changes when selecting other locations. If we wern't using
				//                 the right source to get the location then the rack would be carried over
				//                 as a legitimate option between locations.
				//
				html = '<select '+is_disabled_attr+' >';

				if(( cable.origin.rack != '' ) && dict.rack_is_not_known( cable.origin.loc, cable.origin.rack ))
					html += '<option selected="selected" value="'+cable.origin.rack+'">'+cable.origin.rack+'</option>';
				for( var rack in dict.racks( cable.origin.read_loc())) {
					html += '<option';
					if( rack == cable.origin.rack ) html += ' selected="selected"';
					html += ' value="'+rack+'">'+rack+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new rack for the selected location" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
					}
				});
				cable.origin.read_rack = read_select;
			}
			break;

        case 'origin_ele':
			base.html('<input type="text" value="'+cable.origin.ele+'" size="1" '+is_disabled_attr+' />');
            cable.origin.read_ele = read_input;
			break;

        case 'origin_side':
			base.html('<input type="text" value="'+cable.origin.side+'" size="2" '+is_disabled_attr+' />');
            cable.origin.read_side = read_input;
			break;

        case 'origin_slot':
			base.html('<input type="text" value="'+cable.origin.slot+'" size="2" '+is_disabled_attr+' />');
            cable.origin.read_slot = read_input;
			break;

        case 'origin_conn':
			base.html('<input type="text" value="'+cable.origin.conn+'" size="2" '+is_disabled_attr+' />');
            cable.origin.read_conn = read_input;
			break;

        case 'origin_station':
			base.html('<input type="text" value="'+cable.origin.station+'" size="2" '+is_disabled_attr+' />');
            cable.origin.read_station = read_input;
			break;

		case 'origin_instr':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' />');
				cable.origin.read_instr = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select '+is_disabled_attr+'>';

				if( dict.instr_is_not_known( cable.origin.instr ))
					html += '<option  selected="selected" value="'+cable.origin.instr+'">'+cable.origin.instr+'</option>';
				for( var instr in dict.instrs()) {
					html += '<option';
					if( instr == cable.origin.instr ) html += ' selected="selected"';
					html += ' value="'+instr+'">'+instr+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new instr" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
					}
				});
				cable.origin.read_instr = read_select;
			}
			break;

		case 'destination_name':
            base.html('<input type="text" value="'+cable.destination.name+ '" size="16" '+is_disabled_attr+' />');
            cable.destination.read_name = read_input;
			break;

		case 'destination_conntype':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.destination.read_conntype = read_input;

			} else if( dict.cable_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.destination.conntype+'" size="4" '+is_disabled_attr+' />');
				cable.destination.read_conntype = read_input;

			} else {

				if(	dict.cable_is_not_known( cable.read_cable_type())) {

					base.html('<input type="text" value="'+cable.destination.conntype+'" size="4" '+is_disabled_attr+' />');
					cable.destination.read_conntype = read_input;

				} else {

					if( dict.connector_dict_is_empty( cable.read_cable_type())) {

						base.html('<input type="text" value="'+cable.destination.conntype+'" size="4" '+is_disabled_attr+' />');
						cable.destination.read_conntype = read_input;

					} else {
						html = '<select '+is_disabled_attr+'>';

						for( var t in dict.connectors( cable.read_cable_type())) {
							html += '<option';
							if( t == cable.destination.conntype ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
						if(this.can_define_new_types()) html += '<option value=""  style="color:maroon;" title="register new connector type for the given cable type" >New...</option>'
						html += '</select>';
						base.html(html);
						base.find('select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property_edit(pidx,cidx,prop,true,false);
								that.cable_property_edit(pidx,cidx,'destination_pinlist',true,false);
							} else {
								that.cable_property_edit(pidx,cidx,'destination_pinlist',false,false);
							}
						});
						cable.destination.read_conntype = read_select;
					}
				}
			}
			break;

		case 'destination_pinlist':

			if( is_new ) {

				base.html('<input type="text" value="" size="4" '+is_disabled_attr+' />');
				cable.destination.read_pinlist = read_input;

			} else if( dict.cable_dict_is_empty()) {

				base.html('<input type="text" value="'+cable.destination.pinlist+'" size="4" '+is_disabled_attr+' />');
				cable.destination.read_pinlist = read_input;

			} else if( dict.cable_is_not_known( cable.read_cable_type())) {

				base.html('<input type="text" value="'+cable.destination.pinlist+'" size="4" '+is_disabled_attr+' />');
				cable.destination.read_pinlist = read_input;

			} else if( dict.connector_is_not_known( cable.read_cable_type(), cable.destination.read_conntype())) {

				base.html('<input type="text" value="'+cable.destination.pinlist+'" size="4" '+is_disabled_attr+' />');
				cable.destination.read_pinlist = read_input;

			} else {

				if( dict.pinlist_dict_is_empty( cable.read_cable_type(), cable.destination.read_conntype())) {

					base.html('<input type="text" value="'+cable.destination.pinlist+'" size="4" '+is_disabled_attr+' />');
					cable.destination.read_pinlist = read_input;

				} else {

					html = '<select '+is_disabled_attr+'>';
					for( var pinlist in dict.pinlists( cable.read_cable_type(), cable.destination.read_conntype())) {
						html += '<option';
						if( pinlist == cable.destination.pinlist ) html += ' selected="selected"';
						html += ' value="'+pinlist+'">'+pinlist+'</option>';
					}
					if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new pinlist for the given connector type" >New...</option>'
					html += '</select>';
					base.html(html);
					base.find('select').change(function() {
						if( $(this).val() == '') {
							that.cable_property_edit(pidx,cidx,prop,true,false);
						}
					});
					cable.destination.read_pinlist = read_select;
				}
			}
			break;

		case 'destination_loc':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' /> ');
				cable.destination.read_loc = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select '+is_disabled_attr+'>';

				if( dict.location_is_not_known( cable.destination.loc ))
					html += '<option  selected="selected" value="'+cable.destination.loc+'">'+cable.destination.loc+'</option>';
				for( var loc in dict.locations()) {
					html += '<option';
					if( loc == cable.destination.loc ) html += ' selected="selected"';
					html += ' value="'+loc+'">'+loc+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new location" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
						that.cable_property_edit(pidx,cidx,'destination_rack',true,false);
					} else {
						that.cable_property_edit(pidx,cidx,'destination_rack',false,false);
					}
				});
				cable.destination.read_loc = read_select;
			}
			break;

		case 'destination_rack':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' />');
				cable.destination.read_rack = read_input;

			} else if( dict.location_is_not_known( cable.destination.read_loc())) {

				// This might came from a pre-existing  database. So we have to respect
				// a choice of a rack because the current location is also not known to
				// the dictionary.
				//
				base.html('<input type="text" value="'+cable.destination.rack+'" size="1" '+is_disabled_attr+' />');
				cable.destination.read_rack = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it would be automatically
				// added to the dictionary.
				//
				// IMPORTANT NOTE: Pay attention that a act that we're making the rack_is_not_known(()
				//                 against the persistent state of the location, not against the transinet
				//                 state which is changes when selecting other locations. If we wern't using
				//                 the right source to get the location then the rack would be carried over
				//                 as a legitimate option between locations.
				//
				html = '<select '+is_disabled_attr+'>';

				if(( cable.destination.rack != '' ) && dict.rack_is_not_known( cable.destination.loc, cable.destination.rack ))
					html += '<option selected="selected" value="'+cable.destination.rack+'">'+cable.destination.rack+'</option>';
				for( var rack in dict.racks( cable.destination.read_loc())) {
					html += '<option';
					if( rack == cable.destination.rack ) html += ' selected="selected"';
					html += ' value="'+rack+'">'+rack+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new rack for the selected location" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
					}
				});
				cable.destination.read_rack = read_select;
			}
			break;

        case 'destination_ele':
			base.html('<input type="text" value="'+cable.destination.ele+'" size="1" '+is_disabled_attr+' />');
            cable.destination.read_ele = read_input;
			break;

        case 'destination_side':
			base.html('<input type="text" value="'+cable.destination.side+'" size="2" '+is_disabled_attr+' />');
            cable.destination.read_side = read_input;
			break;

        case 'destination_slot':
			base.html('<input type="text" value="'+cable.destination.slot+'" size="2" '+is_disabled_attr+' />');
            cable.destination.read_slot = read_input;
			break;

        case 'destination_conn':
			base.html('<input type="text" value="'+cable.destination.conn+'" size="2" '+is_disabled_attr+' />');
            cable.destination.read_conn = read_input;
			break;

        case 'destination_station':
			base.html('<input type="text" value="'+cable.destination.station+'" size="2" '+is_disabled_attr+' />');
            cable.destination.read_station = read_input;
			break;

		case 'destination_instr':

			if( is_new ) {

				// Start with empty
				//
				base.html('<input type="text" value="" size="1" '+is_disabled_attr+' />');
				cable.destination.read_instr = read_input;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select '+is_disabled_attr+' >';

				if( dict.instr_is_not_known( cable.destination.instr ))
					html += '<option  selected="selected" value="'+cable.destination.instr+'">'+cable.destination.instr+'</option>';
				for( var instr in dict.instrs()) {
					html += '<option';
					if( instr == cable.destination.instr ) html += ' selected="selected"';
					html += ' value="'+instr+'">'+instr+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new instr" >New...</option>'
				html += '</select>';
				base.html(html);
				base.find('select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true,false);
					}
				});
				cable.destination.read_instr = read_select;
			}
			break;
		}
	};
    
    function toggle_cable_editor(pidx,cidx,on) {
        if(on) {
            $('#proj-con-'+pidx          ).removeClass('proj-vis').addClass('proj-hdn');
            $('#proj-con-'+pidx+'-shadow').removeClass('proj-hdn').addClass('proj-vis');
        } else {
            $('#proj-con-'+pidx          ).removeClass('proj-hdn').addClass('proj-vis');
            $('#proj-con-'+pidx+'-shadow').removeClass('proj-vis').addClass('proj-hdn');
        }
    }
    this.edit_cable = function(pidx,cidx) {
        var cable = this.project[pidx].cable[cidx];
        var required_field_html = '<span style="color:red; font-size:110%; font-weight:bold;"> * </span>';
        var html =
'<div style="margin-bottom:10px; padding-bottom:10px; border-bottom:1px dashed #c0c0c0;">'+
'<div style="float:left; margin-bottom: 20px; margin-right: 40px; color: #900; width: 640px;">'+
'  Please, note that choices for some cable parameters found on this page are loaded from a dictionary.'+
'  Regular users of this software are not allowed to modify the dictionary neither assign arbitrary values'+
'  to those parameters. This has been done to enforce the corresponding SLAC CAPTOR and PCDS naming convention for cables'+
'  Please, contact administrators of the Cable Management Software'+
'  to request dictionary extensions if you feel the dictionary is not complete, or if you need'+
'  any non-standard names.'+
'</div>'+
'<div style="float:left; padding-top: 20px;">'+
'  <button name="save">Save</button>'+
'  <button name="cancel">Cancel</button>'+
'</div>'+
'<div style="clear:both;"></div>'+
'<div style="float:left; margin-right:40px;">'+
'  <table><tbody>'+
'    <tr><td class="table_cell table_cell_left">Status</td>'+
'        <td class="table_cell table_cell_right">'+cable.status+'</td></tr>'+
'    <tr><td class="table_cell table_cell_left">Cable #</td>'+
'        <td class="table_cell table_cell_right">'+cable.cable+'</td></tr>'+
'    <tr><td class="table_cell table_cell_left" valign="top">Device'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">'+
'            <table><tbody>'+
'              <tr><td class="table_cell table_cell_left ">location'+required_field_html+'</td>'+
'                  <td class="table_cell table_cell_right">                     <div class="device_location"></div></td></tr>'+
'              <tr><td class="table_cell table_cell_left ">region'+required_field_html+'</td>'+
'                  <td class="table_cell table_cell_right">                     <div class="device_region"></div></td></tr>'+
'              <tr><td class="table_cell table_cell_left ">component'+required_field_html+'</td>'+
'                  <td class="table_cell table_cell_right">                     <div class="device_component"></div></td></tr>'+
'              <tr><td class="table_cell table_cell_left ">counter'+required_field_html+'</td>'+
'                  <td class="table_cell table_cell_right">                     <div class="device_counter"></div></td></tr>'+
'              <tr><td class="table_cell table_cell_left  table_cell_bottom">suffix</td>'+
'                  <td class="table_cell table_cell_right table_cell_bottom">   <div class="device_suffix"></div></td></tr>'+
'            </tbody></table></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Function'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="func"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Type'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="cable_type"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Length</td>'+
'        <td class="table_cell table_cell_right">                               <div class="length"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left  table_cell_bottom">Routing</td>'+
'        <td class="table_cell table_cell_right table_cell_bottom">             <div class="routing"></div></td></tr>'+
'  </tbody></table>'+
'</div>'+
'<div style="float:left; margin-right:40px;">'+
'  <table><tbody>'+
'    <tr><td class="table_cell table_cell_left">Origin'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_name"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Location'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_loc"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Rack</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_rack"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Elevation</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_ele"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Side</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_side"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Slot</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_slot"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Conn #</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_conn"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Station</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_station"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Connector Type'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_conntype"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Pinlist'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="origin_pinlist"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left  table_cell_bottom">Instruction</td>'+
'        <td class="table_cell table_cell_right table_cell_bottom">             <div class="origin_instr"></div></td></tr>'+
'  </tbody></table>'+
'</div>'+
'<div style="float:left; margin-right:40px;">'+
'  <table><tbody>'+
'    <tr><td class="table_cell table_cell_left">Destination'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_name"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Location'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_loc"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Rack</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_rack"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Elevation</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_ele"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Side</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_side"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Slot</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_slot"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Conn #</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_conn"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Station</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_station"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Connector Type'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_conntype"</div></td></tr>'+
'    <tr><td class="table_cell table_cell_left">Pinlist'+required_field_html+'</td>'+
'        <td class="table_cell table_cell_right">                               <div class="destination_pinlist"></div></td></tr>'+
'    <tr><td class="table_cell table_cell_left  table_cell_bottom">Instruction</td>'+
'        <td class="table_cell table_cell_right table_cell_bottom">             <div class="destination_instr"></div></td></tr>'+
'  </tbody></table>'+
'</div>'+
'<div style="clear:both;"></div>'+
'</div>'+
required_field_html+' required feild';

        var elem = $('#proj-con-'+pidx+'-shadow');
        elem.html(html);
        elem.find('button[name="save"]').button().click(function() {
            that.edit_cable_save(pidx,cidx);
        });
        elem.find('button[name="cancel"]').button().click(function() {
            that.edit_cable_cancel(pidx,cidx);
        });
		this.cable_property_edit(pidx,cidx,'device',              false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'device_location',     false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'device_region',       false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'device_component',    false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'device_counter',      false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'device_suffix',       false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'func',                false,false);
		this.cable_property_edit(pidx,cidx,'cable_type',          false,false);
		this.cable_property_edit(pidx,cidx,'length',              false,false);
		this.cable_property_edit(pidx,cidx,'routing',             false,false);

		this.cable_property_edit(pidx,cidx,'origin_name',         false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'origin_loc',          false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Registered'));
		this.cable_property_edit(pidx,cidx,'origin_rack',         false,false);
		this.cable_property_edit(pidx,cidx,'origin_ele',          false,false);
		this.cable_property_edit(pidx,cidx,'origin_side',         false,false);
		this.cable_property_edit(pidx,cidx,'origin_slot',         false,false);
		this.cable_property_edit(pidx,cidx,'origin_conn',         false,false);
		this.cable_property_edit(pidx,cidx,'origin_station',      false,false);
		this.cable_property_edit(pidx,cidx,'origin_conntype',     false,false);
		this.cable_property_edit(pidx,cidx,'origin_pinlist',      false,false);
		this.cable_property_edit(pidx,cidx,'origin_instr',        false,false);

		this.cable_property_edit(pidx,cidx,'destination_name',    false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Labeled'));
		this.cable_property_edit(pidx,cidx,'destination_loc',     false,global_cable_status2rank(cable.status) >= global_cable_status2rank('Registered'));
		this.cable_property_edit(pidx,cidx,'destination_rack',    false,false);
		this.cable_property_edit(pidx,cidx,'destination_ele',     false,false);
		this.cable_property_edit(pidx,cidx,'destination_side',    false,false);
		this.cable_property_edit(pidx,cidx,'destination_slot',    false,false);
		this.cable_property_edit(pidx,cidx,'destination_conn',    false,false);
		this.cable_property_edit(pidx,cidx,'destination_station', false,false);
		this.cable_property_edit(pidx,cidx,'destination_conntype',false,false);
		this.cable_property_edit(pidx,cidx,'destination_pinlist', false,false);
		this.cable_property_edit(pidx,cidx,'destination_instr',   false,false);

        toggle_cable_editor(pidx,cidx,true);
    };
	this.edit_cable_save = function(pidx,cidx) {

        var cable = this.project[pidx].cable[cidx];

        // Save modifications to the database backend first before making any
        // updates to the transient store or making changes to the UI.
		//
        var params = {
            cable_id             : cable.id,
            device               : global_truncate_device          (cable.read_device          ()),
            device_location      : global_truncate_device_location (cable.read_device_location ()),
            device_region        : global_truncate_device_region   (cable.read_device_region   ()),
            device_component     : global_truncate_device_component(cable.read_device_component()),
            device_counter       : global_truncate_device_counter  (cable.read_device_counter  ()),
            device_suffix        : global_truncate_device_suffix   (cable.read_device_suffix   ()),
            func                 : global_truncate_func   (cable.read_func()),
            cable_type           : global_truncate_cable  (cable.read_cable_type()),
            length               : global_truncate_length (cable.read_length()),
            routing              : global_truncate_routing(cable.read_routing()),

            origin_name          : cable.origin.read_name(),
            origin_loc           : global_truncate_location (cable.origin.read_loc()),
            origin_rack          : global_truncate_rack     (cable.origin.read_rack()),
            origin_ele           : global_truncate_ele      (cable.origin.read_ele()),
            origin_side          : global_truncate_side     (cable.origin.read_side()),
            origin_slot          : global_truncate_slot     (cable.origin.read_slot()),
            origin_conn          : global_truncate_conn     (cable.origin.read_conn()),
            origin_station       : global_truncate_station  (cable.origin.read_station()),
            origin_conntype      : global_truncate_connector(cable.origin.read_conntype()),
            origin_pinlist       : global_truncate_pinlist  (cable.origin.read_pinlist()),
            origin_instr         : global_truncate_instr    (cable.origin.read_instr()),

            destination_name     : cable.destination.read_name(),
            destination_loc      : global_truncate_location (cable.destination.read_loc()),
            destination_rack     : global_truncate_rack     (cable.destination.read_rack()),
            destination_ele      : global_truncate_ele      (cable.destination.read_ele()),
            destination_side     : global_truncate_side     (cable.destination.read_side()),
            destination_slot     : global_truncate_slot     (cable.destination.read_slot()),
            destination_conn     : global_truncate_conn     (cable.destination.read_conn()),
            destination_station  : global_truncate_station  (cable.destination.read_station()),
            destination_conntype : global_truncate_connector(cable.destination.read_conntype()),
            destination_pinlist  : global_truncate_pinlist  (cable.destination.read_pinlist()),
            destination_instr    : global_truncate_instr    (cable.destination.read_instr())
       };
        $.ajax({
            type: 'POST',
            url: '../neocaptar/cable_save.php',
            data: params,
            success: function(data) {
                if( data.status != 'success' ) {
                    report_error(data.message);
                    // TODO: Re-enable editing dialog buttons to allow
                    //       another attempt.
                    //
                    return;
                }
                var cable = that.project[pidx].cable[cidx] = data.cable;

                toggle_cable_editor(pidx,cidx,false);

                that.update_project_hdr  (pidx);
                that.view_cable(pidx,cidx);

                dict.update(cable);

                admin.update();
            },
            error: function() {
                report_error('The request can not go through due a failure to contact the server.');
                // TODO: Re-enable editing dialog buttons to allow
                //       another attempt.
                //
                return;
            },
            dataType: 'json'
        });		
	};
    this.edit_cable_cancel = function(pidx,cidx) {
        toggle_cable_editor(pidx,cidx,false);
        this.view_cable(pidx,cidx);
    };
	this.view_cable = function(pidx,cidx) {

        var cable = this.project[pidx].cable[cidx];
        cable.editing = false;

        $('#proj-cable-action-'+pidx+'-'+cidx+'-reverse').html(this.cable_action2html(pidx,cidx,true ));
        $('#proj-cable-action-'+pidx+'-'+cidx           ).html(this.cable_action2html(pidx,cidx,false));
        $('.proj-cable-tool').button();
		this.update_cable_tools(pidx,cidx);

        var base_1 = '#proj-cable-'+pidx+'-'+cidx+'-1 td div.';

        $(base_1+'status'    ).html('&nbsp;'+cable.status);
        $(base_1+'cable'     ).html('&nbsp;'+cable.cable);
        $(base_1+'device'    ).html('&nbsp;'+cable.device);
        $(base_1+'func'      ).html('&nbsp;'+cable.func);
        $(base_1+'cable_type').html('&nbsp;'+cable.cable_type);
        $(base_1+'length'    ).html('&nbsp;'+cable.length);
        $(base_1+'routing'   ).html('&nbsp;'+cable.routing);

        $(base_1+'origin_name'    ).html('&nbsp;'+cable.origin.name);
        $(base_1+'origin_loc'     ).html('&nbsp;'+cable.origin.loc);
        $(base_1+'origin_rack'    ).html('&nbsp;'+cable.origin.rack);
        $(base_1+'origin_ele'     ).html('&nbsp;'+cable.origin.ele);
        $(base_1+'origin_side'    ).html('&nbsp;'+cable.origin.side);
        $(base_1+'origin_slot'    ).html('&nbsp;'+cable.origin.slot);
        $(base_1+'origin_conn'    ).html('&nbsp;'+cable.origin.conn);
        $(base_1+'origin_station' ).html('&nbsp;'+cable.origin.station);
        $(base_1+'origin_conntype').html('&nbsp;'+cable.origin.conntype);
        $(base_1+'origin_pinlist' ).html('&nbsp;'+dict.pinlist2url(cable.cable_type,cable.origin.conntype,cable.origin.pinlist));
        $(base_1+'origin_instr'   ).html('&nbsp;'+cable.origin.instr);

        var base_2 = '#proj-cable-'+pidx+'-'+cidx+'-2 td div.';

        $(base_2+'destination_name'    ).html('&nbsp;'+cable.destination.name);
        $(base_2+'destination_loc'     ).html('&nbsp;'+cable.destination.loc);
        $(base_2+'destination_rack'    ).html('&nbsp;'+cable.destination.rack);
        $(base_2+'destination_ele'     ).html('&nbsp;'+cable.destination.ele);
        $(base_2+'destination_side'    ).html('&nbsp;'+cable.destination.side);
        $(base_2+'destination_slot'    ).html('&nbsp;'+cable.destination.slot);
        $(base_2+'destination_conn'    ).html('&nbsp;'+cable.destination.conn);
        $(base_2+'destination_station' ).html('&nbsp;'+cable.destination.station);
        $(base_2+'destination_conntype').html('&nbsp;'+cable.destination.conntype);
        $(base_2+'destination_pinlist' ).html('&nbsp;'+dict.pinlist2url(cable.cable_type,cable.destination.conntype,cable.destination.pinlist));
        $(base_2+'destination_instr'   ).html('&nbsp;'+cable.destination.instr);

        $(base_1+'modified'            ).html('&nbsp;'+cable.modified.time);
        $(base_1+'modified_uid'        ).html('&nbsp;'+cable.modified.uid);
	};
	this.show_cable_label = function(pidx,cidx) {
        if(cidx) {
    		var cable = this.project[pidx].cable[cidx];
        	window.open('../neocaptar/cable_label.php?cable_id='+cable.id,'cable label');
        } else {
    		var proj = this.project[pidx];
        	window.open('../neocaptar/cable_label.php?project_id='+proj.id,'cable labels for the project');
        }
	};
    this.change_cable_status = function(pidx,cidx,new_status) {
		var params = {cable_id: this.project[pidx].cable[cidx].id, status: new_status};
		var jqXHR = $.post(
			'../neocaptar/cable_save.php', params,
			function(data) {
				if( data.status != 'success' ) {
					report_error(data.message);
					return;
				}
				that.project[pidx].cable[cidx] = data.cable;

                that.view_cable        (pidx,cidx);
				that.update_project_hdr(pidx);

                admin.update();
			},
			'JSON'
		).error(
			function () {
				report_error('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
			}
		);
    };
	this.register_cable   = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Registered');   };
	this.label_cable      = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Labeled');      };
	this.fabricate_cable  = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Fabrication');  };
	this.ready_cable      = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Ready');        };
	this.install_cable    = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Installed');    };
	this.commission_cable = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Commissioned'); };
	this.damage_cable     = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Damaged');      };
	this.retire_cable     = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Retired');      };

    this.un_register_cable   = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Planned');      };
	this.un_label_cable      = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Registered');   };
	this.un_fabricate_cable  = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Labeled');      };
	this.un_ready_cable      = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Fabrication');  };
	this.un_install_cable    = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Ready');        };
	this.un_commission_cable = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Installed');    };
	this.un_damage_cable     = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Commissioned'); };
	this.un_retire_cable     = function(pidx,cidx) { this.change_cable_status(pidx,cidx,'Damaged');      };

    this.sort_cables = function(pidx) {
        var sort_by = $('#proj-displ-'+pidx).find('select[name="sort"]').val();
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
        this.project[pidx].cable.sort(sorter);
    };
    this.select_cables_by_status = function(pidx) {
		var proj = this.project[pidx];
		var status = $('#proj-cables-hdr-'+pidx+' td select').val();
		if( '- status -' == status ) {
			for( var cidx in proj.cable ) {
				$('#proj-cable-'+pidx+'-'+cidx+'-1').css('display','');
				$('#proj-cable-'+pidx+'-'+cidx+'-2').css('display','');
			}
		} else {
			for( var cidx in proj.cable ) {
				var style = proj.cable[cidx].status == status ? '' : 'none';
				$('#proj-cable-'+pidx+'-'+cidx+'-1').css('display',style);
				$('#proj-cable-'+pidx+'-'+cidx+'-2').css('display',style);
			}
		}
	};
	this.project2html = function(idx) {
		var p = this.project[idx];
		var html =
'<div class="proj-hdr" id="proj-hdr-'+idx+'" onclick="projects.toggle_project('+idx+');">'+
'  <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e" id="proj-tgl-'+idx+'"></span></div>'+
'  <div class="proj-created"                       >'+p.created            +'</div>'+
'  <div class="proj-owner"                         >'+p.owner              +'</div>'+
'  <div class="proj-title"                         >'+p.title              +'</div>'+
'  <div class="proj-job"                           >'+(p.job?p.job:'FIXME')+'</div>'+
'  <div class="proj-num-cables-total total"        >'+p.status.total       +'</div>'+
'  <div class="proj-num-cables       Planned"      >'+p.status.Planned     +'</div>'+
'  <div class="proj-num-cables       Registered"   >'+p.status.Registered  +'</div>'+
'  <div class="proj-num-cables       Labeled"      >'+p.status.Labeled     +'</div>'+
'  <div class="proj-num-cables       Fabrication"  >'+p.status.Fabrication +'</div>'+
'  <div class="proj-num-cables       Ready"        >'+p.status.Ready       +'</div>'+
'  <div class="proj-num-cables       Installed"    >'+p.status.Installed   +'</div>'+
'  <div class="proj-num-cables       Commissioned" >'+p.status.Commissioned+'</div>'+
'  <div class="proj-num-cables       Damaged"      >'+p.status.Damaged     +'</div>'+
'  <div class="proj-num-cables-last  Retired"      >'+p.status.Retired     +'</div>'+
'  <div class="proj-due"                           >'+p.due                +'</div>'+
'  <div class="proj-modified"                      >'+p.modified           +'</div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="proj-con proj-cable-editor proj-hdn" id="proj-con-'+idx+'-shadow"></div>'+
'<div class="proj-con proj-hdn" id="proj-con-'+idx+'">'+
'  <div style="padding-bottom:10px;">'+
'    <div style="float:left; width:620px; height:140px; overflow:auto; padding:5px; background-color:#f0f0f0; border-bottom:1px solid #c0c0c0; border-right:1px solid #c0c0c0;"><pre class="proj-description" style="font-family:Lucida Grande, Lucida Sans, Arial, sans-serif" title="Click on the Edit Project Attributes button to edit this description"></pre></div>'+
'    <div style="float:left; margin-left:30px; " >'+
'      <div>'+
'        <button id="proj-edit-'+idx+'" title="edit attributes of the project">edit project attributes</button>'+
'        <button id="proj-delete-'+idx+'" title="delete the project and all associated cables from the database">delete project</button>'+
'        <button id="proj-clone-'+idx+'" title="clone the whole project and all associated cables and create a new project with a temporary name (which can be changed later)">clone project</button>'+
'        <button id="proj-history-'+idx+'" title="show major events in the project history">show project history</button>'+
'      </div>'+
'      <div style="margin-top:5px; ">'+
'        <button id="proj-add-'+idx+'" title="add new cable to the project. Thsi will open cable editor dialog.">add cable</button>'+
'        <button id="proj-label-'+idx+'" title="show cable labels.">generate cable labels</button>'+
'      </div>'+
'      <div style="margin-top:15px; color:maroon;">'+
'        <b><u>THERMOMARK X1.2 SETTINGS FOR LABELS:</u></b><br>'+
'        <table style="margin-top:5px; font-size:120%;"><tbody>'+
'          <tr>'+
'            <td class="table_cell table_cell_left"           >orientation</td>'+
'            <td class="table_cell"                           >auto</td>'+
'            <td class="table_cell table_cell_left"           >size option</td>'+
'            <td class="table_cell table_cell_right"          >actual size</td>'+
'          </tr>'+
'          <tr>'+
'            <td class="table_cell table_cell_left"           >width</td>'+
'            <td class="table_cell"                           >3.54 in</td>'+
'            <td class="table_cell table_cell_left"           >height</td>'+
'            <td class="table_cell table_cell_right"          >1.0 in</td>'+
'          </tr>'+
'            <td class="table_cell table_cell_left  table_cell_bottom"          >all margins</td>'+
'            <td class="table_cell table_cell_right table_cell_bottom" colspan=3>0 in</td>'+
'          <tr>'+
'          </tr>'+
'        </tbody></table>'+
'      </div>'+
'    </div>'+
'    <div style="clear:both;"></div>'+
'    <div style="margin-top:15px;">'+
'      <div id="proj-displ-'+idx+'">'+
'        <div style=font-size:80%;">'+
'          <table style="font-size:120%;"><tbody>'+
'            <tr>'+
'              <td rowspan=2><button class="export" name="excel" title="Export into Microsoft Excel 2007 File"><img src="img/EXCEL_icon.gif" /></button></td>'+
'              <td><div style="width:20px;"></div></td>'+
'              <td><input type="checkbox" name="status"   checked="checked"></input>status</td>'+
'              <td><input type="checkbox" name="project"  checked="checked"></input>project</td>'+
'              <td><input type="checkbox" name="job"      checked="checked"></input>job #</td>'+
'              <td><input type="checkbox" name="cable"    checked="checked"></input>cable #</td>'+
'              <td><input type="checkbox" name="device"   checked="checked"></input>device</td>'+
'              <td><input type="checkbox" name="func"     checked="checked"></input>function</td>'+
'              <td rowspan=2><div style="width:20px;"></div></td>'+
'              <td rowspan=2><b>Sort by:</b></td>'+
'              <td rowspan=2><select name="sort" style="padding:1px;">'+
'                    <option>status</option>'+
'                    <option>job</option>'+
'                    <option>cable</option>'+
'                    <option>device</option>'+
'                    <option>function</option>'+
'                    <option>origin</option>'+
'                    <option>destination</option>'+
'                    <option>modified</option>'+
'                  </select></td>'+
'              <td rowspan=2><div style="width:20px;"></div></td>'+
'              <td rowspan=2><button name="reverse">Show in Reverse Order</button></td>'+
'            </tr>'+
'            <tr>'+
'              <td colspan=1></td>'+
'              <td          ><input type="checkbox" name="length"   checked="checked"></input>length</td>'+
'              <td          ><input type="checkbox" name="routing"  checked="checked"></input>routing</td>'+
'              <td colspan=3><input type="checkbox" name="sd"                        ></input>expanded ORIGIN/DESTINATION</td>'+
'              <td          ><input type="checkbox" name="modified" checked="checked"></input>modified</td>'+
'            </tr>'+
'          </tbody></table>'+
'        </div>'+
'      </div>'+
'    </div>'+
'  </div>'+
'  <div class="table"></div>'+
'  <div id="proj-cables-load-'+idx+'" style="margin-top:5px; color:maroon;"></div>'+
'</div>';
		return html;
	};
	this.display = function() {
		var total = 0;
		var html = '';
		for( var pidx in this.project ) {
			var proj = this.project[pidx];
			proj.is_loaded = false;
            proj.cols2display = {
                project:  false,
                tools:    true,
                cable:    true,
                device:   true,
                func:     true,
                length:   true,
                routing:  true,
                sd:       false,
                modified: true
            };
			html += this.project2html(pidx);
            total++;
		}
		var info_html = '<b>'+total+'</b> project'+(total==1?'':'s');
		$('#projects-search-info').html(info_html);
		$('#projects-search-list').html(html);

        // Initializations on the rendered HTML.
        //
		for( pidx in this.project ) {

            var proj = this.project[pidx];

            // Delayed initialization of the description box contents to prevent
            // the text to be interpreted/parsed by the browser as HTML.
            //
            $('#proj-con-'+pidx).find('.proj-description').
                text(proj.description?proj.description:'<no project description yet>');

            // Event handlers can't be registered inline due to some weird
            // problem with JavaScript. Failure to do so will always register
            // handlers for the last iteration of the loop.
            //
            this.register_display_handlers(pidx);
        }
	};
    this.register_display_handlers = function(pidx) {
        $('#proj-displ-'+pidx).find('input:checkbox').
            change( function() {
                that.project[pidx].cols2display[this.name] = this.checked;
                that.display_cables(pidx); });
        $('#proj-displ-'+pidx).find('select[name="sort"]').
            change( function() {
                that.sort_cables(pidx);
                that.display_cables(pidx); });
        $('#proj-displ-'+pidx).find('button[name="reverse"]').
            button().
            click(function() {
                that.project[pidx].cable.reverse(pidx);
                that.display_cables(pidx); });
        $('#proj-con-'+pidx).find('.export:button').
            button().
            click(function() {
                that.export_project(that.project[pidx].id,this.name); });
    };
    this.export_project = function(project_id,outformat) {
        var params = {project_id:project_id};
        global_export_cables(params,outformat);
    };
    this.search_project_by_id = function(id) {
		$('#projects-search-info').html('Searching...');
		var params = {id:id};
		var jqXHR = $.get(
			'../neocaptar/project_search.php', params,
			function(data) {
				if( data.status != 'success' ) {
					report_error( data.message );
					return;
				}
				that.project = data.project;
                that.display();
                for( var pidx in that.project )
                    that.toggle_project(pidx);

                // Simulate using the form to search for projects based on its
                // title and owner.
                //
                var search_controls = $('#projects-search-controls');
                search_controls.find('input[name="title"]').val(data.project[0].title);
                search_controls.find('select[name="owner"]').val(data.project[0].owner);
			},
			'JSON'
		).error(
			function () {
				report_error('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
			}
		);
	};
    this.search_projects_by_owner = function(uid) {
        this.init();
        this.search_reset();
        var search_controls = $('#projects-search-controls');
        search_controls.find('select[name="owner"]').val(uid);
		var params = {owner:uid};
		this.search_impl(params);
    }
	this.search_reset = function() {
        var search_controls = $('#projects-search-controls');
        search_controls.find('input[name="title"]').val('');
        search_controls.find('select[name="owner"]').val('');
        search_controls.find('input[name="begin"]').val('');
        search_controls.find('input[name="end"]').val('');
        this.project = [];
        this.display();
    }
	this.search = function() {
        var search_controls = $('#projects-search-controls');
        var title = search_controls.find('input[name="title"]').val();
        var owner = search_controls.find('select[name="owner"]').val();
        var begin = search_controls.find('input[name="begin"]').val();
        var end = search_controls.find('input[name="end"]').val();
		var params = {title:title,owner:owner,begin:begin,end:end};
		this.search_impl(params);
	};
	this.search_impl = function(params) {
        $('#projects-search-info').html('Searching...');
		var jqXHR = $.get(
			'../neocaptar/project_search.php', params,
			function(data) {
				if( data.status != 'success' ) {
					report_error( data.message );
					return;
				}
				that.project = data.project;
				that.display();
			},
			'JSON'
		).error(
			function () {
				report_error('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
			}
		);
	};
	return this;
}
var projects = new p_appl_projects();
