
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

    this.select = function(context,when_done) {
		that.context   = context;
		this.when_done = when_done;
		this.init();
	};
	this.select_default = function() {
		if( this.context == '' ) this.context = 'search';
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
    this.can_manage_project = function(pidx) {
        return global_current_user.is_administrator || (global_current_user.can_manage_projects && (global_current_user.uid == this.project[pidx].owner));
    }
    this.can_define_new_types = function() {
        return !global_current_user.is_administrator;
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
				$('#proj-add-'+pidx).
                    button().
                    button(this.can_manage_project(pidx)?'enable':'disable').
                    click(function() { that.add_cable(pidx); });
				$('#proj-delete-'+pidx ).
                    button().
                    button(this.can_manage_project(pidx)?'enable':'disable').
                    click(function() { that.delete_project(pidx); });
				$('#proj-edit-'+pidx ).
                    button().
                    button(this.can_manage_project(pidx)?'enable':'disable').
                    click(function() { that.edit_attributes(pidx); });
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
        for( var i in global_users ) {
            var user = global_users[i];
            html +=
'                  <option '+(proj.owner==user?'selected="selected"':'')+'>'+user+'</option>';
        }
        html +=
'                </select></td>'+
'            <td><b>Title: </b></td>'+
'            <td><input type="text" size="36" name="title" value="'+proj.title+'" style="padding:1px;"></td>'+
'            <td></td>'+
'            <td><b>Due by: </b></td>'+
'            <td><input type="text" size="6" name="due" value="'+proj.due+'" style="padding:1px;"></td>'+
'          </tr>'+
'          <tr>'+
'            <td><b>Descr: </b></td>'+
'            <td colspan="6"><textarea cols=60 rows=4 name="description" style="padding:4px;" title="Here be the project description">'+proj.description+'</textarea></td>'+
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
            url: '../portal/neocaptar_project_save.php',
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
	this.cable_action2html = function(pidx,cidx) {
		var c = this.project[pidx].cable[cidx];
		var html = '';
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
'    <button class="proj-cable-tool" name="edit_save"   onclick="projects.edit_cable_save  ('+pidx+','+cidx+')" title="save changes to the database"         >save</button>'+
'    <button class="proj-cable-tool" name="edit_cancel" onclick="projects.edit_cable_cancel('+pidx+','+cidx+')" title="cancel editing and ignore any changes">cancel</button>'+
'  </td>' : '';
        html += cols2display.project ?
'  <td nowrap="nowrap" class="table_cell table_cell_bottom ">&nbsp;<a class="link" href="javascript:global_search_project_by_id('+proj.id+');">'+proj.title+'</a></td>' : '';
        html += cols2display.job ?
'  <td nowrap="nowrap" class="table_cell table_cell_left table_cell_bottom "><div class="job"            >&nbsp;'+c.job            +'</div></td>' : '';
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
        html += cols2display.sd ?
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_name"    >&nbsp;'+c.origin.name    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_loc"     >&nbsp;'+c.origin.loc     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_rack"    >&nbsp;'+c.origin.rack    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_ele"     >&nbsp;'+c.origin.ele     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_side"    >&nbsp;'+c.origin.side    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_slot"    >&nbsp;'+c.origin.slot    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_conn"    >&nbsp;'+c.origin.conn    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_station" >&nbsp;'+c.origin.station +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_conntype">&nbsp;'+c.origin.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "                                  ><div class="origin_pinlist" >&nbsp;'+c.origin.pinlist +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right "                 ><div class="origin_instr"   >&nbsp;'+c.origin.instr   +'</div></td>' : '';
        html +=
'</tr>'+
'<tr class="table_row " id="proj-cable-'+pidx+'-'+cidx+'-2">';
        html += cols2display.tools ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left " align="right"><div id="proj-cable-action-'+pidx+'-'+cidx+'">'+this.cable_action2html(pidx,cidx)+'</div></td>' :
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
        html += cols2display.job ?
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
        html += cols2display.sd ?
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_name"    >&nbsp;'+c.destination.name    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_loc"     >&nbsp;'+c.destination.loc     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_rack"    >&nbsp;'+c.destination.rack    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_ele"     >&nbsp;'+c.destination.ele     +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_side"    >&nbsp;'+c.destination.side    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_slot"    >&nbsp;'+c.destination.slot    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_conn"    >&nbsp;'+c.destination.conn    +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_station" >&nbsp;'+c.destination.station +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_conntype">&nbsp;'+c.destination.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "                 ><div class="destination_pinlist" >&nbsp;'+c.destination.pinlist +'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight table_cell_right "><div class="destination_instr"   >&nbsp;'+c.destination.instr   +'</div></td>' : '';
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
'    <td nowrap="nowrap" class="table_hdr">TOOLS</td>' : '';
        html += cols2display.project ?
'    <td nowrap="nowrap" class="table_hdr">project</td>' : '';
        html += cols2display.job ?
'    <td nowrap="nowrap" class="table_hdr">job #</td>' : '';
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
        html += cols2display.sd ?
'    <td nowrap="nowrap" class="table_hdr">ORIGIN / DESTINATION</td>'+
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
			this.update_cable_tools(pidx,cidx,false);
	};
	this.load_cables = function(pidx) {
		$('#proj-cables-load-'+pidx).html('Loading...');
		var params = {project_id:this.project[pidx].id};
		var jqXHR = $.get(
			'../portal/neocaptar_cable_search.php', params,
			function(data) {
				if( data.status != 'success') {
					report_error('failed to load cables because of: '+data.message);
					return;
				}
				that.project[pidx].cable = data.cable;
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
	this.update_cable_tools = function(pidx,cidx,editing) {
		var cable = this.project[pidx].cable[cidx];
        if(!this.can_manage_project(pidx)) {
            $('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'       ).button('disable');
            $('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_save"]'  ).button('disable');
            $('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_cancel"]').button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="clone"]'      ).button('disable');
            $('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]'     ).button('disable');
            if( cable.status == 'Planned') {
                 $('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]' ).button('disable');
            } else {
                $('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]' ).button('enable');
            }
            $('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="history"]').button('enable');
            $('#proj-cable-action-'+pidx+'-'+cidx+' button').button('disable');
            return;
        }
		if( editing ) {
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'       ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_save"]'  ).button('enable' );
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_cancel"]').button('enable' );
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="clone"]'      ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]'     ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]'      ).button('disable');
			$('#proj-cable-action-'+pidx+'-'+cidx+' button').button('disable');
			return;
		}
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_save"]'  ).button('disable');
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_cancel"]').button('disable');
		if( cable.status == 'Planned') {
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'  ).button('enable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]').button('enable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]' ).button('disable');
		} else {
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'  ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]').button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]' ).button('enable');
		}
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="clone"]'  ).button('enable');
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="history"]').button('enable');
		$('#proj-cable-action-'+pidx+'-'+cidx+' button').button('enable');
	};
	this.update_project_tools = function(pidx) {
		var html = '';
		var proj = this.project[pidx];
		if(proj.num_edited) {
			html =
'<b>Warning:</b> total of '+proj.num_edited+' cables are being edited.<br>'+
'Some controls on this project page will be disabled before you save<br>'+
'or cancel the editing dialogs.';
			$('#proj-displ-'+pidx).children().attr('disabled','disabled');
			$('#proj-delete-'+pidx).button('disable');
			$('#proj-edit-'+pidx).button('disable');
		} else {
			$('#proj-displ-'+pidx).children().removeAttr('disabled');
			$('#proj-delete-'+pidx).button('enable');
			$('#proj-edit-'+pidx).button('enable');
		}
		$('#proj-alerts-'+pidx).html(html);
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
            url: '../portal/neocaptar_cable_new.php',
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
        var params = {owner:owner,title:title,description:description,due_time:due_time};
        var jqXHR = $.get('../portal/neocaptar_project_new.php',params,function(data) {
            $('#projects-create-info').html('&nbsp;');
            if(data.status != 'success') { report_error(data.message, null); return; }
            that.create_form_changed = false;
            if( that.when_done ) {
                that.when_done.execute();
                that.when_done = null;
            }
            var proj = data.project;
            global_search_project_by_id(proj.id);
            form.find('input[name="owner"]').val('');
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
                var jqXHR = $.get('../portal/neocaptar_project_delete.php',params,function(data) {
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
	this.submit_project = function(pidx) {
		var proj = this.project[pidx];
		proj.is_submitted = 1;
		var sec = mktime();
		proj.submitted_sec = sec*1000*1000*1000;
		var d = new Date(sec*1000);
		proj.modified = d.format('yyyy-mm-dd');
		$('#proj-hdr-'+pidx).find('.proj-modified').each(function() { $(this).html(proj.modified); });
		this.toggle_project(pidx);
		this.toggle_project(pidx);
	};
	this.show_cable_history = function(pidx,cidx) {
        var params = {id:this.project[pidx].cable[cidx].id};
        var jqXHR = $.get('../portal/neocaptar_cable_history.php',params,function(data) {
            if(data.status != 'success') { report_error(data.message, null); return; }
            var html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr">time</td>'+
'    <td nowrap="nowrap" class="table_hdr">user</td>'+
'    <td nowrap="nowrap" class="table_hdr">event</td>'+
'  </tr>';
            for( var i in data.event) {
                var event = data.event[i];
                html +=
'  <tr>'+
'    <td nowrap="nowrap" class="table_cell table_cell_left "  >'+event.event_time+'</td>'+
'    <td nowrap="nowrap" class="table_cell "                  >'+event.event_uid+'</td>'+
'    <td nowrap="nowrap" class="table_cell table_cell_right " >'+event.event+'</td>'+
'  </tr>';            }
            html +=
'</tbody></table>';
            report_info('Cable History',html);
        },
        'JSON').error(function () {
            report_error('failed to obtain the cable history because of: '+jqXHR.statusText, null);
            return;
        });
    };
	this.clone_cable = function(pidx,cidx) {
		var proj  = this.project[pidx];
		var cable = proj.cable[cidx];

        // First we need to submit the request to the database service in order
		// to get a unique cable identifier.
		//
        $.ajax({
            type: 'POST',
            url: '../portal/neocaptar_cable_new.php',
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
                var jqXHR = $.get('../portal/neocaptar_cable_delete.php',params,function(data) {
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
	this.cable_property_edit = function(pidx,cidx,prop,is_new) {

        var cable  = this.project[pidx].cable[cidx];

        cable.editing = true;

        var base_1 = '#proj-cable-'+pidx+'-'+cidx+'-1 td div.'+prop;
        var base_2 = '#proj-cable-'+pidx+'-'+cidx+'-2 td div.'+prop;

        var read_input_1 = function() { return $(base_1+' input').val(); };
        var read_input_2 = function() { return $(base_2+' input').val(); };

        var read_select_1 = function() { return $(base_1+' select').val(); };
        var read_select_2 = function() { return $(base_2+' select').val(); };

        var html  = '';

		switch(prop) {

		case 'status':
			$(base_1).html('<span style="color:maroon;">Editing...</span>');
            break;

		case 'device':
			$(base_1).html('<input type="text" value="'+cable.device+'" size="10" />');
            cable.read_device = read_input_1;
			break;

		case 'func':
			$(base_1).html('<input type="text" value="'+cable.func+'" size="24" />');
            cable.read_func = read_input_1;
			break;

		case 'cable_type':

			if( is_new ) {

				$(base_1).html('<input type="text" value="" size="4" />');
				cable.read_cable_type = read_input_1;

			} else if( dict.cable_dict_is_empty()) {

				$(base_1).html('<input type="text" value="'+cable.cable_type+'" size="4" />');
				cable.read_cable_type = read_input_1;

			} else {

				html = '<select>';
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
				$(base_1).html(html);
				$(base_1+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true);
						that.cable_property_edit(pidx,cidx,'origin_conntype',true);
						that.cable_property_edit(pidx,cidx,'origin_pinlist',true);
						that.cable_property_edit(pidx,cidx,'destination_conntype',true);
						that.cable_property_edit(pidx,cidx,'destination_pinlist',true);
					} else {
						that.cable_property_edit(pidx,cidx,'origin_conntype',false);
						that.cable_property_edit(pidx,cidx,'origin_pinlist',false);
						that.cable_property_edit(pidx,cidx,'destination_conntype',false);
						that.cable_property_edit(pidx,cidx,'destination_pinlist',false);
					}
				});
				cable.read_cable_type = read_select_1;
			}
			break;

		case 'length':
            $(base_1).html('<input type="text" value="'+cable.length+ '" size="1"  />');
            cable.read_length = read_input_1;
			break;

		case 'routing':

			if( is_new ) {

				// Start with empty
				//
				$(base_1).html('<input type="text" value="" size="1" />');
				cable.read_routing = read_input_1;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select>';

				if( dict.routing_is_not_known( cable.routing ))
					html += '<option  selected="selected" value="'+cable.routing+'">'+cable.routing+'</option>';
				for( var routing in dict.routings()) {
					html += '<option';
					if( routing == cable.routing ) html += ' selected="selected"';
					html += ' value="'+routing+'">'+routing+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new routing" >New...</option>'
				html += '</select>';
				$(base_1).html(html);
				$(base_1+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true);
					}
				});
				cable.read_routing = read_select_1;
			}
			break;

		case 'origin_name':
            $(base_1).html('<input type="text" value="'+cable.origin.name+ '" size="8" />');
            cable.origin.read_name = read_input_1;
			break;

		case 'origin_conntype':

			if( is_new ) {

				$(base_1).html('<input type="text" value="" size="4" />');
				cable.origin.read_conntype = read_input_1;

			} else if( dict.cable_dict_is_empty()) {

				$(base_1).html('<input type="text" value="'+cable.origin.conntype+'" size="4" />');
				cable.origin.read_conntype = read_input_1;

			} else {

				if(	dict.cable_is_not_known( cable.read_cable_type())) {

					$(base_1).html('<input type="text" value="'+cable.origin.conntype+'" size="4" />');
					cable.origin.read_conntype = read_input_1;

				} else {

					if( dict.connector_dict_is_empty( cable.read_cable_type())) {

						$(base_1).html('<input type="text" value="'+cable.origin.conntype+'" size="4" />');
						cable.origin.read_conntype = read_input_1;

					} else {

						html = '<select>';
						for( var t in dict.connectors( cable.read_cable_type())) {
							html += '<option';
							if( t == cable.origin.conntype ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
						if(this.can_define_new_types()) html += '<option value=""  style="color:maroon;" title="register new connector type for the given cable type" >New...</option>'
						html += '</select>';
						$(base_1).html(html);
						$(base_1+' select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property_edit(pidx,cidx,prop,true);
								that.cable_property_edit(pidx,cidx,'origin_pinlist',true);
							} else {
								that.cable_property_edit(pidx,cidx,'origin_pinlist', false);
							}
						});
						cable.origin.read_conntype = read_select_1;
					}
				}
			}
			break;

		case 'origin_pinlist':

			if( is_new ) {

				$(base_1).html('<input type="text" value="" size="4" />');
				cable.origin.read_pinlist = read_input_1;

			} else if( dict.cable_dict_is_empty()) {

				$(base_1).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
				cable.origin.read_pinlist = read_input_1;

			} else if( dict.cable_is_not_known( cable.read_cable_type())) {

				$(base_1).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
				cable.origin.read_pinlist = read_input_1;

			} else if( dict.connector_is_not_known( cable.read_cable_type(), cable.origin.read_conntype())) {

				$(base_1).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
				cable.origin.read_pinlist = read_input_1;

			} else {

				if( dict.pinlist_dict_is_empty( cable.read_cable_type(), cable.origin.read_conntype())) {

					$(base_1).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
					cable.origin.read_pinlist = read_input_1;

				} else {

					html = '<select>';
					for( var pinlist in dict.pinlists( cable.read_cable_type(), cable.origin.read_conntype())) {
						html += '<option';
						if( pinlist == cable.origin.pinlist ) html += ' selected="selected"';
						html += ' value="'+pinlist+'">'+pinlist+'</option>';
					}
					if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new pinlist for the given connector type" >New...</option>'
					html += '</select>';
					$(base_1).html(html);
					$(base_1+' select').change(function() {
						if( $(this).val() == '') {
							that.cable_property_edit(pidx,cidx,prop,true);
						}
					});
					cable.origin.read_pinlist = read_select_1;
				}
			}
			break;
	
		case 'origin_loc':

			if( is_new ) {

				// Start with empty
				//
				$(base_1).html('<input type="text" value="" size="1" />');
				cable.origin.read_loc = read_input_1;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select>';

				if( dict.location_is_not_known( cable.origin.loc ))
					html += '<option  selected="selected" value="'+cable.origin.loc+'">'+cable.origin.loc+'</option>';
				for( var loc in dict.locations()) {
					html += '<option';
					if( loc == cable.origin.loc ) html += ' selected="selected"';
					html += ' value="'+loc+'">'+loc+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new location" >New...</option>'
				html += '</select>';
				$(base_1).html(html);
				$(base_1+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop, true);
						that.cable_property_edit(pidx,cidx,'origin_rack',true);
					} else {
						that.cable_property_edit(pidx,cidx,'origin_rack',false);
					}
				});
				cable.origin.read_loc = read_select_1;
			}
			break;

		case 'origin_rack':

			if( is_new ) {

				// Start with empty
				//
				$(base_1).html('<input type="text" value="" size="1" />');
				cable.origin.read_rack = read_input_1;

			} else if( dict.location_is_not_known( cable.origin.read_loc())) {

				// This might came from a pre-existing  database. So we have to respect
				// a choice of a rack because the current location is also not known to
				// the dictionary.
				//
				$(base_1).html('<input type="text" value="'+cable.origin.rack+'" size="1" />');
				cable.origin.read_rack = read_input_1;

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
				html = '<select>';

				if(( cable.origin.rack != '' ) && dict.rack_is_not_known( cable.origin.loc, cable.origin.rack ))
					html += '<option selected="selected" value="'+cable.origin.rack+'">'+cable.origin.rack+'</option>';
				for( var rack in dict.racks( cable.origin.read_loc())) {
					html += '<option';
					if( rack == cable.origin.rack ) html += ' selected="selected"';
					html += ' value="'+rack+'">'+rack+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new rack for the selected location" >New...</option>'
				html += '</select>';
				$(base_1).html(html);
				$(base_1+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true);
					}
				});
				cable.origin.read_rack = read_select_1;
			}
			break;

        case 'origin_ele':
			$(base_1).html('<input type="text" value="'+cable.origin.ele+'" size="1" />');
            cable.origin.read_ele = read_input_1;
			break;

        case 'origin_side':
			$(base_1).html('<input type="text" value="'+cable.origin.side+'" size="2" />');
            cable.origin.read_side = read_input_1;
			break;

        case 'origin_slot':
			$(base_1).html('<input type="text" value="'+cable.origin.slot+'" size="2" />');
            cable.origin.read_slot = read_input_1;
			break;

        case 'origin_conn':
			$(base_1).html('<input type="text" value="'+cable.origin.conn+'" size="2" />');
            cable.origin.read_conn = read_input_1;
			break;

        case 'origin_station':
			$(base_1).html('<input type="text" value="'+cable.origin.station+'" size="2" />');
            cable.origin.read_station = read_input_1;
			break;

		case 'origin_instr':

			if( is_new ) {

				// Start with empty
				//
				$(base_1).html('<input type="text" value="" size="1" />');
				cable.origin.read_instr = read_input_1;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select>';

				if( dict.instr_is_not_known( cable.origin.instr ))
					html += '<option  selected="selected" value="'+cable.origin.instr+'">'+cable.origin.instr+'</option>';
				for( var instr in dict.instrs()) {
					html += '<option';
					if( instr == cable.origin.instr ) html += ' selected="selected"';
					html += ' value="'+instr+'">'+instr+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new instr" >New...</option>'
				html += '</select>';
				$(base_1).html(html);
				$(base_1+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true);
					}
				});
				cable.origin.read_instr = read_select_1;
			}
			break;

		case 'destination_name':
            $(base_2).html('<input type="text" value="'+cable.destination.name+ '" size="8" />');
            cable.destination.read_name = read_input_2;
			break;

		case 'destination_conntype':

			if( is_new ) {

				$(base_2).html('<input type="text" value="" size="4" />');
				cable.destination.read_conntype = read_input_2;

			} else if( dict.cable_dict_is_empty()) {

				$(base_2).html('<input type="text" value="'+cable.destination.conntype+'" size="4" />');
				cable.destination.read_conntype = read_input_2;

			} else {

				if(	dict.cable_is_not_known( cable.read_cable_type())) {

					$(base_2).html('<input type="text" value="'+cable.destination.conntype+'" size="4" />');
					cable.destination.read_conntype = read_input_2;

				} else {

					if( dict.connector_dict_is_empty( cable.read_cable_type())) {

						$(base_2).html('<input type="text" value="'+cable.destination.conntype+'" size="4" />');
						cable.destination.read_conntype = read_input_2;

					} else {
						html = '<select>';

						for( var t in dict.connectors( cable.read_cable_type())) {
							html += '<option';
							if( t == cable.destination.conntype ) html += ' selected="selected"';
							html += ' value="'+t+'">'+t+'</option>';
						}
						if(this.can_define_new_types()) html += '<option value=""  style="color:maroon;" title="register new connector type for the given cable type" >New...</option>'
						html += '</select>';
						$(base_2).html(html);
						$(base_2+' select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property_edit(pidx,cidx,prop,true);
								that.cable_property_edit(pidx,cidx,'destination_pinlist',true);
							} else {
								that.cable_property_edit(pidx,cidx,'destination_pinlist', false);
							}
						});
						cable.destination.read_conntype = read_select_2;
					}
				}
			}
			break;

		case 'destination_pinlist':

			if( is_new ) {

				$(base_2).html('<input type="text" value="" size="4" />');
				cable.destination.read_pinlist = read_input_2;

			} else if( dict.cable_dict_is_empty()) {

				$(base_2).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
				cable.destination.read_pinlist = read_input_2;

			} else if( dict.cable_is_not_known( cable.read_cable_type())) {

				$(base_2).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
				cable.destination.read_pinlist = read_input_2;

			} else if( dict.connector_is_not_known( cable.read_cable_type(), cable.destination.read_conntype())) {

				$(base_2).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
				cable.destination.read_pinlist = read_input_2;

			} else {

				if( dict.pinlist_dict_is_empty( cable.read_cable_type(), cable.destination.read_conntype())) {

					$(base_2).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
					cable.destination.read_pinlist = read_input_2;

				} else {

					html = '<select>';
					for( var pinlist in dict.pinlists( cable.read_cable_type(), cable.destination.read_conntype())) {
						html += '<option';
						if( pinlist == cable.destination.pinlist ) html += ' selected="selected"';
						html += ' value="'+pinlist+'">'+pinlist+'</option>';
					}
					if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new pinlist for the given connector type" >New...</option>'
					html += '</select>';
					$(base_2).html(html);
					$(base_2+' select').change(function() {
						if( $(this).val() == '') {
							that.cable_property_edit(pidx,cidx,prop,true);
						}
					});
					cable.destination.read_pinlist = read_select_2;
				}
			}
			break;

		case 'destination_loc':

			if( is_new ) {

				// Start with empty
				//
				$(base_2).html('<input type="text" value="" size="1" />');
				cable.destination.read_loc = read_input_2;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select>';

				if( dict.location_is_not_known( cable.destination.loc ))
					html += '<option  selected="selected" value="'+cable.destination.loc+'">'+cable.destination.loc+'</option>';
				for( var loc in dict.locations()) {
					html += '<option';
					if( loc == cable.destination.loc ) html += ' selected="selected"';
					html += ' value="'+loc+'">'+loc+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new location" >New...</option>'
				html += '</select>';
				$(base_2).html(html);
				$(base_2+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop, true);
						that.cable_property_edit(pidx,cidx,'destination_rack',true);
					} else {
						that.cable_property_edit(pidx,cidx,'destination_rack',false);
					}
				});
				cable.destination.read_loc = read_select_2;
			}
			break;

		case 'destination_rack':

			if( is_new ) {

				// Start with empty
				//
				$(base_2).html('<input type="text" value="" size="1" />');
				cable.destination.read_rack = read_input_2;

			} else if( dict.location_is_not_known( cable.destination.read_loc())) {

				// This might came from a pre-existing  database. So we have to respect
				// a choice of a rack because the current location is also not known to
				// the dictionary.
				//
				$(base_2).html('<input type="text" value="'+cable.destination.rack+'" size="1" />');
				cable.destination.read_rack = read_input_2;

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
				html = '<select>';

				if(( cable.destination.rack != '' ) && dict.rack_is_not_known( cable.destination.loc, cable.destination.rack ))
					html += '<option selected="selected" value="'+cable.destination.rack+'">'+cable.destination.rack+'</option>';
				for( var rack in dict.racks( cable.destination.read_loc())) {
					html += '<option';
					if( rack == cable.destination.rack ) html += ' selected="selected"';
					html += ' value="'+rack+'">'+rack+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new rack for the selected location" >New...</option>'
				html += '</select>';
				$(base_2).html(html);
				$(base_2+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true);
					}
				});
				cable.destination.read_rack = read_select_2;
			}
			break;

        case 'destination_ele':
			$(base_2).html('<input type="text" value="'+cable.destination.ele+'" size="1" />');
            cable.destination.read_ele = read_input_2;
			break;

        case 'destination_side':
			$(base_2).html('<input type="text" value="'+cable.destination.side+'" size="2" />');
            cable.destination.read_side = read_input_2;
			break;

        case 'destination_slot':
			$(base_2).html('<input type="text" value="'+cable.destination.slot+'" size="2" />');
            cable.destination.read_slot = read_input_2;
			break;

        case 'destination_conn':
			$(base_2).html('<input type="text" value="'+cable.destination.conn+'" size="2" />');
            cable.destination.read_conn = read_input_2;
			break;

        case 'destination_station':
			$(base_2).html('<input type="text" value="'+cable.destination.station+'" size="2" />');
            cable.destination.read_station = read_input_2;
			break;

		case 'destination_instr':

			if( is_new ) {

				// Start with empty
				//
				$(base_2).html('<input type="text" value="" size="1" />');
				cable.destination.read_instr = read_input_2;

			} else {

				// Finally, we suggest options found in the dictionary. If the input entry already
				// has someting non-empty(!) which isn't known to the dictionary then we should
				// add this as as the first option. If it's selected then it woudl be automatically
				// added to the dictionary.
				//
				html = '<select>';

				if( dict.instr_is_not_known( cable.destination.instr ))
					html += '<option  selected="selected" value="'+cable.destination.instr+'">'+cable.destination.instr+'</option>';
				for( var instr in dict.instrs()) {
					html += '<option';
					if( instr == cable.destination.instr ) html += ' selected="selected"';
					html += ' value="'+instr+'">'+instr+'</option>';
				}
				if(this.can_define_new_types()) html += '<option value="" style="color:maroon;" title="register new instr" >New...</option>'
				html += '</select>';
				$(base_2).html(html);
				$(base_2+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property_edit(pidx,cidx,prop,true);
					}
				});
				cable.destination.read_instr = read_select_2;
			}
			break;
		}
	};
    this.cancel_editing = function(e) {
        if( e.keyCode == 27 ) {
            ask_yes_no(
				'Unsaved Data Warning',
				'You are about to leave the page while there are unsaved data in the form. Are you sure?',
				function() {
                    for( var pidx in that.project ) {
                        var proj = that.project[pidx];
                        for( var cidx in proj.cable ) {
                            var cable = proj.cable[cidx];
                            if(cable.editing) {
                                that.edit_cable_cancel(pidx,cidx);
                            }
                        }
                    }
                },
				null
            );
        }
    };
	this.edit_cable = function(pidx,cidx) {
        
		this.project[pidx].num_edited++;
        if( this.project[pidx].num_edited == 1 ) {
            $('#proj-con-'+pidx+' .table').addClass('top');
            $('body').bind('keyup',projects.cancel_editing);
        }
        var prev = $('#proj-cable-'+pidx+'-'+cidx+'-1').prev();
        if( !prev.hasClass('table_first_header')) {
            var html =
'      <tr class="table_header">'+
'        <td nowrap="nowrap" class="table_hdr table_hdr_tight" >&nbsp;</td>'+
'        <td nowrap="nowrap" class="table_hdr">TOOLS</td>'+
'        <td nowrap="nowrap" class="table_hdr">job #</td>'+
'        <td nowrap="nowrap" class="table_hdr">cable #</td>'+
'        <td nowrap="nowrap" class="table_hdr">device</td>'+
'        <td nowrap="nowrap" class="table_hdr">function</td>'+
'        <td nowrap="nowrap" class="table_hdr">cable type</td>'+
'        <td nowrap="nowrap" class="table_hdr">length</td>'+
'        <td nowrap="nowrap" class="table_hdr">routing</td>'+
'        <td nowrap="nowrap" class="table_hdr">ORIGIN / DESTINATION</td>'+
'        <td nowrap="nowrap" class="table_hdr">loc</td>'+
'        <td nowrap="nowrap" class="table_hdr">rack</td>'+
'        <td nowrap="nowrap" class="table_hdr">ele</td>'+
'        <td nowrap="nowrap" class="table_hdr">side</td>'+
'        <td nowrap="nowrap" class="table_hdr">slot</td>'+
'        <td nowrap="nowrap" class="table_hdr">conn #</td>'+
'        <td nowrap="nowrap" class="table_hdr">station</td>'+
'        <td nowrap="nowrap" class="table_hdr">contype</td>'+
'        <td nowrap="nowrap" class="table_hdr">pinlist</td>'+
'        <td nowrap="nowrap" class="table_hdr">instr</td>'+
'      </tr>';
            $('#proj-cable-'+pidx+'-'+cidx+'-1').before(html);
        }
		this.update_cable_tools(pidx,cidx,true);
		this.update_project_tools(pidx);

		this.cable_property_edit(pidx,cidx,'status',false);
		this.cable_property_edit(pidx,cidx,'device',false);
		this.cable_property_edit(pidx,cidx,'func',false);
		this.cable_property_edit(pidx,cidx,'cable_type',false);
		this.cable_property_edit(pidx,cidx,'length',false);
		this.cable_property_edit(pidx,cidx,'routing',false);

		this.cable_property_edit(pidx,cidx,'origin_name',false);
		this.cable_property_edit(pidx,cidx,'origin_loc',false);
		this.cable_property_edit(pidx,cidx,'origin_rack',false);
		this.cable_property_edit(pidx,cidx,'origin_ele',false);
		this.cable_property_edit(pidx,cidx,'origin_side',false);
		this.cable_property_edit(pidx,cidx,'origin_slot',false);
		this.cable_property_edit(pidx,cidx,'origin_conn',false);
		this.cable_property_edit(pidx,cidx,'origin_station',false);
		this.cable_property_edit(pidx,cidx,'origin_conntype',false);
		this.cable_property_edit(pidx,cidx,'origin_pinlist',false);
		this.cable_property_edit(pidx,cidx,'origin_instr',false);

		this.cable_property_edit(pidx,cidx,'destination_name',false);
		this.cable_property_edit(pidx,cidx,'destination_loc',false);
		this.cable_property_edit(pidx,cidx,'destination_rack',false);
		this.cable_property_edit(pidx,cidx,'destination_ele',false);
		this.cable_property_edit(pidx,cidx,'destination_side',false);
		this.cable_property_edit(pidx,cidx,'destination_slot',false);
		this.cable_property_edit(pidx,cidx,'destination_conn',false);
		this.cable_property_edit(pidx,cidx,'destination_station',false);
		this.cable_property_edit(pidx,cidx,'destination_conntype',false);
		this.cable_property_edit(pidx,cidx,'destination_pinlist',false);
		this.cable_property_edit(pidx,cidx,'destination_instr',false);
	};
	this.edit_cable_save = function(pidx,cidx) {

        var cable = this.project[pidx].cable[cidx];

        // Save modifications to the database backend first before making any
        // updates to the transient store or making changes to the UI.
		//
        var params = {
            cable_id             : cable.id,
            device               : cable.read_device(),
            func                 : cable.read_func(),
            cable_type           : cable.read_cable_type(),
            length               : cable.read_length(),
            routing              : cable.read_routing(),

            origin_name          : cable.origin.read_name(),
            origin_loc           : cable.origin.read_loc(),
            origin_rack          : cable.origin.read_rack(),
            origin_ele           : cable.origin.read_ele(),
            origin_side          : cable.origin.read_side(),
            origin_slot          : cable.origin.read_slot(),
            origin_conn          : cable.origin.read_conn(),
            origin_station       : cable.origin.read_station(),
            origin_conntype      : cable.origin.read_conntype(),
            origin_pinlist       : cable.origin.read_pinlist(),
            origin_instr         : cable.origin.read_instr(),

            destination_name     : cable.destination.read_name(),
            destination_loc      : cable.destination.read_loc(),
            destination_rack     : cable.destination.read_rack(),
            destination_ele      : cable.destination.read_ele(),
            destination_side     : cable.destination.read_side(),
            destination_slot     : cable.destination.read_slot(),
            destination_conn     : cable.destination.read_conn(),
            destination_station  : cable.destination.read_station(),
            destination_conntype : cable.destination.read_conntype(),
            destination_pinlist  : cable.destination.read_pinlist(),
            destination_instr    : cable.destination.read_instr()
       };
        $.ajax({
            type: 'POST',
            url: '../portal/neocaptar_cable_save.php',
            data: params,
            success: function(data) {
                if( data.status != 'success' ) {
                    report_error(data.message);
                    this.update_cable_tools(pidx,cidx,true);    // Re-enable editing dialog buttons to allow
                                                                // another attempt.
                    return;
                }
                var cable = that.project[pidx].cable[cidx] = data.cable;

                that.project[pidx].num_edited--;
                if( that.project[pidx].num_edited == 0 ) {
                    $('#proj-con-'+pidx+' .table').removeClass('top');
                    $('body').unbind('keyup',this.cancel_editing);
                }
                var prev = $('#proj-cable-'+pidx+'-'+cidx+'-1').prev();
                if( prev.hasClass('table_header')) {
                    prev.remove();
                }

                that.update_project_tools(pidx);
                that.update_project_hdr  (pidx);

                that.view_cable(pidx,cidx);

                dict.update(cable);

                admin.update();
            },
            error: function() {
                report_error('The request can not go through due a failure to contact the server.');
                this.update_cable_tools(pidx,cidx,true);    // Re-enable editing dialog buttons to allow
                                                            // another attempt.
                return;
            },
            dataType: 'json'
        });

        // We should make this call in order to disable 'Save', 'Cancel',
        // or other buttons while the above initiated transcation is
        // in progress. If it fail then the error handler will re-enable
        // the buttons to allow corrections or cancelling editiing session.
        //
		this.update_cable_tools(pidx,cidx,false);		
	};
	this.edit_cable_cancel = function(pidx,cidx) {

		this.project[pidx].num_edited--;
        if( this.project[pidx].num_edited == 0 ) {
            $('#proj-con-'+pidx+' .table').removeClass('top');
            $('body').unbind('keyup',this.cancel_editing);
        }
        var prev = $('#proj-cable-'+pidx+'-'+cidx+'-1').prev();
        if( prev.hasClass('table_header')) {
            prev.remove();
        }

		this.update_cable_tools(pidx,cidx,false);
		this.update_project_tools(pidx);

		this.view_cable(pidx,cidx);
	};
	this.view_cable = function(pidx,cidx) {

        var cable = this.project[pidx].cable[cidx];
        cable.editing = false;

        $('#proj-cable-action-'+pidx+'-'+cidx).html(this.cable_action2html(pidx,cidx));
        $('.proj-cable-tool').button();
		this.update_cable_tools(pidx,cidx,false);

        var base_1 = '#proj-cable-'+pidx+'-'+cidx+'-1 td div.';

        $(base_1+'status'    ).html('&nbsp;'+cable.status);
        $(base_1+'job'       ).html('&nbsp;'+cable.job);
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
        $(base_1+'origin_pinlist' ).html('&nbsp;'+cable.origin.pinlist);
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
        $(base_2+'destination_pinlist' ).html('&nbsp;'+cable.destination.pinlist);
        $(base_2+'destination_instr'   ).html('&nbsp;'+cable.destination.instr);
	};
	this.show_cable_label = function(pidx,cidx) {
		var cable = this.project[pidx].cable[cidx];
		report_error('show_cable_label: '+pidx+'.'+cidx);
	};
    this.change_cable_status = function(pidx,cidx,new_status) {
		var params = {cable_id: this.project[pidx].cable[cidx].id, status: new_status};
		var jqXHR = $.post(
			'../portal/neocaptar_cable_save.php', params,
			function(data) {
				if( data.status != 'success' ) {
					report_error(data.message);
					return;
				}
				that.project[pidx].cable[cidx] = data.cable;

                that.view_cable        (pidx,cidx);
                that.update_cable_tools(pidx,cidx,false);
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
'<div class="proj-con proj-hdn" id="proj-con-'+idx+'">'+
'  <div style="padding-bottom:10px;">'+
'    <div style="float:left; width:695px; padding:5px; background-color:#f0f0f0; border-bottom:1px solid #c0c0c0; border-right:1px solid #c0c0c0;"><pre class="proj-description" style="font-family:Lucida Grande, Lucida Sans, Arial, sans-serif" title="Click on the Edit Project Attributes button to edit this description"></pre></div>'+
'    <div style="float:left; margin-left:10px; ">'+
'      <button class="export" name="excel" title="Export into Microsoft Excel 2007 File"><img src="img/EXCEL_icon.gif" /></button>'+
'    </div>'+
'    <div style="float:left; margin-left:10px; " >'+
'      <div>'+
'        <button id="proj-add-'+idx+'" title="add new cable to the project. You will be temporarily redirected to the bew cable registration form.">add cable</button>'+
'        <button id="proj-delete-'+idx+'" title="delete the project and all associated cables from the database">delete project</button>'+
'      </div>'+
'      <div style="margin-top:5px; ">'+
'        <button id="proj-edit-'+idx+'" title="edit attributes of the project">edit project attributes</button>'+
'      </div>'+
'    </div>'+
'    <div style="clear:both;"></div>'+
'    <div style="margin-top:15px;">'+
'      <div style="float:left;" id="proj-displ-'+idx+'">'+
'        <input type="checkbox" name="tools"   checked="checked"></input>tools'+
'        <input type="checkbox" name="project"                  ></input>project'+
'        <input type="checkbox" name="job"     checked="checked"></input>job #'+
'        <input type="checkbox" name="cable"   checked="checked"></input>cable #'+
'        <input type="checkbox" name="device"  checked="checked"></input>device'+
'        <input type="checkbox" name="func"    checked="checked"></input>function'+
'        <input type="checkbox" name="length"  checked="checked"></input>length'+
'        <input type="checkbox" name="routing" checked="checked"></input>routing'+
'        <input type="checkbox" name="sd"      checked="checked"></input>source & destination'+
'      </div>'+
'      <div style="float:left; margin-left:20px; padding-left:10px;" class="proj-alerts" id="proj-alerts-'+idx+'"></div>'+
'      <div style="clear:both;"></div>'+
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
			proj.num_edited = 0;
            proj.cols2display = {
                project: false,
                tools:   true,
                job:     true,
                cable:   true,
                device:  true,
                func:    true,
                length:  true,
                routing: true,
                sd:      true
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
			'../portal/neocaptar_project_search.php', params,
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
			'../portal/neocaptar_project_search.php', params,
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
