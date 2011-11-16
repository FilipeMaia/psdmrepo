
function p_appl_projects() {
	var that = this;
	this.name = 'projects';
	this.full_name = 'Projects';
	this.context = '';
	this.when_done=null;
	this.create_form_changed=false;
	this.select = function(ctx,when_done) {
		that.context = ctx;
		this.when_done = when_done;
		this.init();
	};
	this.select_default = function() {
		if( this.context == '' ) this.context = 'search';
		this.init();
	};
	this.if_ready2giveup = function( handler2call ) {
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
	this.initialized = false;
	this.init = function() {
		if( this.initialized ) return;
		this.initialized = true;
		$('#projects-search-search').button().click(function() {
			that.load();
		});
		$('#projects-search-reset').button().click(function() {
			alert('to be implemented');
		});
		$('#projects-search-after').datepicker().datepicker('option','dateFormat','yy-mm-dd');
		$('#projects-search-before').datepicker().datepicker('option','dateFormat','yy-mm-dd');
		$('#projects-create-form').find('input[name="due"]').each(function() { $(this).datepicker().datepicker('option','dateFormat','yy-mm-dd'); });
		$('#projects-create-save').button().click(function() {
			$(this).button('disable');
			// TODO: Here be the actual POST form submittion to the server
			//
			that.create_form_changed = false;
			if( that.when_done ) {
				that.when_done.execute();
				that.when_done = null;
			}
		}).button('disable');
		$('#projects-create-reset').button().click(function() {});
		$('.projects-create-form-element').change(function() {
			that.create_form_changed = true;
			$('#projects-create-save').button('enable');
		});
		this.load();
	};
	this.projects = null;
	this.toggle_project = function(idx) {
		var toggler='#proj-tgl-'+idx;
		var container='#proj-con-'+idx;
		if( $(container).hasClass('proj-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('proj-hdn').addClass('proj-vis');

			/* If the project's cables haven't been loaded then initialize
			 * the project page and load the cables.
			 */
			var proj = this.projects[idx];
			if( !proj.is_loaded ) {
				proj.is_loaded = true;
				$('#proj-change-due-'+idx).datepicker().datepicker('option','dateFormat','yy-mm-dd').datepicker('setDate',proj.due);
				$('#proj-change-'+idx+'-save').button().click(function() {
					that.projects[idx].owner = $('#proj-change-owner-'+idx).val();
					that.projects[idx].title = $('#proj-change-title-'+idx).val();
					$('#proj-hdr-'+idx).find('.proj-owner').each(function() { $(this).html(that.projects[idx].owner); });
					$('#proj-hdr-'+idx).find('.proj-title').each(function() { $(this).html(that.projects[idx].title); });
					$(this).button('disable');
					that.toggle_project_editor(idx);
				}).button('disable');
				$('.proj-change-elem').change(function() { $('#proj-change-'+idx+'-save').button('enable'); });
				$('#proj-add-'+idx    ).button().click(function() { that.add_cable(idx); });
				$('#proj-delete-'+idx ).button().click(function() { that.delete_project(idx); });
				$('#proj-submit-'+idx ).button().click(function() {
					ask_yes_no(
						"Confirm Project Submission",
						"You are about to submit the project for production."+
						" When the process will finish each cable will be assigned a job number corresponding to the project's owner"+
						" and a unique cable number. Are you ready to proceed?",
						function() { that.submit_project(idx); },
						null );							
				});
				this.load_cables(idx);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('proj-vis').addClass('proj-hdn');
		}
	};
	this.toggle_project_editor = function(idx) {
		var toggler='#proj-editor-tgl-'+idx;
		var container='#proj-editor-con-'+idx;
		if( $(container).hasClass('proj-editor-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('proj-editor-hdn').addClass('proj-editor-vis');
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('proj-editor-vis').addClass('proj-editor-hdn');
		}
	};
	this.cable2html = function(pidx,cidx) {
		var c = this.projects[pidx].cables[cidx];
		var html =
'<tr class="table_row " id="proj-cable-'+pidx+'-'+cidx+'-1">'+
'  <td nowrap="nowrap" class="table_cell table_cell_left table_cell_bottom "><div class="status"><b>'+c.status+'</b></div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom " id="proj-cable-tools-'+pidx+'-'+cidx+'-1" >'+
'    <button class="proj-cable-tool" name="edit"        onclick="projects.edit_cable('+pidx+','+cidx+')"        title="edit"><b>E</b></button>'+
'    <button class="proj-cable-tool" name="edit_save"   onclick="projects.edit_cable_save('+pidx+','+cidx+')"   title="save changes to the database">save</button>'+
'    <button class="proj-cable-tool" name="edit_cancel" onclick="projects.edit_cable_cancel('+pidx+','+cidx+')" title="cancel editing and ignore any changes">cancel</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left table_cell_bottom ">'+c.jobnum+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom ">'+c.cablenum+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "><div class="system">' +c.system+ '</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "><div class="func">'   +c.func+   '</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "><div class="type">'   +c.type+   '</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "><div class="length">' +c.length+ '</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_bottom "><div class="routing">'+c.routing+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell ">'+c.origin.name+'</td>'+
'  <td nowrap="nowrap" class="table_cell "><div class="origin_loc">'+c.origin.loc+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell "><div class="origin_rack">'+c.origin.rack+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell ">'+c.origin.ele+'</td>'+
'  <td nowrap="nowrap" class="table_cell ">'+c.origin.side+'</td>'+
'  <td nowrap="nowrap" class="table_cell ">'+c.origin.slot+'</td>'+
'  <td nowrap="nowrap" class="table_cell ">'+c.origin.connum+'</td>'+
'  <td nowrap="nowrap" class="table_cell "><div class="origin_pinlist">'+c.origin.pinlist+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell ">'+c.origin.station+'</td>'+
'  <td nowrap="nowrap" class="table_cell "><div class="origin_conntype">'+c.origin.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right "><div class="origin_instr">'+c.origin.instr+'</div></td>'+
'</tr>'+
'<tr class="table_row " id="proj-cable-'+pidx+'-'+cidx+'-2">';
		var action = '';
		switch(c.status) {
		case 'Planned':      action = '<button class="proj-cable-tool" name="register"   onclick="projects.register_cable('+pidx+','+cidx+')"   title="proceed to the formal registration to obtain official JOB and CABLE numbers" ><b>REG</b></button>'; break;
		case 'Registered':   action = '<button class="proj-cable-tool" name="label"      onclick="projects.label_cable('+pidx+','+cidx+')"      title="produce a standard label" ><b>LBL</b></button>'; break;
		case 'Labeled':      action = '<button class="proj-cable-tool" name="fabricate"  onclick="projects.fabricate_cable('+pidx+','+cidx+')"  title="order fabrication" ><b>FAB</b></button>'; break;
		case 'Fabrication':  action = '<button class="proj-cable-tool" name="ready"      onclick="projects.ready_cable('+pidx+','+cidx+')"      title="get ready for installation" ><b>RDY</b></button>'; break;
		case 'Ready':        action = '<button class="proj-cable-tool" name="install"    onclick="projects.install_cable('+pidx+','+cidx+')"    title="install" ><b>INS</b></button>'; break;
		case 'Installed':    action = '<button class="proj-cable-tool" name="commission" onclick="projects.commission_cable('+pidx+','+cidx+')" title="test and commission" ><b>COM</b></button>'; break;
		case 'Commissioned': action = '<button class="proj-cable-tool" name="damage"     onclick="projects.damage_cable('+pidx+','+cidx+')"     title="mark as damaged, needing replacement" ><b>DMG</b></button>'; break;
		case 'Damaged':      action = '<button class="proj-cable-tool" name="retire"     onclick="projects.retire_cable('+pidx+','+cidx+')"     title="mark as retired" ><b>RTR</b></button>'; break;
		}
		html +=
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left " align="right"><div id="proj-cable-action-'+pidx+'-'+cidx+'">'+action+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "id="proj-cable-tools-'+pidx+'-'+cidx+'-2" >'+
'    <button class="proj-cable-tool" name="clone"   onclick="projects.clone_cable('+pidx+','+cidx+')"        title="clone"  ><b>C</b></button>'+
'    <button class="proj-cable-tool" name="delete"  onclick="projects.delete_cable('+pidx+','+cidx+')"       title="delete" ><b>D</b></button>'+
'    <button class="proj-cable-tool" name="history" onclick="projects.show_cable_history('+pidx+','+cidx+')" title="history"><b>H</b></button>'+
'    <button class="proj-cable-tool" name="label"   onclick="projects.show_cable_label('+pidx+','+cidx+')"   title="label"  ><b>L</b></button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_left "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom "></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight ">'+c.destination.name+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_loc">'+c.destination.loc+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_rack">'+c.destination.rack+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight ">'+c.destination.ele+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight ">'+c.destination.side+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight ">'+c.destination.slot+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight ">'+c.destination.connum+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_pinlist">'+c.destination.pinlist+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight ">'+c.destination.station+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight "><div class="destination_conntype">'+c.destination.conntype+'</div></td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_strong_bottom table_cell_highlight table_cell_right "><div class="destination_instr">'+c.destination.instr+'</div></td>'+
'</tr>';
		return html;
	};
	this.display_cables = function(pidx) {
		var proj = this.projects[pidx];
		var prev = $('#proj-cables-hdr-'+pidx);
		prev.siblings().remove();
		for( var cidx in proj.cables ) {
			var html = this.cable2html(pidx, cidx);
			prev.after( html );
			prev = $('#proj-cable-'+pidx+'-'+cidx+'-2');
		}
		$('.proj-cable-tool').button();
		for( var cidx in proj.cables )
			this.update_cable_tools(pidx,cidx,false);
	};
	this.load_cables = function(pidx) {
		$('#proj-cables-load-'+pidx).html('Loading...');
		var params = {};
		var jqXHR = $.get(
			'../portal/neocaptar_search_cables_JSON.php', params,
			function(data) {
				if( data.status != 'success') {
					alert('failed to load cables because of: '+data.message);
					return;
				}
				that.projects[pidx].cables = data.cables;
				that.display_cables(pidx);
			},
			'JSON'
		).error(
			function () {
				alert('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
				$('#proj-cables-load-'+pidx).html('');
			}
		);
	};
	this.update_cable_tools = function(pidx,cidx,editing) {
		var cable = this.projects[pidx].cables[cidx];
		if( editing ) {
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'       ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_save"]'  ).button('enable' );
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_cancel"]').button('enable' );
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="clone"]'      ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]'     ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="history"]'    ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]'      ).button('disable');
			$('#proj-cable-action-'+pidx+'-'+cidx+' button').button('disable');
			return;
		}
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_save"]'  ).button('disable');
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit_cancel"]').button('disable');
		if( cable.status == 'Planned') {
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'  ).button('enable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]').button('enable');
		} else {
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-1 button[name="edit"]'  ).button('disable');
			$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="delete"]').button('disable');
		}
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="clone"]'  ).button('enable');
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="history"]').button('enable');
		$('#proj-cable-tools-'+pidx+'-'+cidx+'-2 button[name="label"]'  ).button('enable');
		$('#proj-cable-action-'+pidx+'-'+cidx+' button').button('enable');
	};
	this.update_project_tools = function(pidx) {
		var html = '';
		var proj = this.projects[pidx];
		if(proj.num_edited) {
			html =
'<b>Warning:</b> total of '+proj.num_edited+' cables are being edited.<br>'+
'Some controls on this project page will be disabled before you save<br>'+
'or cancel the editing dialogs.';
			$('#proj-sort-'+pidx).children().attr('disabled','disabled');
			$('#proj-displ-'+pidx).children().attr('disabled','disabled');
			$('#proj-delete-'+pidx).button('disable');
			$('#proj-submit-'+pidx).button('disable');
		} else {
			$('#proj-sort-'+pidx).children().removeAttr('disabled');
			$('#proj-displ-'+pidx).children().removeAttr('disabled');
			$('#proj-delete-'+pidx).button('enable');
			$('#proj-submit-'+pidx).button('enable');
		}
		$('#proj-alerts-'+pidx).html(html);
	};
	this.add_cable = function(pidx) {
		var proj = this.projects[pidx];
		/* First we need to submit the request to the database service in order
		 * to get a unique cable identifier.
		 */
		var params = { pid: proj.id };
		var jqXHR = $.get(
			'../portal/neocaptar_new_cable_JSON.php', params,
			function(data) {
				if( data.status != 'success') {
					alert('failed to load cables because of: '+data.message);
					return;
				}
				var cidx_new = proj.cables.push(data.cable) - 1;
				$('#proj-cables-hdr-'+pidx).after( that.cable2html(pidx, cidx_new))
				$('#proj-cable-'+pidx+'-'+cidx_new+'-1').find('.proj-cable-tool').each( function() { $(this).button(); });
				$('#proj-cable-'+pidx+'-'+cidx_new+'-2').find('.proj-cable-tool').each( function() { $(this).button(); });
				that.edit_cable(pidx,cidx_new);
			},
			'JSON'
		).error(
			function () {
				alert('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
				$('#proj-cables-load-'+pidx).html('');
			}
		);
	};
	this.delete_project = function(pidx) {
		ask_yes_no(
			'Data Deletion Warning',
			'You are about to delete the project and all cables associated with it. Are you sure?',
			function() {
				delete that.projects[pidx];
				that.display();
			},
			null );
	};
	this.submit_project = function(pidx) {
		var proj = this.projects[pidx];
		proj.is_submitted = 1;
		var sec = mktime();
		proj.submitted_sec = sec*1000*1000*1000;
		var d = new Date(sec*1000);
		proj.submitted = d.format('yyyy-mm-dd HH:MM:ss');
		$('#proj-hdr-'+pidx).find('.proj-status').each(function() { $(this).html('<b>submitted</b>'); });
		$('#proj-hdr-'+pidx).find('.proj-submitted').each(function() { $(this).html(proj.submitted); });
		this.toggle_project(pidx);
		this.toggle_project(pidx);
	};
	this.show_cable_history = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('show_cable_history: '+pidx+'.'+cidx);
	};
	this.clone_cable = function(pidx,cidx) {
		var proj  = this.projects[pidx];
		var cable = proj.cables[cidx];
		/* First we need to submit the request to the database service in order
		 * to get a unique cable identifier.
		 */
		var params = { cid: cable.id };
		var jqXHR = $.get(
			'../portal/neocaptar_new_cable_JSON.php', params,
			function(data) {
				if( data.status != 'success') {
					alert('failed to load cables because of: '+data.message);
					return;
				}
				var cidx_new = proj.cables.push(data.cable) - 1;
				$('#proj-cable-'+pidx+'-'+cidx+'-1').before( that.cable2html(pidx, cidx_new))
				$('#proj-cable-'+pidx+'-'+cidx_new+'-1').find('.proj-cable-tool').each( function() { $(this).button(); });
				$('#proj-cable-'+pidx+'-'+cidx_new+'-2').find('.proj-cable-tool').each( function() { $(this).button(); });
				that.edit_cable(pidx,cidx_new);
			},
			'JSON'
		).error(
			function () {
				alert('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
				$('#proj-cables-load-'+pidx).html('');
			}
		);
	};
	this.delete_cable = function(pidx,cidx) {
		ask_yes_no(
			'Data Deletion Warning',
			'You are about to delete the cable and all information associated with it. Are you sure?',
			function() {
				var proj  = that.projects[pidx];
				var cable = proj.cables[cidx];
				$('#proj-cable-'+pidx+'-'+cidx+'-1').remove();
				$('#proj-cable-'+pidx+'-'+cidx+'-2').remove();
				proj.cables.total--;
				switch(cable.status) {
				case 'Planned':      proj.cables.Planned--;      break;
				case 'Registered':   proj.cables.Registered--;   break;
				case 'Labeled':      proj.cables.Labeled--;      break;
				case 'Fabrication':  proj.cables.Fabrication--;  break;
				case 'Ready':        proj.cables.Ready--;        break;
				case 'Installed':    proj.cables.Installed--;    break;
				case 'Commissioned': proj.cables.Commissioned--; break;
				case 'Damaged':      proj.cables.Damaged--;      break;
				case 'Retired':      proj.cables.Retired--;      break;
				}
				/*
				 * TODO: Also update the project header bar
				 */
				delete proj.cables[cidx];
			},
			null );
	};
	this.cable_property2html_edit = function(pidx,cidx,prop,is_new) {

		var cable = this.projects[pidx].cables[cidx];
		var base  = '#proj-cable-'+pidx+'-'+cidx;
		var html  = '';

		switch(prop) {

		case 'status':
			$(base+'-1 td div.'+prop).html('<span style="color:maroon;">Editing...</span>');
			break;

		case 'system':
			$(base+'-1 td div.'+prop).html('<input type="text" value="'+cable.system+'" size="10" />');
			break;

		case 'func':
			$(base+'-1 td div.'+prop).html('<input type="text" value="'+cable.func+'" size="24" />');
			break;

		case 'type':

			base += '-1 td div.'+prop;

			if( is_new ) {

				$(base).html('<input type="text" value="" size="4" />');
				cable.read_type = function() { return $(base+' input').val(); };

			} else if( dict.cable_dict_is_empty()) {

				$(base).html('<input type="text" value="'+cable.type+'" size="4" />');
				cable.read_type = function() { return $(base+' input').val(); };

			} else {

				html = '<select>';
				if( cable.type == '' ) {
					for( var type in dict.cables()) {
						html += '<option';
						if( type == cable.type ) html += ' selected="selected"';
						html += ' value="'+type+'">'+type+'</option>';
					}
				} else {

					if( dict.cable_is_not_known( cable.type )) {

						html += '<option selected="selected" value="'+cable.type+'">'+cable.type+'</option>';
						for( var type in dict.cables())
							html += '<option value="'+type+'">'+type+'</option>';

					} else {

						for( var type in dict.cables()) {
							html += '<option';
							if( type == cable.type ) html += ' selected="selected"';
							html += ' value="'+type+'">'+type+'</option>';
						}
					}
				}
				html += '<option value="" style="color:maroon;" title="register new cable type" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop,true);
						that.cable_property2html_edit(pidx,cidx,'origin_conntype',true);
						that.cable_property2html_edit(pidx,cidx,'origin_pinlist',true);
						that.cable_property2html_edit(pidx,cidx,'destination_conntype',true);
						that.cable_property2html_edit(pidx,cidx,'destination_pinlist',true);
					} else {
						that.cable_property2html_edit(pidx,cidx,'origin_conntype',false);
						that.cable_property2html_edit(pidx,cidx,'origin_pinlist',false);
						that.cable_property2html_edit(pidx,cidx,'destination_conntype',false);
						that.cable_property2html_edit(pidx,cidx,'destination_pinlist',false);
					}
				});
				cable.read_type = function() { return $(base+' select').val(); };
			}
			break;

		case 'length':
			$(base+'-1 td div.'+prop).html('<input type="text" value="'+cable.length+ '" size="1"  />');
			break;

		case 'routing':
			base += '-1 td div.'+prop;
			if( is_new ) {
				$(base).html('<input type="text" value="" size="8"  />');
				cable.read_routing = function() { return $(base+' input').val(); };
			} else {
				html = '<select>';
				for( var i in global_routing ) {
					var routing = global_routing[i];
					html += '<option';
					if( routing == cable.routing ) html += ' selected="selected"';
					html += ' value="'+routing+'">'+routing+'</option>';
				}
				html += '<option value="" style="color:maroon;" title="register new routing" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop,true);
					}
				});
				cable.read_routing = function() { return $(base+' select').val(); };
			}
			break;

		case 'origin_conntype':

			base += '-1 td div.'+prop;

			if( is_new ) {

				$(base).html('<input type="text" value="" size="4" />');
				cable.origin.read_conntype = function() { return $(base+' input').val(); };

			} else if( dict.cable_dict_is_empty()) {

				$(base).html('<input type="text" value="'+cable.origin.conntype+'" size="4" />');
				cable.origin.read_conntype = function() { return $(base+' input').val(); };

			} else {

				if(	dict.cable_is_not_known( cable.read_type())) {

					$(base).html('<input type="text" value="'+cable.origin.conntype+'" size="4" />');
					cable.origin.read_conntype = function() { return $(base+' input').val(); };

				} else {

					if( dict.connector_dict_is_empty( cable.read_type())) {

						$(base).html('<input type="text" value="'+cable.origin.conntype+'" size="4" />');
						cable.origin.read_conntype = function() { return $(base+' input').val(); };

					} else {

						html = '<select>';
						for( var type in dict.connectors( cable.read_type())) {
							html += '<option';
							if( type == cable.origin.conntype ) html += ' selected="selected"';
							html += ' value="'+type+'">'+type+'</option>';
						}
						html += '<option value=""  style="color:maroon;" title="register new connector type for the given cable type" >New...</option>'
						html += '</select>';
						$(base).html(html);
						$(base+' select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property2html_edit(pidx,cidx,prop,true);
								that.cable_property2html_edit(pidx,cidx,'origin_pinlist',true);
							} else {
								that.cable_property2html_edit(pidx,cidx,'origin_pinlist', false);
							}
						});
						cable.origin.read_conntype = function() { return $(base+' select').val(); };
					}
				}
			}
			break;

		case 'origin_pinlist':

			base += '-1 td div.'+prop;

			if( is_new ) {

				$(base).html('<input type="text" value="" size="4" />');
				cable.origin.read_pinlist = function() { return $(base+' input').val(); };

			} else if( dict.cable_dict_is_empty()) {

				$(base).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
				cable.origin.read_pinlist = function() { return $(base+' input').val(); };

			} else if( dict.cable_is_not_known( cable.read_type())) {

				$(base).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
				cable.origin.read_pinlist = function() { return $(base+' input').val(); };

			} else if( dict.connector_is_not_known( cable.read_type(), cable.origin.read_conntype())) {

				$(base).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
				cable.origin.read_pinlist = function() { return $(base+' input').val(); };

			} else {

				if( dict.pinlist_dict_is_empty( cable.read_type(), cable.origin.read_conntype())) {

					$(base).html('<input type="text" value="'+cable.origin.pinlist+'" size="4" />');
					cable.origin.read_pinlist = function() { return $(base+' input').val(); };

				} else {

					html = '<select>';
					for( var pinlist in dict.pinlists( cable.read_type(), cable.origin.read_conntype())) {
						html += '<option';
						if( pinlist == cable.origin.pinlist ) html += ' selected="selected"';
						html += ' value="'+pinlist+'">'+pinlist+'</option>';
					}
					html += '<option value="" style="color:maroon;" title="register new pinlist for the given connector type" >New...</option>'
						html += '</select>';
					$(base).html(html);
					$(base+' select').change(function() {
						if( $(this).val() == '') {
							that.cable_property2html_edit(pidx,cidx,prop,true);
						}
					});
					cable.origin.read_pinlist = function() { return $(base+' select').val(); };
				}
			}
			break;
	
		case 'origin_loc':

			base += '-1 td div.'+prop;

			if( is_new ) {

				// Start with empty
				//
				$(base).html('<input type="text" value="" size="1" />');
				cable.origin.read_loc = function() { return $(base+' input').val(); };

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
				html += '<option value="" style="color:maroon;" title="register new location" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop, true);
						that.cable_property2html_edit(pidx,cidx,'origin_rack',true);
					} else {
						that.cable_property2html_edit(pidx,cidx,'origin_rack',false);
					}
				});
				cable.origin.read_loc = function() { return $(base+' select').val(); };
			}
			break;

		case 'origin_rack':

			base += '-1 td div.'+prop;

			if( is_new ) {

				// Start with empty
				//
				$(base).html('<input type="text" value="" size="1" />');
				cable.origin.read_rack = function() { return $(base+' input').val(); };

			} else if( dict.location_is_not_known( cable.origin.read_loc())) {

				// This might came from a pre-existing  database. So we have to respect
				// a choice of a rack because the current location is also not known to
				// the dictionary.
				//
				$(base).html('<input type="text" value="'+cable.origin.rack+'" size="1" />');
				cable.origin.read_rack = function() { return $(base+' input').val(); };

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
				html += '<option value="" style="color:maroon;" title="register new rack for the selected location" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop,true);
					}
				});
				cable.origin.read_rack = function() { return $(base+' select').val(); };
			}
			break;

		case 'origin_instr':
			base += '-1 td div.'+prop;
			if( is_new ) {
				$(base).html('<input type="text" value="" size="1" />');
				cable.origin.read_instr = function() { return $(base+' input').val(); };
			} else {
				html = '<select>';
				for( var i in global_instr ) {
					var instr = global_instr[i];
					html += '<option';
					if( instr == cable.origin.instr ) html += ' selected="selected"';
					html += ' value="'+instr+'">'+instr+'</option>';
				}
				html += '<option value="" style="color:maroon;" title="register new instructions" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop,true);
					}
				});
				cable.origin.read_instr = function() { return $(base+' select').val(); };
			}
			break;

		case 'destination_conntype':

			base += '-2 td div.'+prop;

			if( is_new ) {

				$(base).html('<input type="text" value="" size="4" />');
				cable.destination.read_conntype = function() { return $(base+' input').val(); };

			} else if( dict.cable_dict_is_empty()) {

				$(base).html('<input type="text" value="'+cable.destination.conntype+'" size="4" />');
				cable.destination.read_conntype = function() { return $(base+' input').val(); };

			} else {

				if(	dict.cable_is_not_known( cable.read_type())) {

					$(base).html('<input type="text" value="'+cable.destination.conntype+'" size="4" />');
					cable.destination.read_conntype = function() { return $(base+' input').val(); };

				} else {

					if( dict.connector_dict_is_empty( cable.read_type())) {

						$(base).html('<input type="text" value="'+cable.destination.conntype+'" size="4" />');
						cable.destination.read_conntype = function() { return $(base+' input').val(); };

					} else {
						html = '<select>';

						for( var type in dict.connectors( cable.read_type())) {
							html += '<option';
							if( type == cable.destination.conntype ) html += ' selected="selected"';
							html += ' value="'+type+'">'+type+'</option>';
						}
						html += '<option value=""  style="color:maroon;" title="register new connector type for the given cable type" >New...</option>'
						html += '</select>';
						$(base).html(html);
						$(base+' select').change(function() {
							if( $(this).val() == '' ) {
								that.cable_property2html_edit(pidx,cidx,prop,true);
								that.cable_property2html_edit(pidx,cidx,'destination_pinlist',true);
							} else {
								that.cable_property2html_edit(pidx,cidx,'destination_pinlist', false);
							}
						});
						cable.destination.read_conntype = function() { return $(base+' select').val(); };
					}
				}
			}
			break;

		case 'destination_pinlist':

			base += '-2 td div.'+prop;

			if( is_new ) {

				$(base).html('<input type="text" value="" size="4" />');
				cable.destination.read_pinlist = function() { return $(base+' input').val(); };

			} else if( dict.cable_dict_is_empty()) {

				$(base).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
				cable.destination.read_pinlist = function() { return $(base+' input').val(); };

			} else if( dict.cable_is_not_known( cable.read_type())) {

				$(base).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
				cable.destination.read_pinlist = function() { return $(base+' input').val(); };

			} else if( dict.connector_is_not_known( cable.read_type(), cable.destination.read_conntype())) {

				$(base).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
				cable.destination.read_pinlist = function() { return $(base+' input').val(); };

			} else {

				if( dict.pinlist_dict_is_empty( cable.read_type(), cable.destination.read_conntype())) {

					$(base).html('<input type="text" value="'+cable.destination.pinlist+'" size="4" />');
					cable.destination.read_pinlist = function() { return $(base+' input').val(); };

				} else {

					html = '<select>';
					for( var pinlist in dict.pinlists( cable.read_type(), cable.destination.read_conntype())) {
						html += '<option';
						if( pinlist == cable.destination.pinlist ) html += ' selected="selected"';
						html += ' value="'+pinlist+'">'+pinlist+'</option>';
					}
					html += '<option value="" style="color:maroon;" title="register new pinlist for the given connector type" >New...</option>'
						html += '</select>';
					$(base).html(html);
					$(base+' select').change(function() {
						if( $(this).val() == '') {
							that.cable_property2html_edit(pidx,cidx,prop,true);
						}
					});
					cable.destination.read_pinlist = function() { return $(base+' select').val(); };
				}
			}
			break;

		case 'destination_loc':

			base += '-2 td div.'+prop;

			if( is_new ) {

				// Start with empty
				//
				$(base).html('<input type="text" value="" size="1" />');
				cable.destination.read_loc = function() { return $(base+' input').val(); };

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
				html += '<option value="" style="color:maroon;" title="register new location" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop, true);
						that.cable_property2html_edit(pidx,cidx,'destination_rack',true);
					} else {
						that.cable_property2html_edit(pidx,cidx,'destination_rack',false);
					}
				});
				cable.destination.read_loc = function() { return $(base+' select').val(); };
			}
			break;

		case 'destination_rack':

			base += '-2 td div.'+prop;

			if( is_new ) {

				// Start with empty
				//
				$(base).html('<input type="text" value="" size="1" />');
				cable.destination.read_rack = function() { return $(base+' input').val(); };

			} else if( dict.location_is_not_known( cable.destination.read_loc())) {

				// This might came from a pre-existing  database. So we have to respect
				// a choice of a rack because the current location is also not known to
				// the dictionary.
				//
				$(base).html('<input type="text" value="'+cable.destination.rack+'" size="1" />');
				cable.destination.read_rack = function() { return $(base+' input').val(); };

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
				html += '<option value="" style="color:maroon;" title="register new rack for the selected location" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop,true);
					}
				});
				cable.destination.read_rack = function() { return $(base+' select').val(); };
			}
			break;

		case 'destination_instr':
			base += '-2 td div.'+prop;
			if( is_new ) {
				$(base).html('<input type="text" value="" size="1" />');
				cable.destination.read_instr = function() { return $(base+' input').val(); };
			} else {
				html = '<select>';
				for( var i in global_instr ) {
					var instr = global_instr[i];
					html += '<option';
					if( instr == cable.destination.instr ) html += ' selected="selected"';
					html += ' value="'+instr+'">'+instr+'</option>';
				}
				html += '<option value="" style="color:maroon;" title="register new instructions" >New...</option>'
				html += '</select>';
				$(base).html(html);
				$(base+' select').change(function() {
					if( $(this).val() == '' ) {
						that.cable_property2html_edit(pidx,cidx,prop,true);
					}
				});
				cable.destination.read_instr = function() { return $(base+' select').val(); };
			}
			break;
		}
	};
	this.cable_property2html_view = function(pidx,cidx,prop) {

		var cable = this.projects[pidx].cables[cidx];
		var base  = '#proj-cable-'+pidx+'-'+cidx;

		switch(prop) {

		case 'status' : $(base+'-1 td div.'+prop).html(cable.status); break;
		case 'system' : $(base+'-1 td div.'+prop).html(cable.system); break;
		case 'func'   : $(base+'-1 td div.'+prop).html(cable.func); break;
		case 'type'   : $(base+'-1 td div.'+prop).html(cable.type); break;
		case 'length' : $(base+'-1 td div.'+prop).html(cable.length); break;
		case 'routing': $(base+'-1 td div.'+prop).html(cable.routing); break;

		case 'origin_loc'     : $(base+'-1 td div.'+prop).html(cable.origin.loc); break;
		case 'origin_rack'    : $(base+'-1 td div.'+prop).html(cable.origin.rack); break;
		case 'origin_conntype': $(base+'-1 td div.'+prop).html(cable.origin.conntype); break;
		case 'origin_pinlist' : $(base+'-1 td div.'+prop).html(cable.origin.pinlist); break;
		case 'origin_instr'   : $(base+'-1 td div.'+prop).html(cable.origin.instr); break;

		case 'destination_loc'     : $(base+'-2 td div.'+prop).html(cable.destination.loc); break;
		case 'destination_rack'    : $(base+'-2 td div.'+prop).html(cable.destination.rack); break;
		case 'destination_conntype': $(base+'-2 td div.'+prop).html(cable.destination.conntype); break;
		case 'destination_pinlist' : $(base+'-2 td div.'+prop).html(cable.destination.pinlist); break;
		case 'destination_instr'   : $(base+'-2 td div.'+prop).html(cable.destination.instr); break;
		}
	};
	this.edit_cable = function(pidx,cidx) {

		this.projects[pidx].num_edited++;

		this.update_cable_tools(pidx,cidx,true);
		this.update_project_tools(pidx);

		this.cable_property2html_edit(pidx,cidx,'status',false);
		this.cable_property2html_edit(pidx,cidx,'system',false);
		this.cable_property2html_edit(pidx,cidx,'func',false);
		this.cable_property2html_edit(pidx,cidx,'type',false);
		this.cable_property2html_edit(pidx,cidx,'length',false);
		this.cable_property2html_edit(pidx,cidx,'routing',false);

		this.cable_property2html_edit(pidx,cidx,'origin_loc',false);
		this.cable_property2html_edit(pidx,cidx,'origin_rack',false);
		this.cable_property2html_edit(pidx,cidx,'origin_conntype',false);
		this.cable_property2html_edit(pidx,cidx,'origin_pinlist',false);
		this.cable_property2html_edit(pidx,cidx,'origin_instr',false);


		this.cable_property2html_edit(pidx,cidx,'destination_loc',false);
		this.cable_property2html_edit(pidx,cidx,'destination_rack',false);
		this.cable_property2html_edit(pidx,cidx,'destination_conntype',false);
		this.cable_property2html_edit(pidx,cidx,'destination_pinlist',false);
		this.cable_property2html_edit(pidx,cidx,'destination_instr',false);
	};
	this.edit_cable_save = function(pidx,cidx) {

		this.projects[pidx].num_edited--;

		this.update_cable_tools(pidx,cidx,false);
		this.update_project_tools(pidx);

		var cable = this.projects[pidx].cables[cidx];

		cable.system = $('#proj-cable-'+pidx+'-'+cidx+'-1 td div.system input').val();
		cable.func   = $('#proj-cable-'+pidx+'-'+cidx+'-1 td div.func input').val();

		cable.length  = $('#proj-cable-'+pidx+'-'+cidx+'-1 td div.length input' ).val();

		// ** dictionary (routing) **

		var routing = cable.read_routing();
		if( global_routing.indexOf(routing) == -1 ) global_routing.push(routing);
		cable.routing = routing;

		// ** dictionary: (location,rack) **

		var origin_loc = cable.origin.read_loc();
		cable.origin.loc = origin_loc;
		dict.save_location(origin_loc);

		var origin_rack = cable.origin.read_rack();
		cable.origin.rack = origin_rack;
		dict.save_rack(origin_loc, origin_rack);

		var destination_loc = cable.destination.read_loc();
		cable.destination.loc = destination_loc;
		dict.save_location(destination_loc);

		var destination_rack = cable.destination.read_rack();
		cable.destination.rack = destination_rack;
		dict.save_rack(destination_loc, destination_rack);

		// ** dictionary: (cable,connector,pinlist) **

		var type = cable.read_type();
		cable.type = type;
		dict.save_cable(type);

		var origin_conntype = cable.origin.read_conntype();
		cable.origin.conntype = origin_conntype;
		dict.save_connector(type,origin_conntype);

		var origin_pinlist = cable.origin.read_pinlist();
		cable.origin.pinlist = origin_pinlist;
		dict.save_pinlist(type,origin_conntype,origin_pinlist);

		var origin_instr = cable.origin.read_instr();
		if( global_instr.indexOf(origin_instr) == -1 ) global_instr.push(origin_instr);
		cable.origin.instr = origin_instr;

		var destination_conntype = cable.destination.read_conntype();
		cable.destination.conntype = destination_conntype;
		dict.save_connector(type,destination_conntype);

		var destination_pinlist = cable.destination.read_pinlist();
		cable.destination.pinlist = destination_pinlist;
		dict.save_pinlist(type,destination_conntype,destination_pinlist);

		// ** dictionary: (instructions) **

		var destination_instr = cable.destination.read_instr();
		if( global_instr.indexOf(destination_instr) == -1 ) global_instr.push(destination_instr);
		cable.destination.instr = destination_instr;


		this.edit_cable_view(pidx,cidx);
	};
	this.edit_cable_cancel = function(pidx,cidx) {

		this.projects[pidx].num_edited--;

		this.update_cable_tools(pidx,cidx,false);
		this.update_project_tools(pidx);

		this.edit_cable_view(pidx,cidx);
	};
	this.edit_cable_view = function(pidx,cidx) {

		this.cable_property2html_view(pidx,cidx,'status');

		this.cable_property2html_view(pidx,cidx,'system');
		this.cable_property2html_view(pidx,cidx,'func');
		this.cable_property2html_view(pidx,cidx,'type');
		this.cable_property2html_view(pidx,cidx,'length');
		this.cable_property2html_view(pidx,cidx,'routing');

		this.cable_property2html_view(pidx,cidx,'origin_loc');
		this.cable_property2html_view(pidx,cidx,'origin_rack');
		this.cable_property2html_view(pidx,cidx,'origin_conntype');
		this.cable_property2html_view(pidx,cidx,'origin_pinlist');
		this.cable_property2html_view(pidx,cidx,'origin_instr');

		this.cable_property2html_view(pidx,cidx,'destination_loc');
		this.cable_property2html_view(pidx,cidx,'destination_rack');
		this.cable_property2html_view(pidx,cidx,'destination_conntype');
		this.cable_property2html_view(pidx,cidx,'destination_pinlist');
		this.cable_property2html_view(pidx,cidx,'destination_instr');
	};
	this.show_cable_label = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('show_cable_label: '+pidx+'.'+cidx);
	};
	this.register_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('register_cable: '+pidx+'.'+cidx);
	};
	this.label_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('label_cable: '+pidx+'.'+cidx);
	};
	this.fabricate_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('fabricate_cable: '+pidx+'.'+cidx);
	};
	this.ready_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('ready_cable: '+pidx+'.'+cidx);
	};
	this.install_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('install_cable: '+pidx+'.'+cidx);
	};
	this.commission_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('commission_cable: '+pidx+'.'+cidx);
	};
	this.damage_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('damage_cable: '+pidx+'.'+cidx);
	};
	this.retire_cable = function(pidx,cidx) {
		var cable = this.projects[pidx].cables[cidx];
		alert('retire_cable: '+pidx+'.'+cidx);
	};
	this.select_cables_by_status = function(pidx) {
		var proj = this.projects[pidx];
		var status = $('#proj-cables-hdr-'+pidx+' td select').val();
		if( '- status -' == status ) {
			for( var cidx in proj.cables ) {
				$('#proj-cable-'+pidx+'-'+cidx+'-1').css('display','');
				$('#proj-cable-'+pidx+'-'+cidx+'-2').css('display','');
			}
		} else {
			for( var cidx in proj.cables ) {
				var style = proj.cables[cidx].status == status ? '' : 'none';
				$('#proj-cable-'+pidx+'-'+cidx+'-1').css('display',style);
				$('#proj-cable-'+pidx+'-'+cidx+'-2').css('display',style);
			}
		}
	};
	this.project2html = function(idx) {
		var p = this.projects[idx];
		var html =
'<div class="proj-hdr" id="proj-hdr-'+idx+'" onclick="projects.toggle_project('+idx+');">'+
'  <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e" id="proj-tgl-'+idx+'"></span></div>'+
'  <div style="float:left;" class="proj-created">'+p.created+'</div>'+
'  <div style="float:left;" class="proj-owner">'+p.owner+'</div>'+
'  <div style="float:left;" class="proj-title">'+p.title+'</div>'+
'  <div style="float:left;" class="proj-due">'+p.due+'</div>'+
'  <div style="float:left;" class="proj-num-cables-total">'+p.status.total+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Planned+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Registered+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Labeled+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Fabrication+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Ready+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Installed+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Commissioned+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Damaged+'</div>'+
'  <div style="float:left;" class="proj-num-cables">'+p.status.Retired+'</div>'+
'  <div style="float:left;" class="proj-status">'+(p.is_submitted?'<b>submitted</b>':'in-progress')+'</div>'+
'  <div style="float:left;" class="proj-submitted">'+(p.is_submitted?p.submitted:'&nbsp;')+'</div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="proj-con proj-hdn" id="proj-con-'+idx+'">'+
'  <div style="padding-bottom:20px;">'+
'    <div style="float:left;">'+
'      <div>'+
'        <table style="font-size:95%;"><tbody>'+
'          <tr><td></td></tr>'+
'          <tr>'+
'            <td><b>Export to:</b></td>'+
'            <td><a class="link" href="" target="_blank" title="Microsoft Excel 2008 File"><img src="img/EXCEL_icon.gif" /></a>'+
'                <a class="link" href="" target="_blank" title="Text File to be embeded into Confluence Wiki"><img src="img/WIKI_icon.png"/></a>'+
'                <a class="link" href="" target="_blank" title="Plain Text File"><img src="img/TEXT_icon.png" /></a></td>'+
'          </tr>'+
'          <tr><td></td></tr>'+
'          <tr><td></td></tr>'+
'          <tr>'+
'            <td><b>Sort by:</b></td>'+
'            <td>'+
'              <div id="proj-sort-'+idx+'">'+
'                <select name="sort">'+
'                  <option>status</option>'+
'                  <option>job #</option>'+
'                  <option>cable #</option>'+
'                  <option>system</option>'+
'                  <option>function</option>'+
'                  <option>cable type</option>'+
'                  <option>routing</option>'+
'                  <option>routing</option>'+
'                  <option>length</option>'+
'                  <option>destination</option>'+
'                </select>'+
'                <input type="checkbox" name="reverse" ></input>reverse'+
'              </td>'+
'            </div>'+
'          </tr>'+
'          <tr><td></td></tr>'+
'          <tr><td></td></tr>'+
'          <tr><td></td></tr>'+
'          <tr>'+
'            <td><b>Display:</b></td>'+
'            <td>'+
'              <div id="proj-displ-'+idx+'">'+
'                <input type="checkbox" name="status"   checked="checked"></input>status'+
'                <input type="checkbox" name="job"      checked="checked"></input>job #'+
'                <input type="checkbox" name="cable"    checked="checked"></input>cable #'+
'                <input type="checkbox" name="system"   checked="checked"></input>system'+
'                <input type="checkbox" name="function" checked="checked"></input>function'+
'                <input type="checkbox" name="length"   checked="checked"></input>length'+
'                <input type="checkbox" name="routing"  checked="checked"></input>routing'+
'                <input type="checkbox" name="ends"     checked="checked"></input>source & destination'+
'              </div>'+
'            </td>'+
'          </tr>'+
'        </tbody></table>'+
'      </div>'+
'    </div>'+
'    <div style="float:left; margin-left:20px; padding-left:10px;">'+
'      <div class="proj-editor-hdr" id="proj-editor-hdr-'+idx+'" onclick="projects.toggle_project_editor('+idx+');">'+
'        <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e" id="proj-editor-tgl-'+idx+'"></span></div>'+
'        <div style="float:left;"><b>Project Attributes</b></div>'+
'        <div style="clear:both;"></div>'+
'      </div>'+
'      <div class="proj-editor-con proj-editor-hdn" id="proj-editor-con-'+idx+'">'+
'        <table style="font-size:95%;"><tbody>'+
'          <tr>'+
'            <td><b>Owner: </b></td>'+
'            <td><select id="proj-change-owner-'+idx+'" class="proj-change-elem" style="padding:1px;">'+
'                  <option '+(p.owner=='gapon'  ?'selected="selected"':'')+'>gapon</option>'+
'                  <option '+(p.owner=='perazzo'?'selected="selected"':'')+'>perazzo</option>'+
'                </select></td>'+
'            <td><b>Title: </b></td>'+
'            <td><input type="text" size="36" id="proj-change-title-'+idx+'" class="proj-change-elem" value="'+p.title+'" style="padding:1px;"></td>'+
'            <td></td>'+
'            <td><b>Due by: </b></td>'+
'            <td><input type="text" size="6" id="proj-change-due-'+idx+'" class="proj-change-elem" value="" style="padding:1px;"></td>'+
'          </tr>'+
'          <tr>'+
'            <td><b>Descr: </b></td>'+
'            <td colspan="6"><textarea cols=60 rows=4 id="proj-change-descr-'+idx+'" class="proj-change-elem" style="padding:4px;" title="Here be the project description"></textarea></td>'+
'            <td></td>'+
'            <td align="right" valign="bottom"><button id="proj-change-'+idx+'-save">save</button></td>'+
'          </tr>'+
'        </tbody></table>'+
'      </div>'+
'      <div class="proj-alerts" id="proj-alerts-'+idx+'">'+
'      </div>'+
'    </div>'+
'    <div style="float:right;">'+
'      <button id="proj-add-'+idx+'" title="add new cable to the project. You will be temporarily redirected to the bew cable registration form.">add cable</button>'+
'      <button id="proj-delete-'+idx+'" title="delete the project and all associated cables from the database">delete project</button>'+
'      <button id="proj-submit-'+idx+'" title="submit all cables for production to allocate official job and cable numbers">submit project</button>'+
'    </div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div>'+
'    <table><tbody>'+
'      <tr id="proj-cables-hdr-'+idx+'">'+
'        <td nowrap="nowrap" class="table_hdr table_hdr_tight" >'+
'          <select name="status" onchange="projects.select_cables_by_status('+idx+')">'+
'            <option>- status -</option>'+
'            <option>Planned</option>'+
'            <option>Registered</option>'+
'            <option>Labeled</option>'+
'            <option>Fabrication</option>'+
'            <option>Ready</option>'+
'            <option>Installed</option>'+
'            <option>Commissioned</option>'+
'           <option>Damaged</option>'+
'            <option>Retired</option>'+
'          </select>'+
'        </td>'+
'        <td nowrap="nowrap" class="table_hdr">TOOLS</td>'+
'        <td nowrap="nowrap" class="table_hdr">job #</td>'+
'        <td nowrap="nowrap" class="table_hdr">cable #</td>'+
'        <td nowrap="nowrap" class="table_hdr">system</td>'+
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
'        <td nowrap="nowrap" class="table_hdr">pinlist</td>'+
'        <td nowrap="nowrap" class="table_hdr">station</td>'+
'        <td nowrap="nowrap" class="table_hdr">contype</td>'+
'        <td nowrap="nowrap" class="table_hdr">instr</td>'+
'      </tr>'+
'    </tbody></table>'+
'  </div>'+
'  <div id="proj-cables-load-'+idx+'" style="margin-top:5px; color:maroon;"></div>'+
'</div>';
		return html;
	};
	this.display = function() {
		var total = 0;
		var html = '';
		for( var idx in this.projects ) {
			var proj = this.projects[idx];
			proj.is_loaded = false;
			proj.num_edited = 0;
			html += this.project2html(idx);
			total++;
		}
		var info_html = '<b>'+total+'</b> project'+(total==1?'':'s');
		$('#projects-search-info').html(info_html);
		$('#projects-search-list').html(html);
		for( var idx in this.projects ) {
			$('#proj-displ-'+idx+' input').change( function() {
				alert('column reduction is not implemented');
			});
		}
	};
	this.load = function() {
		$('#projects-search-info').html('Searching...');
		var params = {};
		var jqXHR = $.get(
			'../portal/neocaptar_search_projects.php', params,
			function(data) {
				if( data.status != 'success' ) {
					alert( data.message );
					return;
				}
				that.projects = data.projects;
				that.display();
			},
			'JSON'
		).error(
			function () {
				alert('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
			}
		);
	};
	return this;
}
var projects = new p_appl_projects();
