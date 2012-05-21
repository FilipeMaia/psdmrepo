function p_appl_dictionary() {

    var that = this;

    this.when_done = null;

    /* -------------------------------------------------------------------------
     *   Data structures and methods to be used/called by external users
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
     * -------------------------------------------------------------------------
     */
	this.name      = 'dictionary';
	this.full_name = 'Dictionary';
	this.context   = '';
    this.default_context = 'types';

    this.select = function(context) {
		that.context = context;
		this.init();
	};
	this.select_default = function() {
		this.init();
		if( this.context == '' ) this.context = this.default_context;
	};
	this.if_ready2giveup = function(handler2call) {
		this.init();
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
		this.init_cables();
		this.init_locations();
        this.init_routings();
        this.init_devices();
        this.init_instrs();
	};
    this.can_manage = function() {
        return global_current_user.is_administrator;
    };
    this.update = function(cable) {
		this.save_cable    (cable.cable_type);
        this.save_connector(cable.cable_type, cable.origin.conntype);
        this.save_connector(cable.cable_type, cable.destination.conntype);
		this.save_pinlist  (cable.cable_type, cable.origin.conntype,      cable.origin.pinlist, '');
		this.save_pinlist  (cable.cable_type, cable.destination.conntype, cable.destination.pinlist, '');

        this.save_location (cable.origin.loc);
        this.save_location (cable.destination.loc);
        this.save_rack     (cable.origin.loc,      cable.origin.rack);
		this.save_rack     (cable.destination.loc, cable.destination.rack);

        this.save_routing  (cable.routing);

        this.save_instr    (cable.origin.instr);
		this.save_instr    (cable.destination.instr);
    };

    // -----------------------------
    // CABLES, CONNECTORS & PINLISTS
    // -----------------------------

	this.selected_cable = null;
	this.selected_connector = null;
	this.cable = {};
	this.get_cable = function() {
		this.init();
		return this.cable;
	};
	this.cable_dict_is_empty = function() {
		for( var cable in this.cables()) return false;
		return true;
	}
	this.cable_is_not_known = function(cable) {
		return this.cable_dict_is_empty() || ( cable == null ) || ( typeof this.cables()[cable] === 'undefined' );
	};
	this.connector_dict_is_empty = function(cable) {
		for( var connector in this.connectors(cable)) return false;
		return true;
	};
	this.connector_is_not_known = function(cable,connector) {
		return this.cable_is_not_known(cable) || ( connector == null ) || ( typeof this.connectors(cable)[connector] === 'undefined' );
	};
	this.pinlist_dict_is_empty = function(cable,connector) {
		for( var pinlist in this.pinlists(cable, connector)) return false;
		return true;
	};
	this.pinlist_is_not_known = function(cable,connector,pinlist) {
		return this.connector_is_not_known(cable,connector) || ( pinlist == null ) || ( typeof this.pinlists(cable,connector)[pinlist] === 'undefined' );
	};
	this.cables = function() {
		return this.get_cable();
	};
	this.connectors = function(cable) {
		if( this.cable_is_not_known(cable)) return {};
		return this.cables()[cable]['connector'];
	};
	this.pinlists = function(cable,connector) {
		if( this.connector_is_not_known(cable,connector)) return {};
		return this.connectors(cable)[connector]['pinlist'];
	};
	this.init_cables = function() {
		$('#dictionary-types').find('input[name="cable2add"]').
            keyup(function(e) {
        		if( $(this).val() == '' ) { return; }
            	if( e.keyCode == 13     ) { that.new_cable(); return; }
                $(this).val(global_truncate_cable($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-types').find('input[name="connector2add"]').
            keyup(function(e) {
    			if( $(this).val() == '' ) { return; }
        		if( e.keyCode == 13     ) { that.new_connector(); return; }
                $(this).val(global_truncate_connector($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-types').find('input[name="pinlist2add"]').
    		keyup(function(e) {
        		if( $(this).val() == '' ) { return; }
            	if( e.keyCode == 13     ) {	that.new_pinlist(); return;	}
                $(this).val(global_truncate_pinlist($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-types-reload').
            button().
    		click(function() { that.load_cables(); });
		this.load_cables();
	};
	this.new_cable = function() {
		var input = $('#dictionary-types').find('input[name="cable2add"]');
		var cable_name = input.val();
		if( this.cable_is_not_known(cable_name)) {
			input.val('');
			this.selected_cable = cable_name;
			this.selected_connector = null;
			this.save_cable(this.selected_cable);
		}
	};
	this.new_connector = function() {
		var input = $('#dictionary-types').find('input[name="connector2add"]');
		var connector_name = input.val();
		if( this.connector_is_not_known(this.selected_cable, connector_name)) {
			input.val('');
			this.selected_connector = connector_name;
			this.save_connector(this.selected_cable, this.selected_connector);
		}
	};
	this.new_pinlist = function() {
		var input = $('#dictionary-types').find('input[name="pinlist2add"]');
		var pinlist_name = input.val();
		if( this.pinlist_is_not_known(this.selected_cable, this.selected_connector, pinlist_name)) {
			input.val('');
			this.save_pinlist(this.selected_cable, this.selected_connector, pinlist_name,'');
		}
	};
	this.save_cable = function(cable_name) {
		this.init();
		if( cable_name == '' ) return;
		if( this.cable_is_not_known(cable_name)) {
			this.cables()[cable_name] = {id:0, created_time:'', created_uid:'', connector:{}};
			if( this.selected_cable == null ) {
				this.selected_cable = cable_name;
				this.selected_connector = null;
			}
			$('#dictionary-types-info').html('Saving...');
			var params = {cable:cable_name};
			var jqXHR = $.get('../neocaptar/dict_cable_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				$('#dictionary-types-info').html('&nbsp;');
				that.cables()[cable_name] = result.cable[cable_name];
				that.display_cables();
			},
			'JSON').error(function () {
				$('#dictionary-types-info').html('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.save_connector = function(cable_name, connector_name) {
		this.init();
		if(( cable_name == '' ) || ( connector_name == '' )) return;
		if( this.connector_is_not_known(cable_name, connector_name)) {
			this.connectors(cable_name)[connector_name] = {id:0, created_time:'', created_uid:'', pinlist:{}};
			if( this.selected_cable == null ) {
				this.selected_cable = cable_name;
				this.selected_connector = connector_name;
			} else {
				if(( this.selected_cable == cable_name ) && ( this.selected_connector == null ))  {
					this.selected_connector = connector_name;
				}
			}
			$('#dictionary-types-info').html('Saving...');
			var params = {cable:cable_name, connector:connector_name};
			var jqXHR = $.get('../neocaptar/dict_cable_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				$('#dictionary-types-info').html('&nbsp;');
				that.cables()[cable_name] = result.cable[cable_name];
				that.display_cables();
			},
			'JSON').error(function () {
				$('#dictionary-types-info').html('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.save_pinlist = function(cable_name, connector_name, pinlist_name, pinlist_documentation) {
		this.init();
		if(( cable_name == '' ) || ( connector_name == '' ) || ( pinlist_name == '' )) return;
		if( this.pinlist_is_not_known(cable_name, connector_name, pinlist_name)) {
			this.pinlists(cable_name, connector_name)[pinlist_name] = {id:0, created_time:'', created_uid:''};
			if( this.selected_cable == null ) {
				this.selected_cable = cable_name;
				this.selected_connector = connector_name;
			} else {
				if(( this.selected_cable == cable_name ) && ( this.selected_connector == null ))  {
					this.selected_connector = connector_name;
				}
			}
			$('#dictionary-types-info').html('Saving...');
			var params = {cable:cable_name, connector:connector_name, pinlist:pinlist_name, pinlist_documentation:pinlist_documentation};
			var jqXHR = $.get('../neocaptar/dict_cable_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				$('#dictionary-types-info').html('&nbsp;');
				that.cables()[cable_name] = result.cable[cable_name];
				that.display_cables();
			},
			'JSON').error(function () {
				$('#dictionary-types-info').html('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.update_pinlist = function(cable_name, connector_name, pinlist_name, pinlist_documentation) {
		this.init();
        var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
		$('#dictionary-types-info').html('Updating...');
		var params = {id:pinlist.id, documentation:pinlist_documentation};
		var jqXHR = $.get('../neocaptar/dict_pinlist_update.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			$('#dictionary-types-info').html('&nbsp;');
			that.cables()[cable_name] = result.cable[cable_name];
			that.display_cables();
		},
		'JSON').error(function () {
			$('#dictionary-types-info').html('update failed because of: '+jqXHR.statusText);
		});
	};
	this.delete_element = function(element,id) {
		this.init();
		$('#dictionary-types-info').html('Deleting '+element+'...');
		var params = {scope:element, id:id};
		var jqXHR = $.get('../neocaptar/dict_cable_delete.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			$('#dictionary-types-info').html('&nbsp;');
			that.load_cables();
		},
		'JSON').error(function () {
			$('#dictionary-types-info').html('deletion failed because of: '+jqXHR.statusText);
		});
	};
	this.load_cables = function() {
		var params = {};
		var jqXHR = $.get('../neocaptar/dict_cable_get.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			$('#dictionary-types-info').html('&nbsp;');
			that.cable = result.cable;
			if( that.cable_is_not_known(that.selected_cable)) {
				that.selected_cable     = null;
				that.selected_connector = null;
			} else if( that.connector_is_not_known(that.selected_cable, that.selected_connector)) {
				that.selected_connector = null;
			}
			that.display_cables();
		},
		'JSON').error(function () {
			$('#dictionary-types-info').html('loading failed because of: '+jqXHR.statusText);
		});
	};
    this.cable2html = function(cable_name,is_selected_cable) {
        var cable = this.cables()[cable_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-cable-delete" name="'+cable.id+'" title="delete this cable from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable '+(is_selected_cable?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_cable('+"'"+cable_name+"'"+')" >'+cable_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+cable.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+cable.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-cable-search" name="'+cable.id+'" title="search all uses of this cable type">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
    this.connector2html = function(cable_name,connector_name,is_selected_connector) {
        var connector = this.connectors(cable_name)[connector_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-connector-delete" name="'+connector.id+'" title="delete this connector from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable '+(is_selected_connector?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_connector('+"'"+connector_name+"'"+')" >'+connector_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+connector.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+connector.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-connector-search" name="'+connector.id+'" title="search all uses of this connector type">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
    this.pinlist2url = function(cable,connector,pinlist_name) {
        if(this.pinlist_is_not_known(cable,connector,pinlist_name)) return pinlist_name;
        var pinlist_documentation = this.pinlists(cable,connector)[pinlist_name].documentation;
        var html = '<a href="'+pinlist_documentation+'" target="_blank" title="click the link to get the external documentation">'+pinlist_name+'</a>';
        return html;
    };

    this.pinlist2html = function(cable_name,connector_name,pinlist_name) {
        var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
        var params = "'"+cable_name+"','"+connector_name+"','"+pinlist_name+"'";
        var html = 
'<tr id="dict-pinlist-url-'+pinlist.id+'" >'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-pinlist-delete" name="'+pinlist.id+'" title="delete this pinlist from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell "><a href="'+pinlist.documentation+'" target="_blank" title="click the link to get the external documentation">'+pinlist_name+'</a></td>'+
'  <td nowrap="nowrap" class="table_cell " >'+
'    <input type="text" name="url" style="display:none;"></input>'+
'    <button class="dict-table-pinlist-edit"   onclick="dict.edit_pinlist_url('+params+')"        title="edit URL for the pinlist">edit</button>'+
'    <button class="dict-table-pinlist-save"   onclick="dict.edit_pinlist_url_save('+params+')"   title="edit URL for the pinlist">save</button>'+
'    <button class="dict-table-pinlist-cancel" onclick="dict.edit_pinlist_url_cancel('+params+')" title="edit URL for the pinlist">cancel</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell " >'+pinlist.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+pinlist.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-pinlist-search" name="'+pinlist.id+'" title="search all uses of this pinlist">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
	this.display_cables = function() {
		var html_cables =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >cable type</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >creator</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		var html_connectors = 
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >connector type</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >creator</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		var html_pinlists = 
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >pin list type</td>'+
'    <td nowrap="nowrap" class="table_hdr " >documentation (URL)</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >creator</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		for( var cable_name in this.cables()) {
			if( this.selected_cable == null ) {
				this.selected_cable = cable_name;
				this.selected_connector = null;
			}
			var is_selected_cable = this.selected_cable === cable_name;
			html_cables += this.cable2html(cable_name,is_selected_cable);
			if( is_selected_cable ) {
				for( var connector_name in this.connectors(cable_name)) {
					if( this.selected_connector == null ) {
						this.selected_connector = connector_name;
					}
					var is_selected_connector = this.selected_connector === connector_name;
					html_connectors += this.connector2html(cable_name,connector_name,is_selected_connector);
					if( is_selected_connector ) {
						for( var pinlist_name in this.pinlists(cable_name, connector_name)) {
							html_pinlists += this.pinlist2html(cable_name,connector_name,pinlist_name);
						}
					}
				}
			}
		}
        html_cables +=
'</tbody></table>';
        html_connectors +=
'</tbody></table>';
        html_pinlists +=
'</tbody></table>';
        $('#dictionary-types-cables'    ).html(html_cables);
		$('#dictionary-types-connectors').html(html_connectors);
		$('#dictionary-types-pinlists'  ).html(html_pinlists);

		$('.dict-table-cable-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
                var cable_id = this.name;
                ask_yes_no(
                    'Data Deletion Warning',
                    'You are about to delete the cable type entry and all information associated with it. Are you sure?',
                    function() { that.delete_element('cable',cable_id); },
                    null
                );
            });
		$('.dict-table-cable-search').
            button().
            click(function() {
    			var cable_id = this.name;
                global_search_cables_by_dict_cable_id(cable_id);
            });
		$('.dict-table-connector-search').
            button().
            click(function() {
    			var connector_id = this.name;
                global_search_cables_by_dict_connector_id(connector_id);
            });
		$('.dict-table-pinlist-search').
            button().
            click(function() {
    			var pinlist_id = this.name;
                global_search_cables_by_dict_pinlist_id(pinlist_id);
            });
		$('.dict-table-connector-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var connector_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the connector type entry and all information associated with it. Are you sure?',
                    function() { that.delete_element('connector',connector_id); },
                    null
                );
            });
		$('.dict-table-pinlist-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var pinlist_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the pinlist type entry and all information associated with it. Are you sure?',
                    function() { that.delete_element('pinlist',pinlist_id); },
                    null
                );
    		});
		$('.dict-table-pinlist-edit').
            button().
            button(this.can_manage()?'enable':'disable');
		$('.dict-table-pinlist-save').
            button().
            button('disable');
		$('.dict-table-pinlist-cancel').
            button().
            button('disable');
        if(this.can_manage()) {
    		                                      $('#dictionary-types').find('input[name="cable2add"]'    ).removeAttr('disabled');
        	if( this.selected_cable     == null ) $('#dictionary-types').find('input[name="connector2add"]').attr      ('disabled','disabled');
            else                                  $('#dictionary-types').find('input[name="connector2add"]').removeAttr('disabled');
            if( this.selected_connector == null ) $('#dictionary-types').find('input[name="pinlist2add"]'  ).attr      ('disabled','disabled');
            else                                  $('#dictionary-types').find('input[name="pinlist2add"]'  ).removeAttr('disabled');
        }
	};
    this.edit_pinlist_url = function(cable_name,connector_name,pinlist_name) {
        var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
        var elem = $('#dict-pinlist-url-'+pinlist.id);
        elem.find('input[name="url"]').
            css('display','block').
            val(pinlist.documentation);
        elem.find('button.dict-table-pinlist-edit').button('disable');
        elem.find('button.dict-table-pinlist-save').button('enable');
        elem.find('button.dict-table-pinlist-cancel').button('enable');
    };
    this.edit_pinlist_url_save = function(cable_name,connector_name,pinlist_name) {
        var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
        var elem = $('#dict-pinlist-url-'+pinlist.id);
        var pinlist_documentation = elem.find('input[name="url"]').
            css('display','none').
            val();
        elem.find('button.dict-table-pinlist-edit').button('enable');
        elem.find('button.dict-table-pinlist-save').button('disable');
        elem.find('button.dict-table-pinlist-cancel').button('disable');
        this.update_pinlist(cable_name, connector_name, pinlist_name, pinlist_documentation);
    };
    this.edit_pinlist_url_cancel = function(cable_name,connector_name,pinlist_name) {
        var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
        var elem = $('#dict-pinlist-url-'+pinlist.id);
        elem.find('input[name="url"]').
            css('display','none');
        elem.find('button.dict-table-pinlist-edit').button('enable');
        elem.find('button.dict-table-pinlist-save').button('disable');
        elem.find('button.dict-table-pinlist-cancel').button('disable');
    };
	this.select_cable = function(cable) {
		if( this.selected_cable != cable ) {
			this.selected_cable     = cable;
			this.selected_connector = null;
			this.display_cables();
		}
	};
	this.select_connector = function(connector) {
		if( this.selected_connector != connector ) {
			this.selected_connector = connector;
			this.display_cables();
		}
	};

    // -----------------
    // LOCATIONS & RACKS
    // -----------------

	this.selected_location = null;
	this.location = {};
	this.get_location = function() {
		this.init();
		return this.location;
	};
	this.location_dict_is_empty = function() {
		for( var location in this.locations()) return false;
		return true;
	}
	this.location_is_not_known = function(location) {
		return this.location_dict_is_empty() || ( location == null ) || ( typeof this.locations()[location] === 'undefined' );
	};
	this.rack_dict_is_empty = function(location) {
		for( var rack in this.racks(location)) return false;
		return true;
	};
	this.rack_is_not_known = function(location,rack) {
		return this.location_is_not_known(location) || ( rack == null ) || ( typeof this.racks(location)[rack] === 'undefined' );
	};
	this.locations = function() {
		return this.get_location();
	};
	this.racks = function(location) {
		if( this.location_is_not_known(location)) return {};
		return this.locations()[location]['rack'];
	};
	this.init_locations = function() {
		$('#dictionary-locations').find('input[name="location2add"]').
    		keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_location(); return; }
                $(this).val(global_truncate_location($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-locations').find('input[name="rack2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_rack(); return; }
                $(this).val(global_truncate_rack($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-locations-reload').
            button().
            click(function() { that.load_locations(); });
		this.load_locations();
	};
	this.new_location = function() {
		var input = $('#dictionary-locations').find('input[name="location2add"]');
		var location_name = input.val();
		if( this.location_is_not_known(location_name)) {
			input.val('');
			this.selected_location = location_name;
			this.save_location(this.selected_location);
		}
	};
	this.new_rack = function() {
		var input = $('#dictionary-locations').find('input[name="rack2add"]');
		var rack_name = input.val();
		if( this.rack_is_not_known(this.selected_location, rack_name)) {
			input.val('');
			this.save_rack(this.selected_location, rack_name);
		}
	};
	this.save_location = function(location_name) {
		this.init();
		if( location_name == '' ) return;
		if( this.location_is_not_known(location_name)) {
			this.locations()[location_name] = {id:0, created_time:'', created_uid:'', rack:{}};
			if( this.selected_location == null ) {
				this.selected_location = location_name;
			}
			var params = {location:location_name};
			var jqXHR = $.get('../neocaptar/dict_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.locations()[location_name] = result.location[location_name];
				that.display_locations();
			},
			'JSON').error(function () {
				report_error('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.save_rack = function(location_name, rack_name) {
		this.init();
		if(( location_name == '' ) || ( rack_name == '' )) return;
		if( this.rack_is_not_known(location_name, rack_name)) {
			this.racks(location_name)[rack_name] = {id:0, created_time:'', created_uid:''};
			if( this.selected_location == null ) {
				this.selected_location = location_name;
			}
			var params = {location:location_name, rack:rack_name};
			var jqXHR = $.get('../neocaptar/dict_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.locations()[location_name] = result.location[location_name];
				that.display_locations();
			},
			'JSON').error(function () {
				report_error('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.delete_location_element = function(element,id) {
		this.init();
		var params = {scope:element, id:id};
		var jqXHR = $.get('../neocaptar/dict_location_delete.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.load_locations();
		},
		'JSON').error(function () {
			report_error('deletion failed because of: '+jqXHR.statusText);
		});
	};
	this.load_locations = function() {
		var params = {};
		var jqXHR = $.get('../neocaptar/dict_location_get.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.location = result.location;
			if( that.location_is_not_known(that.selected_location)) {
				that.selected_location = null;
			}
			that.display_locations();
		},
		'JSON').error(function () {
			report_error('loading failed because of: '+jqXHR.statusText);
		});
	};
    this.location2html = function(location_name,is_selected_location) {
        var location = this.locations()[location_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-location-delete" name="'+location.id+'" title="delete this location from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable '+(is_selected_location?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_location('+"'"+location_name+"'"+')" >'+location_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+location.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+location.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-location-search" name="'+location.id+'" title="search all uses of this location">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
    this.rack2html = function(location_name,rack_name) {
        var rack = this.racks(location_name)[rack_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-rack-delete" name="'+rack.id+'" title="delete this rack from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-nonselectable" >'+rack_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+rack.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+rack.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-rack-search" name="'+rack.id+'" title="search all uses of this rack">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
	this.display_locations = function() {
		var html_locations =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >locations</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >by</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
        var html_racks =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >racks</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >by</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		for( var location_name in this.locations()) {
			if( this.selected_location == null ) { this.selected_location = location_name; }
			var is_selected_location = this.selected_location === location_name;
			if( is_selected_location )
				for( var rack_name in this.racks(location_name)) html_racks += this.rack2html(location_name,rack_name);
			html_locations += this.location2html(location_name,is_selected_location);
		}
        html_locations +=
'</tbody></table>';
        html_racks +=
'</tbody></table>';
		$('#dictionary-locations-locations').html(html_locations);
		$('#dictionary-locations-racks'    ).html(html_racks);

		$('.dict-table-location-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var location_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the location entry and all information associated with it. Are you sure?',
                    function() { that.delete_location_element('location',location_id); },
                    null
                );
            });
		$('.dict-table-location-search').
            button().
            click(function() {
    			var location_id = this.name;
                global_search_cables_by_dict_location_id(location_id);
            });
		$('.dict-table-rack-search').
            button().
            click(function() {
    			var rack_id = this.name;
                global_search_cables_by_dict_rack_id(rack_id);
            });
		$('.dict-table-rack-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var rack_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the rack entry and all information associated with it. Are you sure?',
                    function() { that.delete_location_element('rack',rack_id); },
                    null
                );
            });
        if(this.can_manage()) {
                                                     $('#dictionary-locations').find('input[name="location2add"]').removeAttr('disabled');
    		if( this.selected_location     == null ) $('#dictionary-locations').find('input[name="rack2add"]'    ).attr      ('disabled','disabled');
            else                                     $('#dictionary-locations').find('input[name="rack2add"]'    ).removeAttr('disabled');
        }
	};
	this.select_location = function(location) {
		if( this.selected_location != location ) {
			this.selected_location = location;
			this.display_locations();
		}
	};

    // --------
    // ROUTINGS
    // --------

	this.routing = {};
	this.get_routing = function() {
		this.init();
		return this.routing;
	};
	this.routing_dict_is_empty = function() {
		for( var routing in this.routings()) return false;
		return true;
	}
	this.routing_is_not_known = function(routing) {
		return this.routing_dict_is_empty() || ( routing == null ) || ( typeof this.routings()[routing] === 'undefined' );
	};
	this.routings = function() {
		return this.get_routing();
	};
	this.init_routings = function() {
		$('#dictionary-routings').
            find('input[name="routing2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_routing(); return; }
                $(this).val(global_truncate_routing($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-routings-reload').
            button().
            click(function() { that.load_routings(); });
		this.load_routings();
	};
	this.new_routing = function() {
		var input = $('#dictionary-routings').find('input[name="routing2add"]');
		var routing_name = input.val();
		if( this.routing_is_not_known(routing_name)) {
			input.val('');
			this.save_routing(routing_name);
		}
	};
	this.save_routing = function(routing_name) {
		this.init();
		if( routing_name == '' ) return;
		if( this.routing_is_not_known(routing_name)) {
			this.routings()[routing_name] = {id:0, created_time:'', created_uid:''};
			var params = {routing:routing_name};
			var jqXHR = $.get('../neocaptar/dict_routing_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.routings()[routing_name] = result.routing[routing_name];
				that.display_routings();
			},
			'JSON').error(function () {
				report_error('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.delete_routing_element = function(element,id) {
		this.init();
		var params = {scope:element, id:id};
		var jqXHR = $.get('../neocaptar/dict_routing_delete.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.load_routings();
		},
		'JSON').error(function () {
			report_error('deletion failed because of: '+jqXHR.statusText);
		});
	};
	this.load_routings = function() {
		var params = {};
		var jqXHR = $.get('../neocaptar/dict_routing_get.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.routing = result.routing;
			that.display_routings();
		},
		'JSON').error(function () {
			report_error('loading failed because of: '+jqXHR.statusText);
		});
	};
    this.routing2html = function(routing_name) {
        var routing = this.routings()[routing_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-routing-delete" name="'+routing.id+'" title="delete this routing from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable onclick="dict.select_routing('+"'"+routing_name+"'"+')">'+routing_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+routing.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+routing.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-routing-search" name="'+routing.id+'" title="search all uses of this routing">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
	this.display_routings = function() {
        var html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >routings</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >by</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		for( var routing_name in this.routings()) html += this.routing2html(routing_name);
        html +=
'</tbody></table>';
		$('#dictionary-routings-routings').html(html);
		$('.dict-table-routing-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var routing_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the routing entry and all information associated with it. Are you sure?',
                    function() { that.delete_routing_element('routing',routing_id); },
                    null
                );
            });
		$('.dict-table-routing-search').
            button().
            click(function() {
    			var routing_id = this.name;
                global_search_cables_by_dict_routing_id(routing_id);
            });
        if(this.can_manage()) {
            $('#dictionary-routings').find('input[name="routing2add"]').removeAttr('disabled');
        }
	};


    // -------------------------------------------
    // DEVICE NAME: LOCATIONS, REGIONS, COMPONENTS
    // -------------------------------------------

	this.selected_device_location = null;
	this.selected_device_region = null;
	this.device_location = {};
	this.get_device_location = function() {
		this.init();
		return this.device_location;
	};
	this.device_location_dict_is_empty = function() {
		for( var device_location in this.device_locations()) return false;
		return true;
	}
	this.device_location_is_not_known = function(device_location) {
		return this.device_location_dict_is_empty() || ( device_location == null ) || ( typeof this.device_locations()[device_location] === 'undefined' );
	};
	this.device_region_dict_is_empty = function(device_location) {
		for( var device_region in this.device_regions(device_location)) return false;
		return true;
	};
	this.device_region_is_not_known = function(device_location,device_region) {
		return this.device_location_is_not_known(device_location) || ( device_region == null ) || ( typeof this.device_regions(device_location)[device_region] === 'undefined' );
	};
	this.device_component_dict_is_empty = function(device_location,device_region) {
		for( var device_component in this.device_components(device_location, device_region)) return false;
		return true;
	};
	this.device_component_is_not_known = function(device_location,device_region,device_component) {
		return this.device_region_is_not_known(device_location,device_region) || ( device_component == null ) || ( typeof this.device_components(device_location,device_region)[device_component] === 'undefined' );
	};
	this.device_locations = function() {
		return this.get_device_location();
	};
	this.device_regions = function(device_location) {
		if( this.device_location_is_not_known(device_location)) return {};
		return this.device_locations()[device_location]['region'];
	};
	this.device_components = function(device_location,device_region) {
		if( this.device_region_is_not_known(device_location,device_region)) return {};
		return this.device_regions(device_location)[device_region]['component'];
	};
	this.init_devices = function() {
		$('#dictionary-devices').find('input[name="device_location2add"]').
            keyup(function(e) {
        		if( $(this).val() == '' ) { return; }
            	if( e.keyCode == 13     ) { that.new_device_location(); return; }
                $(this).val(global_truncate_device_location($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-devices').find('input[name="device_region2add"]').
            keyup(function(e) {
    			if( $(this).val() == '' ) { return; }
        		if( e.keyCode == 13     ) { that.new_device_region(); return; }
                $(this).val(global_truncate_device_region($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-devices').find('input[name="device_component2add"]').
    		keyup(function(e) {
        		if( $(this).val() == '' ) { return; }
            	if( e.keyCode == 13     ) {	that.new_device_component(); return;	}
                $(this).val(global_truncate_device_component($(this).val()));
            }).
            attr('disabled','disabled');
		$('#dictionary-devices-reload').
            button().
    		click(function() { that.load_device_locations(); });
		this.load_device_locations();
	};
	this.new_device_location = function() {
		var input = $('#dictionary-devices').find('input[name="device_location2add"]');
		var location_name = input.val();
		if( this.device_location_is_not_known(location_name)) {
			input.val('');
			this.selected_device_location = location_name;
			this.selected_device_region = null;
			this.save_device_location(this.selected_device_location);
		}
	};
	this.new_device_region = function() {
		var input = $('#dictionary-devices').find('input[name="device_region2add"]');
		var region_name = input.val();
		if( this.device_region_is_not_known(this.selected_device_location, region_name)) {
			input.val('');
			this.selected_device_region = region_name;
			this.save_device_region(this.selected_device_location, this.selected_device_region);
		}
	};
	this.new_device_component = function() {
		var input = $('#dictionary-devices').find('input[name="device_component2add"]');
		var component_name = input.val();
		if( this.device_component_is_not_known(this.selected_device_location, this.selected_device_region, component_name)) {
			input.val('');
			this.save_device_component(this.selected_device_location, this.selected_device_region, component_name,'');
		}
	};
	this.save_device_location = function(location_name) {
		this.init();
		if( location_name == '' ) return;
		if( this.device_location_is_not_known(location_name)) {
			this.device_locations()[location_name] = {id:0, created_time:'', created_uid:'', device_region:{}};
			if( this.selected_device_location == null ) {
				this.selected_device_location = location_name;
				this.selected_device_region = null;
			}
			var params = {location:location_name};
			var jqXHR = $.get('../neocaptar/dict_device_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.device_locations()[location_name] = result.location[location_name];
				that.display_device_locations();
			},
			'JSON').error(function () {
                report_error('failed to contact the Web service to save device location in a dictionary of device names');
			});
		}
	};
	this.save_device_region = function(location_name, region_name) {
		this.init();
		if(( location_name == '' ) || ( region_name == '' )) return;
		if( this.device_region_is_not_known(location_name, region_name)) {
			this.device_regions(location_name)[region_name] = {id:0, created_time:'', created_uid:'', device_component:{}};
			if( this.selected_device_location == null ) {
				this.selected_device_location = location_name;
				this.selected_device_region = region_name;
			} else {
				if(( this.selected_device_location == location_name ) && ( this.selected_device_region == null ))  {
					this.selected_device_region = region_name;
				}
			}
			var params = {location:location_name, region:region_name};
			var jqXHR = $.get('../neocaptar/dict_device_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.device_locations()[location_name] = result.location[location_name];
				that.display_device_locations();
			},
			'JSON').error(function () {
                report_error('failed to contact the Web service to save device region in a dictionary of device names');
			});
		}
	};
	this.save_device_component = function(location_name, region_name, component_name) {
		this.init();
		if(( location_name == '' ) || ( region_name == '' ) || ( component_name == '' )) return;
		if( this.device_component_is_not_known(location_name, region_name, component_name)) {
			this.device_components(location_name, region_name)[component_name] = {id:0, created_time:'', created_uid:''};
			if( this.selected_device_location == null ) {
				this.selected_device_location = location_name;
				this.selected_device_region = region_name;
			} else {
				if(( this.selected_device_location == location_name ) && ( this.selected_device_region == null ))  {
					this.selected_device_region = region_name;
				}
			}
			var params = {location:location_name, region:region_name, component:component_name};
			var jqXHR = $.get('../neocaptar/dict_device_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.device_locations()[location_name] = result.location[location_name];
				that.display_device_locations();
			},
			'JSON').error(function () {
                report_error('failed to contact the Web service to save device region in a dictionary of device names');
			});
		}
	};
	this.delete_device_element = function(element,id) {
		this.init();
		var params = {scope:element, id:id};
		var jqXHR = $.get('../neocaptar/dict_device_location_delete.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.load_device_locations();
		},
		'JSON').error(function () {
			report_error('failed to contact the Web service to delete '+element+' from a dictionary of device names');
		});
	};
	this.load_device_locations = function() {
		var params = {};
		var jqXHR = $.get('../neocaptar/dict_device_location_get.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.device_location = result.location;
			if( that.device_location_is_not_known(that.selected_device_location)) {
				that.selected_device_location = null;
				that.selected_device_region   = null;
			} else if( that.device_region_is_not_known(that.selected_device_location, that.selected_device_region)) {
				that.selected_device_region = null;
			}
			that.display_device_locations();
		},
		'JSON').error(function () {
			report_error('failed to contact the Web service to load a dictionary of device names');
		});
	};
    this.device_location2html = function(location_name,is_selected_device_location) {
        var device_location = this.device_locations()[location_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-device-location-delete" name="'+device_location.id+'" title="delete this device_location from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable '+(is_selected_device_location?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_device_location('+"'"+location_name+"'"+')" >'+location_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+device_location.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+device_location.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-device-location-search" name="'+device_location.id+'" title="search all uses of this device_location type">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
    this.device_region2html = function(location_name,region_name,is_selected_device_region) {
        var device_region = this.device_regions(location_name)[region_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-device-region-delete" name="'+device_region.id+'" title="delete this device_region from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable '+(is_selected_device_region?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_device_region('+"'"+region_name+"'"+')" >'+region_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+device_region.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+device_region.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-device-region-search" name="'+device_region.id+'" title="search all uses of this device_region type">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
    this.device_component2url = function(device_location,device_region,component_name) {
        if(this.device_component_is_not_known(device_location,device_region,component_name)) return component_name;
        var device_component_documentation = this.device_components(device_location,device_region)[component_name].documentation;
        var html = '<a href="'+device_component_documentation+'" target="_blank" title="click the link to get the external documentation">'+component_name+'</a>';
        return html;
    };

    this.device_component2html = function(location_name,region_name,component_name) {
        var device_component = this.device_components(location_name,region_name)[component_name];
        var params = "'"+location_name+"','"+region_name+"','"+component_name+"'";
        var html = 
'<tr id="dict-device-component-url-'+device_component.id+'" >'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-device-component-delete" name="'+device_component.id+'" title="delete this device_component from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell ">'+component_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+device_component.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+device_component.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-device-component-search" name="'+device_component.id+'" title="search all uses of this device_component">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
	this.display_device_locations = function() {
		var html_device_locations =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >device location</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >creator</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		var html_device_regions = 
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >device region</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >creator</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		var html_device_components = 
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >device component type</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >creator</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		for( var location_name in this.device_locations()) {
			if( this.selected_device_location == null ) {
				this.selected_device_location = location_name;
				this.selected_device_region = null;
			}
			var is_selected_device_location = this.selected_device_location === location_name;
			html_device_locations += this.device_location2html(location_name,is_selected_device_location);
			if( is_selected_device_location ) {
				for( var region_name in this.device_regions(location_name)) {
					if( this.selected_device_region == null ) {
						this.selected_device_region = region_name;
					}
					var is_selected_device_region = this.selected_device_region === region_name;
					html_device_regions += this.device_region2html(location_name,region_name,is_selected_device_region);
					if( is_selected_device_region ) {
						for( var component_name in this.device_components(location_name, region_name)) {
							html_device_components += this.device_component2html(location_name,region_name,component_name);
						}
					}
				}
			}
		}
        html_device_locations +=
'</tbody></table>';
        html_device_regions +=
'</tbody></table>';
        html_device_components +=
'</tbody></table>';
        $('#dictionary-devices-locations' ).html(html_device_locations);
		$('#dictionary-devices-regions'   ).html(html_device_regions);
		$('#dictionary-devices-components').html(html_device_components);

		$('.dict-table-device-location-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
                var device_location_id = this.name;
                ask_yes_no(
                    'Data Deletion Warning',
                    'You are about to delete the device_location type entry and all information associated with it. Are you sure?',
                    function() { that.delete_device_element('location',device_location_id); },
                    null
                );
            });
		$('.dict-table-device-location-search').
            button().
            click(function() {
    			var device_location_id = this.name;
                global_search_cables_by_dict_device_location_id(device_location_id);
            });
		$('.dict-table-device-region-search').
            button().
            click(function() {
    			var device_region_id = this.name;
                global_search_cables_by_dict_device_region_id(device_region_id);
            });
		$('.dict-table-device-component-search').
            button().
            click(function() {
    			var device_component_id = this.name;
                global_search_cables_by_dict_device_component_id(device_component_id);
            });
		$('.dict-table-device-region-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var device_region_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the device_region type entry and all information associated with it. Are you sure?',
                    function() { that.delete_device_element('region',device_region_id); },
                    null
                );
            });
		$('.dict-table-device-component-delete').
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var device_component_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the device_component type entry and all information associated with it. Are you sure?',
                    function() { that.delete_device_element('component',device_component_id); },
                    null
                );
    		});
		$('.dict-table-device-component-edit').
            button().
            button(this.can_manage()?'enable':'disable');
		$('.dict-table-device-component-save').
            button().
            button('disable');
		$('.dict-table-device-component-cancel').
            button().
            button('disable');
        if(this.can_manage()) {
    		                                            $('#dictionary-devices').find('input[name="device_location2add"]' ).removeAttr('disabled');
        	if( this.selected_device_location == null ) $('#dictionary-devices').find('input[name="device_region2add"]'   ).attr      ('disabled','disabled');
            else                                        $('#dictionary-devices').find('input[name="device_region2add"]'   ).removeAttr('disabled');
            if( this.selected_device_region   == null ) $('#dictionary-devices').find('input[name="device_component2add"]').attr      ('disabled','disabled');
            else                                        $('#dictionary-devices').find('input[name="device_component2add"]').removeAttr('disabled');
        }
	};
	this.select_device_location = function(device_location) {
		if( this.selected_device_location != device_location ) {
			this.selected_device_location = device_location;
			this.selected_device_region   = null;
			this.display_device_locations();
		}
	};
	this.select_device_region = function(device_region) {
		if( this.selected_device_region != device_region ) {
			this.selected_device_region = device_region;
			this.display_device_locations();
		}
	};







    // ------------
    // INSTRUCTIONS
    // ------------

	this.instr = {};
	this.get_instr = function() {
		this.init();
		return this.instr;
	};
	this.instr_dict_is_empty = function() {
		for( var instr in this.instrs()) return false;
		return true;
	}
	this.instr_is_not_known = function(instr) {
		return this.instr_dict_is_empty() || ( instr == null ) || ( typeof this.instrs()[instr] === 'undefined' );
	};
	this.instrs = function() {
		return this.get_instr();
	};
	this.init_instrs = function() {
		$('#dictionary-instrs').
            find('input[name="instr2add"]').
    		keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_instr(); return; }
                $(this).val(global_truncate_instr($(this).val()));
    		}).
            attr('disabled','disabled');
		$('#dictionary-instrs-reload').
            button().
            click(function() { that.load_instrs(); });
		this.load_instrs();
	};
	this.new_instr = function() {
		var input = $('#dictionary-instrs').find('input[name="instr2add"]');
		var instr_name = input.val();
		if( this.instr_is_not_known(instr_name)) {
			input.val('');
			this.save_instr(instr_name);
		}
	};
	this.save_instr = function(instr_name) {
		this.init();
		if( instr_name == '' ) return;
		if( this.instr_is_not_known(instr_name)) {
			this.instrs()[instr_name] = {id:0, created_time:'', created_uid:''};
			var params = {instr:instr_name};
			var jqXHR = $.get('../neocaptar/dict_instr_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				that.instrs()[instr_name] = result.instr[instr_name];
				that.display_instrs();
			},
			'JSON').error(function () {
				report_error('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.delete_instr_element = function(element,id) {
		this.init();
		var params = {scope:element, id:id};
		var jqXHR = $.get('../neocaptar/dict_instr_delete.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.load_instrs();
		},
		'JSON').error(function () {
			report_error('deletion failed because of: '+jqXHR.statusText);
		});
	};
	this.load_instrs = function() {
		var params = {};
		var jqXHR = $.get('../neocaptar/dict_instr_get.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			that.instr = result.instr;
			that.display_instrs();
		},
		'JSON').error(function () {
			report_error('loading failed because of: '+jqXHR.statusText);
		});
	};
    this.instr2html = function(instr_name) {
        var instr = this.instrs()[instr_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-instr-delete" name="'+instr.id+'" title="delete this instr from the dictionary">X</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable onclick="dict.select_instr('+"'"+instr_name+"'"+')">'+instr_name+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+instr.created_time+'</td>'+
'  <td nowrap="nowrap" class="table_cell " >'+instr.created_uid+'</td>'+
'  <td nowrap="nowrap" class="table_cell table_cell_right " >'+
'    <button class="dict-table-instr-search" name="'+instr.id+'" title="search all uses of this instr">search</button>'+
'  </td>'+
'</tr>';
        return html;
    };
	this.display_instrs = function() {
        var html =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >instruction</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >by</td>'+
'    <td nowrap="nowrap" class="table_hdr " >usage</td>'+
'  </tr>';
		for(var instr_name in this.instrs()) html += this.instr2html(instr_name);
        html +=
'</tbody></table>';
		$('#dictionary-instrs-instrs').html(html);
		$('.dict-table-instr-delete' ).
            button().
            button(this.can_manage()?'enable':'disable').
            click(function() {
    			var instr_id = this.name;
        		ask_yes_no(
            		'Data Deletion Warning',
                	'You are about to delete the instr entry and all information associated with it. Are you sure?',
                    function() { that.delete_instr_element('instr',instr_id); },
                    null
                );
		});
		$('.dict-table-instr-search').
            button().
            click(function() {
    			var instr_id = this.name;
                global_search_cables_by_dict_instr_id(instr_id);
            });
        if(this.can_manage()) {
            $('#dictionary-instrs').find('input[name="instr2add"]').removeAttr('disabled');
        }
	};




	return this;
}
var dict = new p_appl_dictionary();
