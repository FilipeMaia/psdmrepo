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

    this.select = function(context) {
		that.context = context;
		this.init();
	};
	this.select_default = function() {
		this.init();
		if( this.context == '' ) this.context = 'types';
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
        this.init_instrs();
	};
    this.can_manage = function() {
        return global_current_user.is_administrator;
    };
    this.update = function(cable) {
		this.save_cable    (cable.cable_type);
        this.save_connector(cable.cable_type, cable.origin.conntype);
        this.save_connector(cable.cable_type, cable.destination.conntype);
		this.save_pinlist  (cable.cable_type, cable.origin.conntype,      cable.origin.pinlist);
		this.save_pinlist  (cable.cable_type, cable.destination.conntype, cable.destination.pinlist);

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
            }).
            attr('disabled','disabled');
		$('#dictionary-types').find('input[name="connector2add"]').
            keyup(function(e) {
    			if( $(this).val() == '' ) { return; }
        		if( e.keyCode == 13     ) { that.new_connector(); return; }
            }).
            attr('disabled','disabled');
		$('#dictionary-types').find('input[name="pinlist2add"]').
    		keyup(function(e) {
        		if( $(this).val() == '' ) { return; }
            	if( e.keyCode == 13     ) {	that.new_pinlist(); return;	}
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
			this.save_pinlist(this.selected_cable, this.selected_connector, pinlist_name);
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
			var jqXHR = $.get('../portal/neocaptar_dict_cable_new.php',params,function(data) {
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
			var jqXHR = $.get('../portal/neocaptar_dict_cable_new.php',params,function(data) {
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
	this.save_pinlist = function(cable_name, connector_name, pinlist_name) {
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
			var params = {cable:cable_name, connector:connector_name, pinlist:pinlist_name};
			var jqXHR = $.get('../portal/neocaptar_dict_cable_new.php',params,function(data) {
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
	this.delete_element = function(element,id) {
		this.init();
		$('#dictionary-types-info').html('Deleting '+element+'...');
		var params = {scope:element, id:id};
		var jqXHR = $.get('../portal/neocaptar_dict_cable_delete.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_cable_get.php',params,function(data) {
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
'    <button class="dict-table-cable-delete" name="'+cable.id+'" title="delete this cable from the dictionary">delete</button>'+
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
'    <button class="dict-table-connector-delete" name="'+connector.id+'" title="delete this connector from the dictionary">delete</button>'+
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
    this.pinlist2html = function(cable_name,connector_name,pinlist_name) {
        var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
        var html =
'<tr>'+
'  <td nowrap="nowrap" class="table_cell table_cell_left " >'+
'    <button class="dict-table-pinlist-delete" name="'+pinlist.id+'" title="delete this pinlist from the dictionary">delete</button>'+
'  </td>'+
'  <td nowrap="nowrap" class="table_cell dict-table-entry-selectable">'+pinlist_name+'</td>'+
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
        if(this.can_manage()) {
    		                                      $('#dictionary-types').find('input[name="cable2add"]'    ).removeAttr('disabled');
        	if( this.selected_cable     == null ) $('#dictionary-types').find('input[name="connector2add"]').attr      ('disabled','disabled');
            else                                  $('#dictionary-types').find('input[name="connector2add"]').removeAttr('disabled');
            if( this.selected_connector == null ) $('#dictionary-types').find('input[name="pinlist2add"]'  ).attr      ('disabled','disabled');
            else                                  $('#dictionary-types').find('input[name="pinlist2add"]'  ).removeAttr('disabled');
        }
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
            }).
            attr('disabled','disabled');
		$('#dictionary-locations').find('input[name="rack2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_rack(); return; }
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
			var jqXHR = $.get('../portal/neocaptar_dict_location_new.php',params,function(data) {
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
			var jqXHR = $.get('../portal/neocaptar_dict_location_new.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_location_delete.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_location_get.php',params,function(data) {
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
'    <button class="dict-table-location-delete" name="'+location.id+'" title="delete this location from the dictionary">delete</button>'+
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
'    <button class="dict-table-rack-delete" name="'+rack.id+'" title="delete this rack from the dictionary">delete</button>'+
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
'    <td nowrap="nowrap" class="table_hdr " ># of uses</td>'+
'  </tr>';
        var html_racks =
'<table><tbody>'+
'  <tr>'+
'    <td nowrap="nowrap" class="table_hdr " ></td>'+
'    <td nowrap="nowrap" class="table_hdr " >racks</td>'+
'    <td nowrap="nowrap" class="table_hdr " >added</td>'+
'    <td nowrap="nowrap" class="table_hdr " >by</td>'+
'    <td nowrap="nowrap" class="table_hdr " ># of uses</td>'+
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
			var jqXHR = $.get('../portal/neocaptar_dict_routing_new.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_routing_delete.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_routing_get.php',params,function(data) {
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
'    <button class="dict-table-routing-delete" name="'+routing.id+'" title="delete this routing from the dictionary">delete</button>'+
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
'    <td nowrap="nowrap" class="table_hdr " ># of uses</td>'+
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
			var jqXHR = $.get('../portal/neocaptar_dict_instr_new.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_instr_delete.php',params,function(data) {
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
		var jqXHR = $.get('../portal/neocaptar_dict_instr_get.php',params,function(data) {
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
'    <button class="dict-table-instr-delete" name="'+instr.id+'" title="delete this instr from the dictionary">delete</button>'+
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
'    <td nowrap="nowrap" class="table_hdr " ># of uses</td>'+
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
