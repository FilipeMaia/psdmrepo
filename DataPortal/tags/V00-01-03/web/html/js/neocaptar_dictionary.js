function p_appl_dictionary() {
	var that = this;
	this.name = 'dictionary';
	this.full_name = 'Dictionary';
	this.context = '';
	this.select = function(ctx) {
		this.init();
		that.context = ctx;
	};
	this.select_default = function() {
		this.init();
		if( this.context == '' ) this.context = 'types';
	};
	this.if_ready2giveup = function(handler2call) {
		this.init();
		handler2call();
	};
	this.initialized = false;
	this.init = function() {
		if( this.initialized ) return;
		this.initialized = true;
		this.init_cables();
		this.init_locations();
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
		$('#dictionary-types').find('input[name="cable2add"]')
		.keyup(function(e) {
			if( $(this).val() == '' ) { return; }
			if( e.keyCode == 13     ) { that.new_cable(); return; }
		})
		.attr('disabled','disabled');

		$('#dictionary-types').find('input[name="connector2add"]')
		.keyup(function(e) {
			if( $(this).val() == '' ) { return; }
			if( e.keyCode == 13     ) { that.new_connector(); return; }
		})
		.attr('disabled','disabled');

		$('#dictionary-types').find('input[name="pinlist2add"]')
		.keyup(function(e) {
			if( $(this).val() == '' ) { return; }
			if( e.keyCode == 13     ) {	that.new_pinlist(); return;	}
		})
		.attr('disabled','disabled');

		$('#dictionary-types-reload').button()
		.click(function() { that.load_cables(); });

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
	this.display_cables = function() {
		var html_cables = '';
		var html_connectors = '';
		var html_pinlists = '';
		for( var cable_name in this.cables()) {
			var cable = this.cables()[cable_name];
			if( this.selected_cable == null ) {
				this.selected_cable = cable_name;
				this.selected_connector = null;
			}
			var is_selected_cable = this.selected_cable === cable_name;
			html_cables +=
'<div class="dict-table-entry">'+
'  <div style="float:left; width:40px;  " class="dict-table-entry-control " ><button class="dict-table-cable-delete" name="'+cable.id+'" title="delete this cable from the dictionary">X</button></div>'+
'  <div style="float:left; width:110px; " class="dict-table-entry-selectable '+(is_selected_cable?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_cable('+"'"+cable_name+"'"+') " '+
' >&nbsp;&nbsp;'+cable_name+'&nbsp;&nbsp;</div>'+
'  <div style="float:left; width:120px; " class="dict-table-entry-attr " >'+cable.created_time+'</div>'+
'  <div style="float:left; width:80px;  " class="dict-table-entry-attr " >'+cable.created_uid+'</div>'+
'  <div style="float:left; width:60px;  " class="dict-table-entry-attr " >[number]</div>'+
'  <div style="clear:both;              " ></div>'+
'</div>';

			if( is_selected_cable ) {
				for( var connector_name in this.connectors(cable_name)) {
					var connector = this.connectors(cable_name)[connector_name];
					if( this.selected_connector == null ) {
						this.selected_connector = connector_name;
					}
					var is_selected_connector = this.selected_connector === connector_name;
					html_connectors +=
'<div class="dict-table-entry">'+
'  <div style="float:left; width:40px;  " class="dict-table-entry-control " ><button class="dict-table-connector-delete" name="'+connector.id+'" title="delete this connector from the dictionary">X</button></div>'+
'  <div style="float:left; width:110px; " class="dict-table-entry-selectable '+(is_selected_connector?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_connector('+"'"+connector_name+"'"+')" >&nbsp;&nbsp;'+connector_name+'&nbsp;&nbsp;</div>'+
'  <div style="float:left; width:120px; " class="dict-table-entry-attr " >'+connector.created_time+'</div>'+
'  <div style="float:left; width:80px;  " class="dict-table-entry-attr " >'+connector.created_uid+'</div>'+
'  <div style="float:left; width:60px;  " class="dict-table-entry-attr " >[number]</div>'+
'  <div style="clear:both;              " ></div>'+
'</div>';
					if( is_selected_connector ) {
						for( var pinlist_name in this.pinlists(cable_name, connector_name)) {
							var pinlist = this.pinlists(cable_name,connector_name)[pinlist_name];
							html_pinlists +=
'<div class="dict-table-entry">'+
'  <div style="float:left; width:40px;  " class="dict-table-entry-control " ><button class="dict-table-pinlist-delete" name="'+pinlist.id+'" title="delete this pinlist from the dictionary">X</button></div>'+
'  <div style="float:left; width:110px; " class="dict-table-entry-nonselectable ">&nbsp;&nbsp;'+pinlist_name+'&nbsp;&nbsp;</div>'+
'  <div style="float:left; width:120px; " class="dict-table-entry-attr " >'+pinlist.created_time+'</div>'+
'  <div style="float:left; width:80px;  " class="dict-table-entry-attr " >'+pinlist.created_uid+'</div>'+
'  <div style="float:left; width:60px;  " class="dict-table-entry-attr " >[number]</div>'+
'  <div style="clear:both;              " ></div>'+
'</div>';
						}
					}
				}
			}
		}
		$('#dictionary-types-cables'    ).html(html_cables);
		$('#dictionary-types-connectors').html(html_connectors);
		$('#dictionary-types-pinlists'  ).html(html_pinlists);

		$('.dict-table-cable-delete'    ).button().click(function() {
			var cable_id = this.name;
			ask_yes_no(
				'Data Deletion Warning',
				'You are about to delete the cable type entry and all information associated with it. Are you sure?',
				function() { that.delete_element('cable',cable_id); },
				null
			);
		});
		$('.dict-table-connector-delete').button().click(function() {
			var connector_id = this.name;
			ask_yes_no(
				'Data Deletion Warning',
				'You are about to delete the connector type entry and all information associated with it. Are you sure?',
				function() { that.delete_element('connector',connector_id); },
				null
			);
		});
		$('.dict-table-pinlist-delete').button().click(function() {
			var pinlist_id = this.name;
			ask_yes_no(
				'Data Deletion Warning',
				'You are about to delete the pinlist type entry and all information associated with it. Are you sure?',
				function() { that.delete_element('pinlist',pinlist_id); },
				null
			);
		});

		                                      $('#dictionary-types').find('input[name="cable2add"]'    ).removeAttr('disabled');
		if( this.selected_cable     == null ) $('#dictionary-types').find('input[name="connector2add"]').attr      ('disabled','disabled');
		else                                  $('#dictionary-types').find('input[name="connector2add"]').removeAttr('disabled');
		if( this.selected_connector == null ) $('#dictionary-types').find('input[name="pinlist2add"]'  ).attr      ('disabled','disabled');
		else                                  $('#dictionary-types').find('input[name="pinlist2add"]'  ).removeAttr('disabled');
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
		$('#dictionary-locations').find('input[name="location2add"]')
		.keyup(function(e) {
			if( $(this).val() == '' ) { return; }
			if( e.keyCode == 13     ) { that.new_location(); return; }
		})
		.attr('disabled','disabled');

		$('#dictionary-locations').find('input[name="rack2add"]')
		.keyup(function(e) {
			if( $(this).val() == '' ) { return; }
			if( e.keyCode == 13     ) { that.new_rack(); return; }
		})
		.attr('disabled','disabled');

		$('#dictionary-locations-reload').button()
		.click(function() { that.load_locations(); });

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
			$('#dictionary-locations-info').html('Saving...');
			var params = {location:location_name};
			var jqXHR = $.get('../portal/neocaptar_dict_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				$('#dictionary-locations-info').html('&nbsp;');
				that.locations()[location_name] = result.location[location_name];
				that.display_locations();
			},
			'JSON').error(function () {
				$('#dictionary-locations-info').html('saving failed because of: '+jqXHR.statusText);
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
			$('#dictionary-locations-info').html('Saving...');
			var params = {location:location_name, rack:rack_name};
			var jqXHR = $.get('../portal/neocaptar_dict_location_new.php',params,function(data) {
				var result = eval(data);
				if(result.status != 'success') { report_error(result.message, null); return; }
				$('#dictionary-locations-info').html('&nbsp;');
				that.locations()[location_name] = result.location[location_name];
				that.display_locations();
			},
			'JSON').error(function () {
				$('#dictionary-locations-info').html('saving failed because of: '+jqXHR.statusText);
			});
		}
	};
	this.delete_location_element = function(element,id) {
		this.init();
		$('#dictionary-locations-info').html('Deleting '+element+'...');
		var params = {scope:element, id:id};
		var jqXHR = $.get('../portal/neocaptar_dict_location_delete.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			$('#dictionary-locations-info').html('&nbsp;');
			that.load_locations();
		},
		'JSON').error(function () {
			$('#dictionary-locations-info').html('deletion failed because of: '+jqXHR.statusText);
		});
	};
	this.load_locations = function() {
		var params = {};
		var jqXHR = $.get('../portal/neocaptar_dict_location_get.php',params,function(data) {
			var result = eval(data);
			if(result.status != 'success') { report_error(result.message, null); return; }
			$('#dictionary-locations-info').html('&nbsp;');
			that.location = result.location;
			if( that.location_is_not_known(that.selected_location)) {
				that.selected_location = null;
			}
			that.display_locations();
		},
		'JSON').error(function () {
			$('#dictionary-locations-info').html('loading failed because of: '+jqXHR.statusText);
		});
	};
	this.display_locations = function() {
		var html_locations = '';
		var html_racks = '';
		for( var location_name in this.locations()) {
			var location = this.locations()[location_name];
			if( this.selected_location == null ) {
				this.selected_location = location_name;
			}
			var is_selected_location = this.selected_location === location_name;
			html_locations +=
'<div class="dict-table-entry">'+
'  <div style="float:left; width:40px;  " class="dict-table-entry-control " ><button class="dict-table-location-delete" name="'+location.id+'" title="delete this location from the dictionary">X</button></div>'+
'  <div style="float:left; width:110px; " class="dict-table-entry-selectable '+(is_selected_location?'dict-table-entry-selectable-selected ':'')+'" onclick="dict.select_location('+"'"+location_name+"'"+') " '+
' >&nbsp;&nbsp;'+location_name+'&nbsp;&nbsp;</div>'+
'  <div style="float:left; width:120px; " class="dict-table-entry-attr " >'+location.created_time+'</div>'+
'  <div style="float:left; width:80px;  " class="dict-table-entry-attr " >'+location.created_uid+'</div>'+
'  <div style="float:left; width:60px;  " class="dict-table-entry-attr " >[number]</div>'+
'  <div style="clear:both;              " ></div>'+
'</div>';

			if( is_selected_location ) {
				for( var rack_name in this.racks(location_name)) {
					var rack = this.racks(location_name)[rack_name];
					html_racks +=
'<div class="dict-table-entry">'+
'  <div style="float:left; width:40px;  " class="dict-table-entry-control " ><button class="dict-table-rack-delete" name="'+rack.id+'" title="delete this rack from the dictionary">X</button></div>'+
'  <div style="float:left; width:110px; " class="dict-table-entry-nonselectable " >&nbsp;&nbsp;'+rack_name+'&nbsp;&nbsp;</div>'+
'  <div style="float:left; width:120px; " class="dict-table-entry-attr " >'+rack.created_time+'</div>'+
'  <div style="float:left; width:80px;  " class="dict-table-entry-attr " >'+rack.created_uid+'</div>'+
'  <div style="float:left; width:60px;  " class="dict-table-entry-attr " >[number]</div>'+
'  <div style="clear:both;              " ></div>'+
'</div>';
				}
			}
		}
		$('#dictionary-locations-locations').html(html_locations);
		$('#dictionary-locations-racks'    ).html(html_racks);

		$('.dict-table-location-delete'    ).button().click(function() {
			var location_id = this.name;
			ask_yes_no(
				'Data Deletion Warning',
				'You are about to delete the location entry and all information associated with it. Are you sure?',
				function() { that.delete_location_element('location',location_id); },
				null
			);
		});
		$('.dict-table-rack-delete').button().click(function() {
			var rack_id = this.name;
			ask_yes_no(
				'Data Deletion Warning',
				'You are about to delete the rack entry and all information associated with it. Are you sure?',
				function() { that.delete_location_element('rack',rack_id); },
				null
			);
		});
		                                         $('#dictionary-locations').find('input[name="location2add"]').removeAttr('disabled');
		if( this.selected_location     == null ) $('#dictionary-locations').find('input[name="rack2add"]'    ).attr      ('disabled','disabled');
		else                                     $('#dictionary-locations').find('input[name="rack2add"]'    ).removeAttr('disabled');
	};
	this.select_location = function(location) {
		if( this.selected_location != location ) {
			this.selected_location = location;
			this.display_locations();
		}
	};
	return this;
}
var dict = new p_appl_dictionary();
