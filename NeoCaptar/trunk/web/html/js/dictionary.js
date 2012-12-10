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
        this.init_pinlists();
        this.init_locations();
        this.init_routings();
        this.init_devices();
        this.init_instrs();
    };
    this.can_manage = function() {
        return global_current_user.has_dict_priv;
    };
    this.update = function(cable) {
        this.save_cable    (cable.cable_type, '', cable.origin.conntype);
        this.save_cable    (cable.cable_type, '', cable.destination.conntype);

        // TODO: These two calls may not be needed because the previous two
        // are supposed to create new connectors (if needed) and establish
        // proper associatosn with the cable type.
        //
        this.save_connector(cable.origin.conntype, '',      cable.cable_type);
        this.save_connector(cable.destination.conntype, '', cable.cable_type);

        this.save_pinlist  (cable.origin.pinlist, '');
        this.save_pinlist  (cable.destination.pinlist, '');

        this.save_location (cable.origin.loc);
        this.save_location (cable.destination.loc);

        this.save_rack     (cable.origin.loc,      cable.origin.rack);
        this.save_rack     (cable.destination.loc, cable.destination.rack);

        this.save_routing  (cable.routing);

        this.save_instr    (cable.origin.instr);
        this.save_instr    (cable.destination.instr);

        this.save_device_location (cable.device_location);
        this.save_device_region   (cable.device_location, cable.device_region);
        this.save_device_component(cable.device_location, cable.device_region, cable.device_component);
    };
    this.web_service_GET = function(url, params, data_handler) {
        this.init();
        var jqXHR = $.get(url,params,function(data) {
            var result = eval(data);
            if(result.status != 'success') { report_error(result.message, null); return; }
            data_handler(result);
        },
        'JSON').error(function () {
            report_error('update failed because of: '+jqXHR.statusText);
        });
    };

    // ---------------------
    // CABLES and CONNECTORS
    // ---------------------

    this.type = {};

    this.cables = function() {
        this.init();
        return this.type.cable;
    };
    this.cable_dict_is_empty = function() {
        for( var cable in this.cables()) return false;
        return true;
    };
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
    this.connectors = function(cable) {
        if( this.cable_is_not_known(cable)) return {};
        return this.cables()[cable]['connector'];
    };

    this.connectors_reverse = function() {
        this.init();
        return this.type.connector;
    };
    this.connector_dict_is_empty_reverse = function() {
        for( var connector in this.connectors_reverse()) return false;
        return true;
    };
    this.connector_is_not_known_reverse = function(connector) {
        return this.connector_dict_is_empty_reverse() || ( connector == null ) || ( typeof this.connectors_reverse()[connector] === 'undefined' );
    };
    this.cables_reverse = function(connector) {
        if( this.connector_is_not_known_reverse(connector)) return {};
        return this.connectors_reverse()[connector]['cable'];
    };

    this.init_cables = function() {
        $('#dictionary-types').find('#tabs').tabs();
        $('#dictionary-types').find('#cables2connectors').find('input[name="cable2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_cable(); return; }
                $(this).val(global_truncate_cable($(this).val()));
            }).
            attr('disabled','disabled');
        $('#dictionary-types').find('#cables2connectors').find('input[name="connector2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_connector(); return; }
                $(this).val(global_truncate_connector($(this).val()));
            }).
            attr('disabled','disabled');
        $('#dictionary-types').find('#connectors2cables').find('input[name="connector2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_connector_reverse(); return; }
                $(this).val(global_truncate_connector($(this).val()));
            }).
            attr('disabled','disabled');
        $('#dictionary-types').find('#connectors2cables').find('input[name="cable2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_cable_reverse(); return; }
                $(this).val(global_truncate_cable($(this).val()));
            }).
            attr('disabled','disabled');
        $('#dictionary-types-reload').
            button().
            click(function() { that.load_types(); });
        this.load_types();
    };
    this.cables_select_tab = function(name) {
        var elem = $('#dictionary-types').find('#tabs');
        var selected = elem.tabs('option','selected');
        var required = name == 'cables2connectors' ? 0 : 1;
        if( required != selected )
            elem.tabs('select', required);
    };

    this.new_cable = function() {
        var input = $('#dictionary-types').find('#cables2connectors').find('input[name="cable2add"]');
        this.save_cable(input.val(),'','');
        input.val(''); };

    this.new_connector = function() {
        var input = $('#dictionary-types').find('#cables2connectors').find('input[name="connector2add"]');
        this.save_connector(input.val(), '', this.table_cables.selected_object());
        input.val(''); };

    this.new_connector_reverse = function() {
        var input = $('#dictionary-types').find('#connectors2cables').find('input[name="connector2add"]');
        this.save_connector(input.val(), '', '');
        input.val(''); };

    this.new_cable_reverse = function() {
        var input = $('#dictionary-types').find('#connectors2cables').find('input[name="cable2add"]');
        this.save_cable(input.val(), '', this.table_connectors_reverse.selected_object());
        input.val(''); };

    this.save_cable = function(cable_name, cable_documentation, connector_name) {
        if( cable_name == '' ) return;
        var params = { cable_name: cable_name, cable_documentation: cable_documentation };
        if((connector_name != null) && (connector_name != '')) params.connector_name = connector_name;
        this.type_action('../neocaptar/ws/dict_cable_new.php', params); };

    this.save_connector = function(connector_name, connector_documentation, cable_name) {
        if( connector_name == '' ) return;
        var params = { connector_name: connector_name, connector_documentation: connector_documentation };
        if((cable_name != null) && (cable_name != '')) params.cable_name = cable_name;
        this.type_action('../neocaptar/ws/dict_connector_new.php', params); };

    this.delete_cable = function(id) {
        this.type_action('../neocaptar/ws/dict_cable_delete.php', { id: id });    };

    this.delete_connector = function(id) {
        this.type_action('../neocaptar/ws/dict_connector_delete.php', { id: id }); };

    this.load_types = function() {
        this.type_action('../neocaptar/ws/dict_types_get.php', {}); };

    this.save_cable_documentation = function(id,documentation) {
        this.type_action('../neocaptar/ws/dict_cable_update.php', { id: id, documentation: documentation }); };

    this.save_connector_documentation = function(id,documentation) {
        this.type_action('../neocaptar/ws/dict_connector_update.php', { id: id, documentation: documentation }); };

    this.type_action = function(url, params, data_handler) {
        function handle_data_and_display(result) {
            if(data_handler) data_handler(result);
            else             that.type = result.type;
            that.display_types();
        }
        this.web_service_GET(url, params, handle_data_and_display);
    };


    this.show_cable_info = function(cable_name) {
        var cable = this.cables()[cable_name];
        report_info_table(
            'Cable Type Info',
            [ { name: 'name'  },
              { name: 'created'  },
              { name: 'by user' },
              { name: 'description' }],
            [ [ cable_name,
                cable.created_time,
                cable.created_uid,
                '<div style="width:420px; overflow:auto;"><pre>'+cable.documentation+'</pre></div>' ]]
        );
    };
    this.cable2url = function(name) {
        var html = this.cable_is_not_known(name) ?
            '<a href="javascript:report_error(\'no such cable found in the Dictionary\')">'+name+'</a>' :
            '<a href="javascript:dict.show_cable_info(\''+name+'\')" title="click the link to get the external documentation">'+name+'</a>';
        return html;
    };
    this.connector2url = function(name) {
        if(this.connector_is_not_known_reverse(name)) return name;
        var html = '<a href="'+this.connectors_reverse()[name].documentation+'" target="_blank" title="click the link to get the external documentation">'+name+'</a>';
        return html;
    };

    this.display_types = function() {
        this.display_cables();
        this.display_connectors_reverse();
    };


    this.table_cables     = null;
    this.table_connectors = null;

    this.display_cables = function(selected_cable_name) {

        var tab = 'cables2connectors';
        if( selected_cable_name !== undefined ) this.cables_select_tab(tab);

        var elem = $('#dictionary-types-cables');

        var hdr = [

            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-cable-delete').
                            button().
                            click(function() {
                                var id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the cable?',
                                    function() { that.delete_cable(id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'cable type', selectable: true,
                type: {
                    select_action : function(cable_name) {
                        that.display_connectors( cable_name );
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-cable-search').
                            button().
                            click(function() {
                                var id = this.name;
                                global_search_cables_by_dict_cable_id(id);
                            });
                    }
                }
            },

            {   name: 'description', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        for( var cable_name in that.cables()) {
                            var cable = that.cables()[cable_name];
                            elem.find('#cable-documentation-'+cable.id).val(cable.documentation);
                        }
                        elem.find('.cable-documentation-save').
                            button().
                            click(function() {
                                var id = this.name;
                                that.save_cable_documentation(id, elem.find('#cable-documentation-'+id).val());
                            });
                    }
                }
            }
        ];

        var rows = [];

        for( var cable_name in this.cables()) {
            var cable = this.cables()[cable_name];
            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    cable.id,
                            classes: 'dict-table-cable-delete',
                            title:   'delete this cable from the dictionary' }) : ' ',

                    cable_name,

                    cable.created_time,
                    cable.created_uid,

                    Button_HTML('search', {
                        name:    cable.id,
                        classes: 'dict-table-cable-search',
                        title:   'search all uses of this cable' }),

                    this.can_manage() ?
                        '<div style="float:left;">'+
                            TextArea_HTML({
                                id:      'cable-documentation-'+cable.id,
                                name:    cable_name,
                            classes: 'description' },
                            4,
                            36)+
                        '</div>'+
                        '<div style="float:left; margin-left:5px;">'+
                            Button_HTML('save', {
                                name:    cable.id,
                                classes: 'cable-documentation-save',
                                title:   'edit description for the cable' })+
                        '</div>'+
                        '<div style="clear:both;">' :
                        '<div style="width:256px; overflow:auto;"><pre>'+cable.documentation+'</pre></div>'
                ]
            );
        }
        this.table_cables = new Table(
            'dictionary-types-cables',  hdr, rows,
            { default_sort_column: 1, selected_col: 1 },
            config.handler('dict', 'table_cables')
        );
        this.table_cables.display();

        if( selected_cable_name !== undefined )
            this.table_cables.select(1, selected_cable_name);

        this.display_connectors( this.table_cables.selected_object());

        if(this.can_manage())
            $('#dictionary-types').find('#'+tab).find('input[name="cable2add"]' ).removeAttr('disabled');
    };

    this.display_connectors = function(cable_name) {

        var tab = 'cables2connectors';

        var elem = $('#dictionary-types-connectors');

        var hdr = [

            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-connector-delete').
                            button().
                            click(function() {
                                var id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the connector?',
                                    function() { that.delete_connector(id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'connector type', selectable: true,
                type: {
                    select_action : function(connector_name) {
                        that.display_connectors_reverse(connector_name);
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-connector-search').
                            button().
                            click(function() {
                                var id = this.name;
                                global_search_cables_by_dict_connector_id(id);
                            });
                    }
                }
            },

            {   name: 'description', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        for( var connector_name in that.connectors(cable_name)) {
                            var connector = that.connectors(cable_name)[connector_name];
                            elem.find('#connector-documentation-'+connector.id).val(connector.documentation);
                        }
                        elem.find('.connector-documentation-save').
                            button().
                            click(function() {
                                var id = this.name;
                                that.save_connector_documentation(id, elem.find('#connector-documentation-'+id).val());
                            });
                    }
                }
            }
        ];

        var rows = [];

        if( cable_name != null ) {
            for( var connector_name in this.connectors(cable_name)) {
                var connector = this.connectors(cable_name)[connector_name];
                rows.push(
                    [   this.can_manage() ?
                            Button_HTML('X', {
                                name:    connector.id,
                                classes: 'dict-table-connector-delete',
                                title:   'delete this connector from the dictionary' }) : ' ',

                        connector_name,

                        connector.created_time,
                        connector.created_uid,

                        Button_HTML('search', {
                            name:    connector.id,
                            classes: 'dict-table-connector-search',
                            title:   'search all uses of this connector' }),

                    this.can_manage() ?
                        '<div style="float:left;">'+
                            TextArea_HTML({
                                id:      'connector-documentation-'+connector.id,
                                name:    connector_name,
                            classes: 'description' },
                            4,
                            36)+
                        '</div>'+
                        '<div style="float:left; margin-left:5px;">'+
                            Button_HTML('save', {
                                name:    connector.id,
                                classes: 'connector-documentation-save',
                                title:   'edit description for the connector' })+
                        '</div>'+
                        '<div style="clear:both;">' :
                        '<div style="width:256px; overflow:auto;"><pre>'+connector.documentation+'</pre></div>'
                    ]
                );
            }
        }
        this.table_connectors = new Table(
            'dictionary-types-connectors', hdr, rows,
            { default_sort_column: 1, selected_col: 1 },
            config.handler('dict', 'table_connectors')
        );
        this.table_connectors.display();

        if(this.can_manage()) {
            var input = $('#dictionary-types').find('#'+tab).find('input[name="connector2add"]');
            if( cable_name == null ) input.attr('disabled','disabled');
            else                     input.removeAttr('disabled');
        }
    };

    this.table_connectors_reverse = null;
    this.table_cables_reverse     = null;

    this.display_connectors_reverse = function(selected_connector_name) {

        var tab = 'connectors2cables';
        if( selected_connector_name !== undefined ) this.cables_select_tab(tab);

        var elem = $('#dictionary-types-connectors-reverse');

        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-connector-delete').
                            button().
                            click(function() {
                                var id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the connector?',
                                    function() { that.delete_connector(id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'connector type', selectable: true,
                type: {
                    select_action : function(connector_name) {
                        that.display_cables_reverse( connector_name );
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-connector-search').
                            button().
                            click(function() {
                                var id = this.name;
                                global_search_cables_by_dict_connector_id(id);
                            });
                    }
                }
            },

            {   name: 'description', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        for( var connector_name in that.connectors_reverse()) {
                            var connector = that.connectors_reverse()[connector_name];
                            elem.find('#connector-documentation-'+connector.id).val(connector.documentation);
                        }
                        elem.find('.connector-documentation-save').
                            button().
                            click(function() {
                                var id = this.name;
                                that.save_connector_documentation(id, elem.find('#connector-documentation-'+id).val());
                            });
                    }
                }
            }
        ];

        var rows = [];

        for( var connector_name in this.connectors_reverse()) {
            var connector = this.connectors_reverse()[connector_name];
            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    connector.id,
                            classes: 'dict-table-connector-delete',
                            title:   'delete this connector from the dictionary' }) : ' ',

                    connector_name,

                    connector.created_time,
                    connector.created_uid,

                    Button_HTML('search', {
                        name:    connector.id,
                        classes: 'dict-table-connector-search',
                        title:   'search all uses of this connector' }),


                    this.can_manage() ?
                        '<div style="float:left;">'+
                            TextArea_HTML({
                                id:      'connector-documentation-'+connector.id,
                                name:    connector_name,
                            classes: 'description' },
                            4,
                            36)+
                        '</div>'+
                        '<div style="float:left; margin-left:5px;">'+
                            Button_HTML('save', {
                                name:    connector.id,
                                classes: 'connector-documentation-save',
                                title:   'edit description for the connector' })+
                        '</div>'+
                        '<div style="clear:both;">' :
                        '<div style="width:256px; overflow:auto;"><pre>'+connector.documentation+'</pre></div>'
                ]
            );
        }
        this.table_connectors_reverse = new Table(
            'dictionary-types-connectors-reverse', hdr, rows,
            { default_sort_column: 1, selected_col: 1},
            config.handler('dict', 'table_connectors_reverse')
        );
        this.table_connectors_reverse.display();

        if( selected_connector_name !== undefined )
            this.table_connectors_reverse.select(1,selected_connector_name);

        this.display_cables_reverse( this.table_connectors_reverse.selected_object());

        if(this.can_manage())
            $('#dictionary-types').find('#'+tab).find('input[name="connector2add"]' ).removeAttr('disabled');
    };

    this.display_cables_reverse = function(connector_name) {

        var tab = 'connectors2cables';

        var elem = $('#dictionary-types-cables-reverse');

        var hdr = [

            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-cable-delete').
                            button().
                            click(function() {
                                var id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the cable?',
                                    function() { that.delete_cable(id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'cable type', selectable: true,
                type: {
                    select_action : function(selected_cable_name) {
                        that.display_cables( selected_cable_name );
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-cable-search').
                            button().
                            click(function() {
                                var id = this.name;
                                global_search_cables_by_dict_cable_id(id);
                            });
                    }
                }
            },

            {   name: 'description', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        for( var cable_name in that.cables_reverse(connector_name)) {
                            var cable = that.cables_reverse(connector_name)[cable_name];
                            elem.find('#cable-documentation-'+cable.id).val(cable.documentation);
                        }
                        elem.find('.cable-documentation-save').
                            button().
                            click(function() {
                                var id = this.name;
                                that.save_cable_documentation(id, elem.find('#cable-documentation-'+id).val());
                            });
                    }
                }
            }
        ];

        var rows = [];

        if( connector_name != null ) {
            for( var cable_name in this.cables_reverse(connector_name)) {
                var cable = this.cables_reverse(connector_name)[cable_name];
                rows.push(
                    [   this.can_manage() ?
                            Button_HTML('X', {
                                name:    cable.id,
                                classes: 'dict-table-cable-delete',
                                title:   'delete this cable from the dictionary' }) : ' ',

                        cable_name,

                        cable.created_time,
                        cable.created_uid,

                        Button_HTML('search', {
                            name:    cable.id,
                            classes: 'dict-table-cable-search',
                            title:   'search all uses of this cable' }),


                    this.can_manage() ?
                        '<div style="float:left;">'+
                            TextArea_HTML({
                                id:      'cable-documentation-'+cable.id,
                                name:    cable_name,
                            classes: 'description' },
                            4,
                            36)+
                        '</div>'+
                        '<div style="float:left; margin-left:5px;">'+
                            Button_HTML('save', {
                                name:    cable.id,
                                classes: 'cable-documentation-save',
                                title:   'edit description for the cable' })+
                        '</div>'+
                        '<div style="clear:both;">' :
                        '<div style="width:256px; overflow:auto;"><pre>'+cable.documentation+'</pre></div>'
                    ]
                );
            }
        }
        this.table_cables_reverse = new Table(
            'dictionary-types-cables-reverse',
            hdr, rows,
            { default_sort_column: 1, selected_col: 1},
            config.handler('dict', 'table_cables_reverse')
        );
        this.table_cables_reverse.display();

        if(this.can_manage()) {
            var input = $('#dictionary-types').find('#'+tab).find('input[name="cable2add"]');
            if( connector_name == null ) input.attr('disabled','disabled');
            else                         input.removeAttr('disabled');
        }
    };

    // -------------------
    // PINLISTS (DRAWINGS)
    // -------------------

    this.pinlist = {};
    this.get_pinlist = function() {
        this.init();
        return this.pinlist;
    };
    this.pinlist_dict_is_empty = function() {
        for( var pinlist in this.pinlists()) return false;
        return true;
    }
    this.pinlist_is_not_known = function(pinlist) {
        return this.pinlist_dict_is_empty() || ( pinlist == null ) || ( typeof this.pinlists()[pinlist] === 'undefined' );
    };
    this.pinlists = function() {
        return this.get_pinlist();
    };
    this.init_pinlists = function() {
        $('#dictionary-pinlists').
            find('input[name="pinlist2add"]').
            keyup(function(e) {
                if( $(this).val() == '' ) { return; }
                if( e.keyCode == 13     ) { that.new_pinlist(); return; }
                $(this).val(global_truncate_pinlist($(this).val()));
            }).
            attr('disabled','disabled');
        $('#dictionary-pinlists-reload').
            button().
            click(function() { that.load_pinlists(); });
        this.load_pinlists();
    };
    this.new_pinlist = function() {
        var input = $('#dictionary-pinlists').find('input[name="pinlist2add"]');
        var pinlist_name = input.val();
        this.save_pinlist(pinlist_name,'');
        input.val('');
    };

    this.save_pinlist = function(name,documentation) {
        if( name == '' ) return;
        this.pinlist_action('../neocaptar/ws/dict_pinlist_new.php', { name: name, documentation: documentation },
            function(result) {that.pinlist[name] = result.pinlist[name]; }); };

    this.delete_pinlist = function(id) {
        this.pinlist_action('../neocaptar/ws/dict_pinlist_delete.php', { id: id }); };

    this.load_pinlists = function() {
        this.pinlist_action('../neocaptar/ws/dict_pinlist_get.php', {}); };

    this.save_pinlist_documentation = function(id,documentation) {
        this.pinlist_action('../neocaptar/ws/dict_pinlist_update.php', { id: id, documentation: documentation }); };

    this.save_pinlist_cable = function(id,cable) {
        this.pinlist_action('../neocaptar/ws/dict_pinlist_update.php', { id: id, cable: cable }); };

    this.save_pinlist_origin_connector = function(id,connector) {
        this.pinlist_action('../neocaptar/ws/dict_pinlist_update.php', { id: id, origin_connector: connector }); };

    this.save_pinlist_destination_connector = function(id,connector) {
        this.pinlist_action('../neocaptar/ws/dict_pinlist_update.php', { id: id, destination_connector: connector }); };

    this.pinlist_action = function(url, params, data_handler) {
        function handle_data_and_display(result) {
            if(data_handler) data_handler(result);
            else             that.pinlist = result.pinlist;
            that.display_pinlists();
        }
        this.web_service_GET(url, params, handle_data_and_display);
    };

    this.pinlist2url = function(name) {
        if(this.pinlist_is_not_known(name)) return name;
        var html = '<a href="'+this.pinlists()[name].documentation+'" target="_blank" title="click the link to get the external documentation">'+name+'</a>';
        return html;
    };

    this.pinlist_table = null;

    this.display_pinlists = function() {
        var elem_pinlists = $('#dictionary-pinlists-pinlists');
        var hdr = [

            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem_pinlists.find('.dict-table-pinlist-delete').
                            button().
                            click(function() {
                                var pinlist_id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the pinlist?',
                                    function() { that.delete_pinlist(pinlist_id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'pinlists' },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem_pinlists.find('.dict-table-pinlist-search').
                            button().
                            click(function() {
                                var pinlist_id = this.name;
                                global_search_cables_by_dict_pinlist_id(pinlist_id);
                            });
                    }
                }
            },

            {   name: 'DOCUMENTATION LINK', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem_pinlists.find('.pinlist-documentation-save').
                            button().
                            click(function() {this
                                var id = this.name;
                                that.save_pinlist_documentation(id, elem_pinlists.find('#pinlist-documentation-'+id).val());
                            });
                    }
                }
            },

            {   name: 'cable type', sorted: false,
                type: {
                    after_sort: function() {
                        elem_pinlists.find('.pinlist-cable').change(function() {
                            var pinlist_id = this.name;
                            var cable = this.value;
                            that.save_pinlist_cable(pinlist_id, cable);
                        });
                    }
                }
            },

            {   name: 'origin conn', sorted: false,
                type: {
                    after_sort: function() {
                        elem_pinlists.find('.pinlist-origin-connector').change(function() {
                            var pinlist_id = this.name;
                            var connector = this.value;
                            that.save_pinlist_origin_connector(pinlist_id, connector);
                        });
                    }
                }
            },

            {   name: 'destination conn', sorted: false,
                type: {
                    after_sort: function() {
                        elem_pinlists.find('.pinlist-destination-connector').change(function() {
                            var pinlist_id = this.name;
                            var connector = this.value;
                            that.save_pinlist_destination_connector(pinlist_id, connector);
                        });
                    }
                }
            }
        ];

        var cables = [''];
        for( var cable in this.cables()) cables.push(cable);

        var rows = [];
        for( var name in this.pinlists()) {
            var pinlist = this.pinlists()[name];
            var connectors = [''];
            for( var connector in this.connectors(pinlist.cable)) connectors.push(connector);

            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    pinlist.id,
                            classes: 'dict-table-pinlist-delete',
                            title:   'delete this pinlist from the dictionary' }) : ' ',

                    name,
                    pinlist.created_time,
                    pinlist.created_uid,

                    Button_HTML('search', {
                        name:    pinlist.id,
                        classes: 'dict-table-pinlist-search',
                        title:   'search all uses of this pinlist' }),

                    TextInput_HTML({
                        id:    'pinlist-documentation-'+pinlist.id,
                        value: pinlist.documentation })+(
                    this.can_manage() ?
                        Button_HTML('save', {
                            name:    pinlist.id,
                            classes: 'pinlist-documentation-save',
                            title:   'edit documentation URL for the pinlist' }) : ' ' ),

                    this.can_manage() ?
                        Select_HTML(cables, pinlist.cable, {
                            name:    pinlist.id,
                            classes: 'pinlist-cable',
                            title:   'define a cable type for the pinlist' }) : pinlist.cable,

                    this.can_manage() ?
                        Select_HTML(connectors, pinlist.origin_connector, {
                            name:    pinlist.id,
                            classes: 'pinlist-origin-connector',
                            title:   'define a connector type at cable origin for the pinlist' }) : pinlist.origin_connector,

                    this.can_manage() ?
                        Select_HTML(connectors, pinlist.destination_connector, {
                            name:    pinlist.id,
                            classes: 'pinlist-destination-connector',
                            title:   'define  a connector type at cable destination for the pinlist' }) : pinlist.destination_connector

                ]
            );
        }
        this.pinlist_table = new Table(
            'dictionary-pinlists-pinlists', hdr, rows,
            {default_sort_column: 1},
            config.handler('dict', 'pinlist_table')
        );
        this.pinlist_table.display();

        if(this.can_manage())
            $('#dictionary-pinlists').find('input[name="pinlist2add"]').removeAttr('disabled');
    };

    // -----------------
    // LOCATIONS & RACKS
    // -----------------

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
        this.save_location(input.val());
        input.val(''); };

    this.new_rack = function() {
        var input = $('#dictionary-locations').find('input[name="rack2add"]');
        this.save_rack(this.table_locations.selected_object(), input.val());
        input.val(''); };

    this.save_location = function(location_name) {
        if( location_name == '' ) return;
        this.location_action('../neocaptar/ws/dict_location_new.php', {location:location_name}); };

    this.save_rack = function(location_name, rack_name) {
        if(( location_name == '' ) || ( rack_name == '' )) return;
        this.location_action('../neocaptar/ws/dict_location_new.php', {location:location_name, rack:rack_name}); };

    this.delete_location_element = function(element,id) {
        this.location_action('../neocaptar/ws/dict_location_delete.php', {scope:element, id:id}); };

    this.load_locations = function() {
        this.location_action('../neocaptar/ws/dict_location_get.php', {}); };

    this.location_action = function(url, params, data_handler) {
        function handle_data_and_display(result) {
            if(data_handler) data_handler(result);
            else             that.location = result.location;
            that.display_locations();
        }
        this.web_service_GET(url, params, handle_data_and_display);
    };

    this.table_locations = null;
    this.table_racks = null;

    this.display_locations = function() {

        var elem = $('#dictionary-locations-locations');

        var hdr = [

            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-location-delete').
                            button().
                            click(function() {
                                var id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the location?',
                                    function() { that.delete_location_element('location',id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'location', selectable: true,
                type:{
                    select_action : function(location_name) {
                        that.display_racks( location_name );
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-location-search').
                            button().
                            click(function() {
                                var id = this.name;
                                global_search_cables_by_dict_location_id(id);
                            });
                    }
                }
            }
        ];

        var rows = [];

        for( var location_name in this.locations()) {
            var location = this.locations()[location_name];
            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    location.id,
                            classes: 'dict-table-location-delete',
                            title:   'delete this location from the dictionary' }) : ' ',

                    location_name,

                    location.created_time,
                    location.created_uid,

                    Button_HTML('search', {
                        name:    location.id,
                        classes: 'dict-table-location-search',
                        title:   'search all uses of this location' })
                ]
            );
        }
        this.table_locations = new Table(
            'dictionary-locations-locations',  hdr, rows,
            {default_sort_column: 1, selected_col: 1},
            config.handler('dict', 'table_locations')
        );
        this.table_locations.display();

        this.display_racks( this.table_locations.selected_object());

        if(this.can_manage())
            $('#dictionary-locations').find('input[name="location2add"]' ).removeAttr('disabled');
    };

    this.display_racks = function(location_name) {

        var elem = $('#dictionary-locations-racks');

        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-rack-delete').
                            button().
                            click(function() {
                                var id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the rack?',
                                    function() { that.delete_location_element('rack',id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'rack' },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-rack-search').
                            button().
                            click(function() {
                                var id = this.name;
                                global_search_cables_by_dict_rack_id(id);
                            });
                    }
                }
            }
        ];

        var rows = [];

        if( location_name != null ) {
            for( var rack_name in this.racks(location_name)) {
                var rack = this.racks(location_name)[rack_name];
                rows.push(
                    [   this.can_manage() ?
                            Button_HTML('X', {
                                name:    rack.id,
                                classes: 'dict-table-rack-delete',
                                title:   'delete this rack from the dictionary' }) : ' ',

                        rack_name,

                        rack.created_time,
                        rack.created_uid,

                        Button_HTML('search', {
                            name:    rack.id,
                            classes: 'dict-table-rack-search',
                            title:   'search all uses of this rack' })
                    ]
                );
            }
        }
        this.table_racks = new Table(
            'dictionary-locations-racks', hdr, rows,
            {default_sort_column: 1},
            config.handler('dict', 'table_racks')
        );
        this.table_racks.display();

        if(this.can_manage()) {
            var input = $('#dictionary-locations').find('input[name="rack2add"]');
            if( location_name == null ) input.attr('disabled','disabled');
            else                        input.removeAttr('disabled');
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
        this.save_routing(input.val());
        input.val(''); };

    this.save_routing = function(name) {
        if( name == '' ) return;
        this.routing_action('../neocaptar/ws/dict_routing_new.php', { name: name }); };

    this.delete_routing = function(id) {
        this.routing_action('../neocaptar/ws/dict_routing_delete.php', { id: id }); };

    this.load_routings = function() {
        this.routing_action('../neocaptar/ws/dict_routing_get.php', {}); };

    this.routing_action = function(url, params, data_handler) {
        function handle_data_and_display(result) {
            if(data_handler) data_handler(result);
            else             that.routing = result.routing;
            that.display_routings();
        }
        this.web_service_GET(url, params, handle_data_and_display);
    };

    this.table_routings = null;

    this.display_routings = function() {
       var elem = $('#dictionary-routings-routings');
        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-routing-delete').
                            button().
                            click(function() {
                                var routing_id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the routing?',
                                    function() { that.delete_routing(routing_id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'routings' },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },
            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-routing-search').
                            button().
                            click(function() {
                                var routing_id = this.name;
                                global_search_cables_by_dict_routing_id(routing_id);
                            });
                    }
                }
            }
        ];
        var rows = [];
        for( var name in this.routings()) {
            var routing = this.routings()[name];
            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    routing.id,
                            classes: 'dict-table-routing-delete',
                            title:   'delete this routing from the dictionary' }) : ' ',

                    name,
                    routing.created_time,
                    routing.created_uid,

                    Button_HTML('search', {
                        name:    routing.id,
                        classes: 'dict-table-routing-search',
                        title:   'search all uses of this routing' })
                ]
            );
        }
        this.table_routings = new Table(
            'dictionary-routings-routings', hdr, rows,
            {default_sort_column: 1},
            config.handler('dict', 'table_routings')
        );
        this.table_routings.display();

        if(this.can_manage())
            $('#dictionary-routings').find('input[name="routing2add"]').removeAttr('disabled');
    };


    // -------------------------------------------
    // DEVICE NAME: LOCATIONS, REGIONS, COMPONENTS
    // -------------------------------------------

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
                if( e.keyCode == 13     ) {    that.new_device_component(); return;    }
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
        var device_location =  input.val();
        if( device_location == null ) return;
        this.save_device_location(device_location);
        input.val('');
    };
    this.new_device_region = function() {
        var input = $('#dictionary-devices').find('input[name="device_region2add"]');
        var device_location =  this.table_device_locations.selected_object();
        if( device_location == null ) return;
        var device_region   =  input.val();
        if( device_region   == '' ) return;
        this.save_device_region(device_location, device_region);
        input.val('');
    };
    this.new_device_component = function() {
        var input = $('#dictionary-devices').find('input[name="device_component2add"]');
        var device_location =  this.table_device_locations.selected_object();
        if( device_location == null ) return;
        var device_region   =  this.table_device_regions.selected_object();
        if( device_region   == null ) return;
        var component_name  =  input.val();
        this.save_device_component(device_location, device_region, component_name);
        input.val('');
    };
    this.save_device_location = function(location_name) {
        if( location_name == '' ) return;
        this.device_action('../neocaptar/ws/dict_device_location_new.php', {location:location_name}); };

    this.save_device_region = function(location_name, region_name) {
        if(( location_name == '' ) || ( region_name == '' )) return;
        this.device_action('../neocaptar/ws/dict_device_location_new.php', {location:location_name, region:region_name}); };

    this.save_device_component = function(location_name, region_name, component_name) {
        if(( location_name == '' ) || ( region_name == '' ) || ( component_name == '' )) return;
        this.device_action('../neocaptar/ws/dict_device_location_new.php', {location:location_name, region:region_name, component:component_name}); };

    this.delete_device_element = function(element,id) {
        this.device_action('../neocaptar/ws/dict_device_location_delete.php', {scope:element, id:id}); };

    this.load_device_locations = function() {
        this.device_action('../neocaptar/ws/dict_device_location_get.php', {}); };

    this.device_action = function(url, params, data_handler) {
        function handle_data_and_display(result) {
            if(data_handler) data_handler(result);
            else             that.device_location = result.location;
            that.display_device_locations();
        }
        this.web_service_GET(url, params, handle_data_and_display);
    };

    this.table_device_locations  = null;
    this.table_device_regions    = null;
    this.table_device_components = null;

    this.display_device_locations = function() {

        var elem = $('#dictionary-devices-locations');

        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-device-location-delete').
                            button().
                            click(function() {
                                var device_location_id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the location?',
                                    function() { that.delete_device_element('location',device_location_id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'device location', selectable: true,
                type: {
                    select_action : function(location_name) {
                        that.display_device_regions( location_name );
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-device-location-search').
                            button().
                            click(function() {
                                var device_location_id = this.name;
                                global_search_cables_by_dict_device_location_id(device_location_id);
                            });
                    }
                }
            }
        ];

        var rows = [];

        for( var location_name in this.device_locations()) {
            var device_location = this.device_locations()[location_name];
            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    device_location.id,
                            classes: 'dict-table-device-location-delete',
                            title:   'delete this location from the dictionary' }) : ' ',

                    location_name,

                    device_location.created_time,
                    device_location.created_uid,

                    Button_HTML('search', {
                        name:    device_location.id,
                        classes: 'dict-table-device-location-search',
                        title:   'search all uses of this location' })
                ]
            );
        }
        this.table_device_locations = new Table(
            'dictionary-devices-locations',  hdr, rows,
            {default_sort_column: 1, selected_col: 1},
            config.handler('dict', 'table_device_locations')
        );
        this.table_device_locations.display();

        this.display_device_regions( this.table_device_locations.selected_object());

        if(this.can_manage())
            $('#dictionary-devices').find('input[name="device_location2add"]' ).removeAttr('disabled');
    };

    this.display_device_regions = function(location_name) {

        var elem = $('#dictionary-devices-regions');

        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-device-region-delete').
                            button().
                            click(function() {
                                var device_region_id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the region?',
                                    function() { that.delete_device_element('region',device_region_id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'device region', selectable: true,
                type: {
                    select_action : function(region_name) {
                        that.display_device_components( location_name, region_name );
                    }
                }
            },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },
            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-device-region-search').
                            button().
                            click(function() {
                                var device_region_id = this.name;
                                global_search_cables_by_dict_device_region_id(device_region_id);
                            });
                    }
                }
            }
        ];

        var rows = [];

        if( location_name != null ) {
            for( var region_name in this.device_regions(location_name)) {
                var device_region = this.device_regions(location_name)[region_name];
                rows.push(
                    [   this.can_manage() ?
                            Button_HTML('X', {
                                name:    device_region.id,
                                classes: 'dict-table-device-region-delete',
                                title:   'delete this region from the dictionary' }) : ' ',

                        region_name,

                        device_region.created_time,
                        device_region.created_uid,

                        Button_HTML('search', {
                            name:    device_region.id,
                            classes: 'dict-table-device-region-search',
                            title:   'search all uses of this region' })
                    ]
                );
            }
        }
        this.table_device_regions = new Table(
            'dictionary-devices-regions', hdr, rows,
            {default_sort_column: 1, selected_col: 1},
            config.handler('dict', 'table_device_regions')
        );
        this.table_device_regions.display();

        this.display_device_components(location_name, this.table_device_regions.selected_object());

        if(this.can_manage()) {
            var input = $('#dictionary-devices').find('input[name="device_region2add"]');
            if( location_name == null ) input.attr('disabled','disabled');
            else                        input.removeAttr('disabled');
        }
    };

    this.display_device_components = function(location_name, region_name) {

        var elem = $('#dictionary-devices-components');

        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-device-component-delete').
                            button().
                            click(function() {
                                var device_component_id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the component?',
                                    function() { that.delete_device_element('component',device_component_id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'device component' },
            {   name: 'created', hideable: true },
            {   name: 'by user', hideable: true },
            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-device-component-search').
                            button().
                            click(function() {
                                var device_component_id = this.name;
                                global_search_cables_by_dict_device_component_id(device_component_id);
                            });
                    }
                }
            }
        ];

        var rows = [];

        if(( location_name != null ) && ( region_name != null )) {
            for( var component_name in this.device_components(location_name, region_name)) {
                var device_component = this.device_components(location_name, region_name)[component_name];
                rows.push(
                    [   this.can_manage() ?
                            Button_HTML('X', {
                                name:    device_component.id,
                                classes: 'dict-table-device-component-delete',
                                title:   'delete this component from the dictionary' }) : ' ',

                        component_name,

                        device_component.created_time,
                        device_component.created_uid,

                        Button_HTML('search', {
                            name:    device_component.id,
                            classes: 'dict-table-device-component-search',
                            title:   'search all uses of this component' })
                    ]
                );
            }
        }
        this.table_device_components = new Table(
            'dictionary-devices-components', hdr, rows,
            {default_sort_column: 1},
            config.handler('dict', 'table_device_components')
        );
        this.table_device_components.display();

        if(this.can_manage()) {
            var input = $('#dictionary-devices').find('input[name="device_component2add"]');
            if(( location_name == null ) || ( region_name == null )) input.attr('disabled','disabled');
            else                                                     input.removeAttr('disabled');
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
        this.save_instr(input.val());
        input.val('');
    };
    this.save_instr = function(instr_name) {
        if( instr_name == '' ) return;
        this.instr_action('../neocaptar/ws/dict_instr_new.php',{instr:instr_name}); };

    this.delete_instr_element = function(element,id) {
        this.instr_action('../neocaptar/ws/dict_instr_delete.php',{scope:element, id:id}); };

    this.load_instrs = function() {
        this.instr_action('../neocaptar/ws/dict_instr_get.php',{}); };

    this.instr_action = function(url, params, data_handler) {
        function handle_data_and_display(result) {
            if(data_handler) data_handler(result);
            else             that.instr = result.instr;
            that.display_instrs();
        }
        this.web_service_GET(url, params, handle_data_and_display);
    };
    
    this.table_instrs = null;

    this.display_instrs = function() {
        var elem = $('#dictionary-instrs-instrs');
        var hdr = [
            {   name: 'DELETE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-instr-delete').
                            button().
                            click(function() {
                                var instr_id = this.name;
                                ask_yes_no(
                                    'Data Deletion Warning',
                                    'Are you sure you want to delete the instruction?',
                                    function() { that.delete_instr_element('instr',instr_id); },
                                    null
                                );
                            });
                    }
                }
            },

            {   name: 'instructions' },

            {   name: 'created', hideable: true },

            {   name: 'by user', hideable: true },

            {   name: 'USAGE', hideable: true, sorted: false,
                type: {
                    after_sort: function() {
                        elem.find('.dict-table-instr-search').
                            button().
                            click(function() {
                                var instr_id = this.name;
                                global_search_cables_by_dict_instr_id(instr_id);
                            });
                    }
                }
            }
        ];
        var rows = [];
        for( var name in this.instrs()) {
            var instr = this.instrs()[name];
            rows.push(
                [   this.can_manage() ?
                        Button_HTML('X', {
                            name:    instr.id,
                            classes: 'dict-table-instr-delete',
                            title:   'delete this instruction from the dictionary' }) : ' ',

                    name,
                    instr.created_time,
                    instr.created_uid,

                    Button_HTML('search', {
                        name:    instr.id,
                        classes: 'dict-table-instr-search',
                        title:   'search all uses of this instructions' })
                ]
            );
        }
        this.table_instrs = new Table(
            'dictionary-instrs-instrs', hdr, rows,
            {default_sort_column: 1},
            config.handler('dict', 'table_instrs')
        );
        this.table_instrs.display();

        if(this.can_manage())
            $('#dictionary-instrs').find('input[name="instr2add"]').removeAttr('disabled');
    };

    return this;
}
var dict = new p_appl_dictionary();



