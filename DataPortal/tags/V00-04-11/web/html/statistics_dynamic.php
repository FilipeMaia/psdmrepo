<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title>Testing Dynamic Table class</title>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="../portal/js/config.js"></script>
<script type="text/javascript" src="../portal/js/Table.js"></script>

<style type="text/css">
body {
  margin: 0;
  padding: 0;
}
.table_container {
  margin-left: 20px;
}
</style>

<script type="text/javascript">

var config = new config_create('statistic_dynamic') ;

$(function() {

    var table_total = new Table(
        'table_total',
         [
            {   name: 'Runs'            },
            {   name: 'Files',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]},
            {   name: 'Size [TB]',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]}
        ]
    );
    table_total.display();

    var table_filesystem = new Table(
        'table_filesystem',
         [
            {   name: 'Path'            },
            {   name: 'Runs'            },
            {   name: 'Files',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]},
            {   name: 'Size [TB]',
                coldef: [
                    {   name: 'XTC'             },
                    {   name: 'HDF5'            },
                    {   name: '&sum;'           },
                    {   name: '&sum; / (&sum;)' },
                    {   name: 'On Disk'         }]}
        ]
    );
    table_filesystem.display();

    var table_instruments = new Table(
        'table_instruments',
         [
            {   name: 'Instr.'          },
            {   name: 'Runs'            },
            {   name: 'Files',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]},
            {   name: 'Size [TB]',
                coldef: [
                    {   name: 'XTC'             },
                    {   name: 'HDF5'            },
                    {   name: '&sum;'           },
                    {   name: '&sum; / (&sum;)' },
                    {   name: 'On Disk'         }]}
        ]
    );
    table_instruments.display();

    var table_experiments = new Table(
        'table_experiments',
         [
            {   name: 'Experiment'      },
            {   name: 'ID'              },
            {   name: 'First Run'       },
            {   name: 'Last Run'        },
            {   name: 'Runs'            },
            {   name: 'Files',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]},
            {   name: 'Size [TB]',
                coldef: [
                    {   name: 'XTC'             },
                    {   name: 'HDF5'            },
                    {   name: '&sum;'           },
                    {   name: '&sum; / (&sum;)' },
                    {   name: 'On Disk'         }]},
            {   name: 'Filesystem '     }
        ]
    );
    table_experiments.display();

    var table_months = new Table(
        'table_months',
         [
            {   name: 'Year-Month'      },
            {   name: 'Runs'            },
            {   name: 'Files',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]},
            {   name: 'Size [TB]',
                coldef: [
                    {   name: 'XTC'             },
                    {   name: 'HDF5'            },
                    {   name: '&sum;'           },
                    {   name: '&sum; / (&sum;)' },
                    {   name: 'On Disk'         }]}
        ]
    );
    table_months.display();

    var table_accumulated = new Table(
        'table_accumulated',
         [
            {   name: 'Year-Month'      },
            {   name: 'Runs'            },
            {   name: 'Files',
                coldef: [
                    {   name: 'XTC'     },
                    {   name: 'HDF5'    },
                    {   name: '&sum;'   },
                    {   name: 'On Disk' }]},
            {   name: 'Size [TB]',
                coldef: [
                    {   name: 'XTC'             },
                    {   name: 'HDF5'            },
                    {   name: '&sum;'           },
                    {   name: '&sum; / (&sum;)' },
                    {   name: 'On Disk'         }]}
        ]
    );
    table_accumulated.display();

    table_total.erase      (Table.Status.Loading);
    table_filesystem.erase (Table.Status.Loading);
    table_instruments.erase(Table.Status.Loading);
    table_experiments.erase(Table.Status.Loading);
    table_months.erase     (Table.Status.Loading);
    table_accumulated.erase(Table.Status.Loading);

    $.ajax({
        type: 'GET',
        url: '../portal/ws/statistics_get.php',
        data: {
            json: ''},
        success: function(result) {
            if(result.status != 'success') {
                table5.erase(Table.Status.error(result.message));
                return;
            }
            var total = result.total;
            var total_data = [[
                total.runs,
                total.files.xtc,
                total.files.hdf5,
                total.files.xtc + total.files.hdf5,
                total.files_disk.xtc + total.files_disk.hdf5,
                total.size_tb.xtc,
                total.size_tb.hdf5,
                total.size_tb.xtc + total.size_tb.hdf5,
                total.size_tb_disk.xtc + total.size_tb_disk.hdf5,
            ]];
            table_total.load(total_data);
        },
        error: function() {
            table_total.erase      (Table.Status.error('service is not available'));
            table_filesystem.erase (Table.Status.error('service is not available'));
            table_instruments.erase(Table.Status.error('service is not available'));
            table_experiments.erase(Table.Status.error('service is not available'));
            table_months.erase     (Table.Status.error('service is not available'));
            table_accumulated.erase(Table.Status.error('service is not available'));
        },
        dataType: 'json'
    });

});

</script>

</head>
<body>

<div style="padding:20px;">

  <h2>About</h2>
  <div style="padding-left:20px;">
    <p>The information found on this page represents a summary data statistics for
       all LCLS experiments we have had so far. The information is break down into
       five sections:</p>
       <ul>
         <li><a href="#total">Total numbers across all experiments</a></li>
         <li><a href="#filesystem">Total numbers across file systems</a></li>
         <li><a href="#instruments">For each instrument</a></li>
         <li><a href="#experiments">For each experiment</a></li>
         <li><a href="#months">For each month of data taking</a></li>
         <li><a href="#accumulated">Accumulated (total) statistics progression by month</a></li>
       </ul>
    <p>In case if someone may want to incorporate this information into
       a dynamic HTML page the report can be also be obtained in JSON format
       from <a href="statistics?json" target="_blank">here</a>.</p>
  </div>

  <h2>Total</h2>
  <div class="table_container" id="table_total"></div>

  <h2>File System</h2>
  <div class="table_container" id="table_filesystem"></div>

  <h2>Instruments</h2>
  <div class="table_container" id="table_instruments"></div>

  <h2>Experiments</h2>
  <div class="table_container" id="table_experiments"></div>

  <h2>Each month of data taking</h2>
  <div class="table_container" id="table_months"></div>

  <h2>Accumulated (total) statistics progression by month</h2>
  <div class="table_container" id="table_accumulated"></div>

</div>

</body>
</html>
