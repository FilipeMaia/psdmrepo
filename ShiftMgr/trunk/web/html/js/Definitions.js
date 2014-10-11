define ([] ,

function () {

    return {

        AreaNames : [

        /*  |shortcuts      |in tables                   |in tooltips
         */

            {key: 'FEL' ,   name: 'FEL',                 description: 'To report problems with the machine, operations, FEE, etc.'} ,

            {key: 'BMLN',   name: 'Beamline',            description: 'To report problems with the photon beamline instrument\n ' +
                                                                      'including HPS and PPS problems'} ,

            {key: 'CTRL',   name: 'Controls',            description: 'To report problems specific to controls like motors, cameras,\n ' +
                                                                      'MPS, laser controls, etc. '} ,

            {key: 'DAQ' ,   name: 'DAQ',                 description: 'To report DAQ computer, data transfer and device problems'} ,

            {key: 'LASR',   name: 'Laser',               description: 'To report problems with the laser, (not laser controls)'} ,

            {key: 'TIME',   name: 'Timing',              description: 'To report problems with the timing system including: EVR triggers,\n ' +
                                                                      'RF, LBL Timing System, fstiming interface (laser related problems\n ' +
                                                                      'should be reported under laser)'} ,

            {key: 'HALL',      name: 'Hutch/Hall',       description: 'To report problem with the hutch like: PCW, temperature, setup space,\n ' +
                                                                      'common stock, etc.  Note that this could be confused with the overall\n ' +
                                                                      'name of this section of the form.'} ,

            {key: 'OTHR',      name: 'Other',            description: 'Any other areas that might have problems can be addressed'}
        ] ,

        AllocationNames : [

            {key: 'tuning'   , name: 'Tuning',           description: 'This is time spent tuning the machine (accelerator, undulator, or FEE)\n ' +
                                                                      'up to the point where the specified photon parameters can be delivered\n ' +
                                                                      'to the PPS stoppers'} ,

            {key: 'alignment', name: 'Alignment',        description: 'This is time spent aligning, calibrating,  turning the photon\n ' +
                                                                      'instrumentation'} ,

            {key: 'daq',       name: 'Data Taking',      description: 'This is time spent taking data that can be used in publication'} ,

            {key: 'access',    name: 'Hutch Access',     description: 'This time spent in the hutch for sample changes, laser tuning\n ' +
                                                                      'and troubleshooting. This includes any problems downstream of\n ' +
                                                                      'the PPS stoppers that prevent alignment or data taking'} ,

            {key: 'machine',   name: 'Machine Downtime', description: 'This is the time during which problems with the machine\n ' +
                                                                      '(accelerator, undulator, or FEE) prevented beam from\n ' +
                                                                      'being delivered to an experiment. Downtime originating from\n ' +
                                                                      'problems downstream of the PPS stoppers should not be included here'} ,

            {key: 'other',     name: 'Other',            description: 'This time spent on extenuating circumstances preventing tuning,\n ' +
                                                                      'alignment or data taking (but not including machine downtime).\n ' +
                                                                      'For example: site wide power outage. The time in other will be\n ' +
                                                                      'automatically calculated to absorb the time not allocated to tuning,\n ' +
                                                                      'alignment, data taking, hutch access and machine downtime.\n ' +
                                                                      'Therefore, any time greater than 30 min in other must be commented.'}
        ] ,

        MinOther2Comment : 30 ,             /* the minimal number of minutes required to comment in time allocations */

        ShiftsUpdateInterval_Sec : 30       /* how frequently to update shift information */
    } ;
}) ;