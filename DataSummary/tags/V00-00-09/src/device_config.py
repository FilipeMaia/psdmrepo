from AppUtils.AppDataPath import AppDataPath

__version__ = '00.00.06'

dp = AppDataPath('DataSummary/device_config.dict')
with open(dp.path(),'r') as f:
    devices = eval(f.read())

del dp
