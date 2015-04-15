from AppUtils.AppDataPath import AppDataPath

dp = AppDataPath('DataSummary/device_config.dict')
with open(dp.path(),'r') as f:
    devices = eval(f.read())

del dp
