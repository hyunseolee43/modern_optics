import time
from instrumental import instrument, list_instruments, u
from pylablib.devices import Thorlabs
#collect information of all devices.
all_instrumental = list_instruments()
print(all_instrumental)
#Camera and DAQ data should print, similar to comment below:
# [<ParamSet[UC480_Camera] serial=b'4103300313' model=b'SC1280G12M' id=1>,
# <ParamSet[NIDAQ] name='Dev1' serial=32801876 model=b'USB-6003'>,


kinstage = Thorlabs.list_kinesis_devices() # pylablib gets info from stage
print(kinstage)
# Kinesis motor/stage data should print, similar to comment below:
# [('27260537', 'Brushed Motor Controller')]

# basic translation stage control prompts:
stage = Thorlabs.KinesisMotor("27260537")  # connect to the stage, INSERT YOUR SERIAL NUMBER HERE
stage.move_to(0) # move stage to this location

# a list of common commands, see pylablib documentation for more
stage.wait_move()  # wait until moving is done
stage.home()  # home the stage
stage.wait_for_home()  # wait until homing is done
stage.move_by(1000)  # move by 1000 steps
stage.wait_move()  # wait until moving is done
stage.jog("+")  # initiate jog (continuous move) in the positive direction
time.sleep(1.)  # wait for 1 second
stage.stop()  # stop the motion
stage.close()