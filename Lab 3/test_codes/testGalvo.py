
from instrumental import instrument, u
from numpy import linspace, pi, sin

# Connect directly to NI DAQ (no device scanning)
myDaq = instrument(module='daq.ni', classname='NIDAQ', name='Dev1')
print("NI DAQ connected:", myDaq)

# Initial test: output 0 V on AO1
myDaq.ao1.write("0V")

# Uncomment only after verifying the galvo's permitted voltage range.
# Generate a sine wave from 0 to 10 V
# data = sin(2*pi*linspace(0, 1, 100, endpoint=False))*5*u.V + 5*u.V
# myDaq.ao1.write(data, duration='3s', fsamp='100Hz')
