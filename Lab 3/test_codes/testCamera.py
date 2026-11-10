from PIL import Image
import scipy.io
from instrumental import instrument, list_instruments, u

#collect information of all devices.
all_instrumental = list_instruments()
print(all_instrumental)
# [<ParamSet[UC480_Camera] serial=b'4103300313' model=b'SC1280G12M' id=1>,
# <ParamSet[NIDAQ] name='Dev1' serial=32801876 model=b'USB-6003'>,
# <ParamSet[TDC001] serial='83860028'>]

# obtain the camera object.
myCam = instrument(all_instrumental[0])

#get the image

img = myCam.grab_image(exposure_time='18ms')


# show the image
pil_img=Image.fromarray(img)
pil_img.save('myimg.png') # or other image format such as .tiff/.jpg
pil_img.show()

# save image to .mat file
scipy.io.savemat("test.mat", dict(img_in_matlab=img))