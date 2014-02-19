#
# dials.algorithms.background.flat_subtractor.py
#
#  Copyright (C) 2013 Diamond Light Source
#
#  Author: Luis Fuentes-Montero "luiso" & James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

from __future__ import division

from dials.algorithms.background.background_subtraction_2d \
          import flat_background_calc_2d

class FlatSubtractor(object):
  ''' The Flat background subtractor '''

  def __init__(self, **kwargs):
    pass

  def __call__(self, reflections):
    layering_and_background_avg(reflections)
    return reflections

def layering_and_background_avg(reflections):
  from dials.algorithms.background import flat_background_flex_2d
  from scitbx.array_family import flex

  shoeboxes = reflections['shoebox']
  for shoebox in shoeboxes:
    #if ref.is_valid():
      data = shoebox.data
      mask = shoebox.mask
      background = shoebox.background
      for i in range(data.all()[0]):
        data2d = data[i:i + 1, :, :]
        mask2d = mask[i:i + 1, :, :]
        data2d.reshape(flex.grid(data.all()[1:]))
        mask2d.reshape(flex.grid(data.all()[1:]))
        background2d = flat_background_flex_2d(data2d.as_double(), mask2d)
        background2d.reshape(flex.grid(1, background2d.all()[0], background2d.all()[1]))
        background[i:i + 1, :, :] = background2d.as_double()
  return reflections

  no_longer_used = '''
  def tmp_numpy_layering_n_bkgr_avg(reflections):
    import numpy
    from scitbx.array_family import flex
    print "averaging background tmp numpy"
    for ref in reflections:
      shoebox = ref.shoebox.as_numpy_array()
      mask = ref.shoebox_mask.as_numpy_array()
      background = numpy.copy(shoebox)
      for i in range(shoebox.shape[0]):
        data2d = shoebox[i]
        mask2d = mask[i]
        background2d = background[i]
        background2d = flat_background_calc_2d(data2d, mask2d)
        background[i] = background2d

      ref.shoebox = flex.double(shoebox)
      ref.shoebox_background = flex.double(background)

    return reflections
  #'''