#
# algorithm.py
#
#  Copyright (C) 2013 Diamond Light Source
#
#  Author: James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

from __future__ import division


class IntegrationAlgorithm(object):
  ''' Class to do reciprocal space profile fitting. '''

  def __init__(self,
               experiments,
               profile_model,
               grid_size=5,
               threshold=0.02,
               single_reference=False,
               debug=False):
    '''
    Initialise algorithm.

    :param experiments: The list of experiments
    :param profile model: The profile model
    :param grid_size: The size of the kabsch space grid
    :param threshold: The intensity fraction used for reference profiles
    :param single_reference: Use a single reference profile
    :param debug: Save debugging information

    '''
    assert(len(experiments) == len(profile_model))
    self._experiments = experiments
    self._profile_model = profile_model
    self._grid_size = grid_size
    self._threshold = threshold
    self._single_reference = single_reference
    self._debug = debug

  def __call__(self, reflections):
    '''
    Process the reflections.

    :param reflections: The reflections to process
    :return: The list of integrated reflections

    '''
    from dials.algorithms.integration.fitrs import ReciprocalSpaceProfileFitting
    from dials.algorithms.integration.fitrs import Spec
    from dials.algorithms.integration.integrator import job
    from dials.util import pprint
    from dials.array_family import flex
    from logging import info, warn, debug
    from time import time

    # Start the profile fitting
    info('')
    info(' Beginning integration by profile fitting')
    start_time = time()

    # Get the flags
    flags = flex.reflection_table.flags
    num = reflections.get_flags(flags.dont_integrate).count(False)
    info('  using %d reflections' % num)

    # Create the algorithm
    algorithm = ReciprocalSpaceProfileFitting(
      self._grid_size,
      self._threshold,
      self._single_reference)

    # Add the specs
    for experiment, model in zip(self._experiments, self._profile_model):

      # Get the sigmas
      sigma_b = model.sigma_b(deg=False)
      sigma_m = model.sigma_m(deg=False)
      n_sigma = model.n_sigma()
      if isinstance(sigma_b, flex.double):
        assert(len(sigma_b) == len(sigma_m))
        sigma_b = sum(sigma_b) / len(sigma_b)
        sigma_m = sum(sigma_m) / len(sigma_m)

      # Add the spec
      algorithm.add(Spec(
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        sigma_b,
        sigma_m,
        n_sigma))

    # Perform the integration
    profiles = algorithm.execute(reflections)

    # Print the number integrated
    num = reflections.get_flags(flags.integrated_prf).count(True)
    info('  successfully processed %d reflections' % num)
    info('  time taken: %g seconds' % (time() - start_time))

    # Print the reference profiles
    for i in range(len(self._experiments)):
      for j in range(profiles.single_size(i)):
        debug("")
        debug("Profile %d for experiment %d" % (j, i))
        debug(pprint.profile3d(profiles.data(i,j)))

    # Print warning
    nbad = profiles.nbad()
    if nbad > 0:
      warn('')
      warn(' ' + '*' * 79)
      warn('  Warning: %d standard profile(s) could not be created' % nbad)
      warn(' ' + '*' * 79)

    # Maybe save some debug info
    if self._debug:
      import cPickle as pickle
      filename = 'debug_%d.pickle' % job.index
      info('Saving debugging information to %s' % filename)
      reference = []
      for i in range(len(profiles)):
        r = []
        for j in range(profiles.single_size(i)):
          r.append(profiles.data(i,j))
        reference.append(r)
      output = {
        'reflections' : reflections,
        'experiments' : self._experiments,
        'profile_model' : self._profile_model,
        'reference' : reference,
      }
      with open(filename, 'wb') as outfile:
        pickle.dump(output, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    # Return the reflections
    return reflections
