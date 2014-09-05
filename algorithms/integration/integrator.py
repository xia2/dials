#
# integrator.py
#
#  Copyright (C) 2013 Diamond Light Source
#
#  Author: James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

from __future__ import division


class ReflectionBlockIntegrator(object):
  ''' A class to perform the integration. '''

  def __init__(self, params, experiments, extractor=None):
    ''' Initialise the integrator. '''
    from math import pi
    from dials.algorithms.profile_model.profile_model import ProfileModelList
    from dials.algorithms.profile_model.profile_model import ProfileModel

    # Ensure we have 1 experiment at the moment
    assert(len(experiments) == 1)
    assert(extractor is not None)

    # Save the parameters
    self.params = params
    self.experiments = experiments
    self.extractor = extractor

    # Create the shoebox masker
    n_sigma = params.integration.shoebox.n_sigma
    sigma_b = params.integration.shoebox.sigma_b
    sigma_m = params.integration.shoebox.sigma_m
    assert(n_sigma > 0)
    assert(sigma_b > 0)
    assert(sigma_m > 0)
    self.profile_model = ProfileModelList()
    self.profile_model.append(ProfileModel(
      n_sigma,
      sigma_b * pi / 180.0,
      sigma_m * pi / 180.0))

  def integrate(self):
    ''' Integrate all the reflections. '''
    from dials.array_family import flex
    from dials.algorithms.shoebox import MaskCode
    from dials.framework.registry import Registry
    result = flex.reflection_table()
    registry = Registry()
    params = registry.params()
    flex.reflection_table._background_algorithm = flex.strategy(
      registry["integration.background"], params)
    flex.reflection_table._intensity_algorithm = flex.strategy(
      registry["integration.intensity"], params)
    flex.reflection_table._centroid_algorithm = flex.strategy(
      registry["integration.centroid"], params)
    for indices, reflections in self.extractor:
      reflections.compute_mask(self.experiments, self.profile_model)
      reflections.integrate(self.experiments, self.profile_model)
      bg_code = MaskCode.Valid | MaskCode.BackgroundUsed
      fg_code = MaskCode.Valid | MaskCode.Foreground
      n_bg = reflections['shoebox'].count_mask_values(bg_code)
      n_fg = reflections['shoebox'].count_mask_values(fg_code)
      reflections['n_background'] = n_bg
      reflections['n_foreground'] = n_fg
      del reflections['shoebox']
      del reflections['rs_shoebox']
      result.extend(reflections)
      print ''
    assert(len(result) > 0)
    result.sort('miller_index')
    return result


class Integrator(object):
  ''' Integrate reflections '''

  def __init__(self, params, exlist, reference=None,
               predicted=None, shoeboxes=None):
    '''Initialise the script.'''

    # Load the reference spots and compute the profile parameters
    if reference:
      self._compute_profile_model(params, exlist, reference)

    # Load the extractor based on the input
    if shoeboxes is not None:
      extractor = self._load_extractor(shoeboxes, params, exlist)
    else:
      if predicted is None:
        predicted = self._predict_reflections(params, exlist)
        predicted = self._filter_reflections(params, exlist, predicted)
      if reference:
        predicted = self._match_with_reference(predicted, reference)
      extractor = self._create_extractor(params, exlist, predicted)

    # Initialise the integrator
    self._integrator = ReflectionBlockIntegrator(params, exlist, extractor)

  def integrate(self):
    ''' Integrate the reflections. '''
    return self._integrator.integrate()

  def _match_with_reference(self, predicted, reference):
    ''' Match predictions with reference spots. '''

    from dials.algorithms.peak_finding.spot_matcher import SpotMatcher
    from dials.util.command_line import Command
    Command.start("Matching reference spots with predicted reflections")
    match = SpotMatcher(max_separation=1)
    rind, pind = match(reference, predicted)
    h1 = predicted.select(pind)['miller_index']
    h2 = reference.select(rind)['miller_index']
    mask = (h1 == h2)
    predicted.set_flags(pind.select(mask), predicted.flags.reference_spot)
    Command.end("Matched %d reference spots with predicted reflections" %
                mask.count(True))
    return predicted

  def _load_extractor(self, filename, params, exlist):
    ''' Load the shoebox extractor. '''
    from dials.model.serialize.reflection_block import ReflectionBlockExtractor
    assert(len(exlist) == 1)
    imageset = exlist[0].imageset
    return ReflectionBlockExtractor(
      filename,
      params.integration.shoebox.block_size,
      imageset)

  def _create_extractor(self, params, exlist, predicted):
    ''' Create the extractor. '''
    from dials.model.serialize.reflection_block import ReflectionBlockExtractor
    assert(len(exlist) == 1)
    imageset = exlist[0].imageset
    return ReflectionBlockExtractor(
      "shoebox.dat",
      params.integration.shoebox.block_size,
      imageset,
      predicted)

  def _compute_profile_model(self, params, experiments, reference):
    ''' Compute the profile model. '''
    from dials.algorithms.profile_model.profile_model import ProfileModel
    from math import pi
    if (params.integration.shoebox.sigma_b is None or
        params.integration.shoebox.sigma_m is None):
      assert(reference is not None)
      assert(len(experiments) == 1)
      profile_model = ProfileModel.compute(experiments[0], reference)
      params.integration.shoebox.sigma_b = profile_model.sigma_b(deg=True)
      params.integration.shoebox.sigma_m = profile_model.sigma_m(deg=True)
      print 'Sigma B: %f' % params.integration.shoebox.sigma_b
      print 'Sigma M: %f' % params.integration.shoebox.sigma_m

  def _predict_reflections(self, params, experiments):
    ''' Predict all the reflections. '''
    from dials.array_family import flex
    from dials.algorithms.profile_model.profile_model import ProfileModelList
    from dials.algorithms.profile_model.profile_model import ProfileModel
    from math import pi
    n_sigma = params.integration.shoebox.n_sigma
    sigma_b = params.integration.shoebox.sigma_b * pi / 180.0
    sigma_m = params.integration.shoebox.sigma_m * pi / 180.0
    profile_model = ProfileModelList()
    profile_model.append(ProfileModel(
      n_sigma,
      sigma_b,
      sigma_m))
    result = flex.reflection_table()
    for i, experiment in enumerate(experiments):
      predicted = flex.reflection_table.from_predictions(experiment)
      predicted['id'] = flex.size_t(len(predicted), i)
      result.extend(predicted)
    result.compute_bbox(experiments, profile_model)
    return result

  def _filter_reflections(self, params, experiments, reflections):
    ''' Filter the reflections to integrate. '''
    from dials.util.command_line import Command
    from dials.algorithms import filtering
    from dials.array_family import flex

    image = experiments[0].imageset[0]
    detector = experiments[0].detector
    if not isinstance(image, tuple):
      image = (image,)
    image_mask = []
    for im, panel in zip(image, detector):
      tr = panel.get_trusted_range()
      m = im > int(tr[0])
      image_mask.append(m)
    image_mask = tuple(image_mask)

    # Set all reflections which overlap bad pixels to zero
    Command.start('Filtering reflections by detector mask')
    array_range = experiments[0].scan.get_array_range()
    mask = filtering.by_detector_mask(
      reflections['panel'],
      reflections['bbox'],
      image_mask,
      array_range)
    reflections.del_selected(mask != True)
    Command.end('Filtered %d reflections by detector mask' % len(reflections))
    assert(len(reflections) > 0)

    # Filter the reflections by zeta
    min_zeta = params.integration.filter.by_zeta
    if min_zeta > 0:
      Command.start('Filtering reflections by zeta >= %f' % min_zeta)
      zeta = reflections.compute_zeta(experiments[0])
      reflections.del_selected(flex.abs(zeta) < min_zeta)
      n = len(reflections)
      Command.end('Filtered %d reflections by zeta >= %f' % (n, min_zeta))
      assert(len(reflections) > 0)
    return reflections
