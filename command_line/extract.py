#!/usr/bin/env python
#
# dials.extract.py
#
#  Copyright (C) 2013 Diamond Light Source
#
#  Author: James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

from __future__ import division
from dials.util.script import ScriptRunner

class Script(ScriptRunner):
  '''A class for running the script.'''

  def __init__(self):
    '''Initialise the script.'''

    # The script usage
    usage = "usage: %prog [options] [param.phil] "\
            "{sweep.json | image1.file [image2.file ...]}"

    # Initialise the base class
    ScriptRunner.__init__(self, usage=usage, home_scope="integration")

    # The block length
    self.config().add_option(
        '-n', '--num-blocks',
        dest = 'num_blocks',
        type = 'int', default = 1,
        help = 'Set the number of blocks')

    # Output filename option
    self.config().add_option(
        '-o', '--output-filename',
        dest = 'output_filename',
        type = 'string', default = 'shoebox.dat',
        help = 'Set the filename for the extracted spots.')

    # Output filename option
    self.config().add_option(
        '--force-static',
        dest = 'force_static',
        action = "store_true", default = False,
        help = 'For a scan varying model force static prediction.')

  def main(self, params, options, args):
    '''Execute the script.'''
    from dials.model.serialize import load, dump
    from dials.util.command_line import Command
    from dials.util.command_line import Importer
    from dials.array_family import flex
    from dials.model.serialize import extract_shoeboxes_to_file
    from dials.algorithms.profile_model.profile_model import ProfileModelList
    from dials.algorithms.profile_model.profile_model import ProfileModel
    from math import pi

    # Check the unhandled arguments
    importer = Importer(args, include=['experiments'])
    if len(importer.unhandled_arguments) > 0:
      print '-' * 80
      print 'The following command line arguments weren\'t handled'
      for arg in importer.unhandled_arguments:
        print '  ' + arg

    # Check the number of experiments
    if importer.experiments is None or len(importer.experiments) == 0:
      self.config().print_help()
      return
    elif len(importer.experiments) > 1:
      print 'Error: only 1 experiment currently supported'
      return

    # Populate the reflection table with predictions
    predicted = flex.reflection_table.from_predictions(
      importer.experiments[0],
      force_static=options.force_static)
    predicted['id'] = flex.size_t(len(predicted), 0)

    # Get the bbox nsigma
    profile_model = ProfileModelList()
    profile_model.append(ProfileModel(
      params.integration.shoebox.n_sigma,
      params.integration.shoebox.sigma_b * pi / 180.0,
      params.integration.shoebox.sigma_m * pi / 180.0))

    # Calculate the bounding boxes
    predicted.compute_bbox(importer.experiments, profile_model)

    # TODO Need to save out reflections
    z = predicted['xyzcal.px'].parts()[2]
    index = sorted(range(len(z)), key=lambda x: z[x])
    predicted.reorder(flex.size_t(index))

    # Extract the shoeboxes to file
    extract_shoeboxes_to_file(
      options.output_filename,
      importer.experiments[0].imageset,
      predicted)


if __name__ == '__main__':
  script = Script()
  script.run()
