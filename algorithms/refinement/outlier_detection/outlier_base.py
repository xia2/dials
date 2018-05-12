from __future__ import absolute_import, division, print_function

import logging
from math import pi

from dials.array_family import flex
from libtbx.phil import parse
from libtbx.table_utils import simple_table

logger = logging.getLogger(__name__)

RAD2DEG = 180./pi

class CentroidOutlier(object):
  """Base class for centroid outlier detection algorithms"""

  def __init__(self, cols=None,
               min_num_obs=20,
               separate_experiments=True,
               separate_panels=True,
               block_width=None):

    # column names of the data in which to look for outliers
    if cols is None:
      cols = ["x_resid", "y_resid", "phi_resid"]
    self._cols = cols

    # minimum number of observations per panel below which all reflections will
    # be marked as potential outliers
    self._min_num_obs = min_num_obs

    # whether to do outlier rejection within each experiment or panel, or across
    # all experiments and panels
    self._separate_experiments = separate_experiments
    self._separate_panels = separate_panels

    # block width for splitting scans over phi, or None for no split
    self._block_width = block_width

    # the number of rejections
    self.nreject = 0

    self._verbosity = 0

    return

  def get_block_width(self, exp_id=None):
    if exp_id is None:
      return self._block_width
    else:
      try:
        bw = self._block_width[exp_id]
      except TypeError:
        bw = self._block_width
      return bw

  def set_block_width(self, block_width):
    """Set the block width for outlier detection in degrees. This can be either
    a single value or a list with one value per experiment. None is accepted
    to mean that the dataset will not be split into blocks."""
    self._block_width = block_width

  def set_verbosity(self, verbosity):
    logger.disabled = (verbosity == 0)
    self._verbosity = verbosity

  def _detect_outliers(cols):
    """Perform outlier detection using the input cols and return a flex.bool
    indicating which rows in the cols are considered outlying. cols should be
    a list of flex arrays of equal lengths"""

    # to be implemented by derived classes
    raise NotImplementedError()

  def __call__(self, reflections):
    """Identify outliers in the input and set the centroid_outlier flag.
    Return True if any outliers were detected, otherwise False"""

    logger.info("Detecting centroid outliers using the {0} algorithm".format(
      type(self).__name__))

    # check the columns are present
    for col in self._cols: assert col in reflections

    sel = reflections.get_flags(reflections.flags.used_in_refinement)
    all_data = reflections.select(sel)
    all_data_indices = sel.iselection()
    nexp = flex.max(all_data['id']) + 1

    jobs = []
    if self._separate_experiments:
      # split the data set by experiment id
      for iexp in xrange(nexp):
        sel = all_data['id'] == iexp
        job = {'id':iexp, 'panel':'all', 'data':all_data.select(sel),
               'indices':all_data_indices.select(sel)}
        jobs.append(job)
    else:
      # keep the whole dataset across all experiment ids
      job = {'id':'all', 'panel':'all', 'data':all_data,
             'indices':all_data_indices}
      jobs.append(job)

    jobs2 = []
    if self._separate_panels:
      # split further by panel id
      for job in jobs:
        data = job['data']
        iexp = job['id']
        indices = job['indices']
        for ipanel in xrange(flex.max(data['panel']) + 1):
          sel = data['panel'] == ipanel
          job = {'id':iexp, 'panel':ipanel, 'data':data.select(sel),
                  'indices':indices.select(sel)}
          jobs2.append(job)
    else:
      # keep the splits as they are
      jobs2 = jobs

    jobs3 = []
    if self.get_block_width() is not None:
      # split into equal-sized phi ranges
      for job in jobs2:
        data = job['data']
        iexp = job['id']
        ipanel = job['panel']
        indices = job['indices']
        phi = data['xyzobs.mm.value'].parts()[2]
        if len(phi) == 0: # detect no data in the job
          jobs3.append(job)
          continue
        phi_low = flex.min(phi)
        phi_range = flex.max(phi) - phi_low
        if phi_range == 0.0: # detect stills and do not split
          jobs3.append(job)
          continue
        bw = self.get_block_width(iexp)
        if bw is None: # detect no split for this experiment
          jobs3.append(job)
          continue
        nblocks = int(round(RAD2DEG * phi_range / bw))
        nblocks = max(1, nblocks)
        real_width = phi_range / nblocks
        block_end = 0.0
        for iblock in xrange(nblocks - 1): # all except the last block
          block_start = iblock * real_width
          block_end = (iblock + 1) * real_width
          sel = (phi >=  (phi_low + block_start)) & \
                (phi < (phi_low + block_end))
          job = {'id':iexp, 'panel':ipanel, 'data':data.select(sel),
                 'indices':indices.select(sel),
                 'phi_start':RAD2DEG*(phi_low + block_start),
                 'phi_end':RAD2DEG*(phi_low + block_end)}
          jobs3.append(job)
        # now last block
        sel = phi >= (phi_low + block_end)
        job = {'id':iexp, 'panel':ipanel, 'data':data.select(sel),
               'indices':indices.select(sel),
               'phi_start':RAD2DEG*(phi_low + block_end),
               'phi_end':RAD2DEG*(phi_low + phi_range)}
        jobs3.append(job)
    else:
      # keep the splits as they are
      jobs3 = jobs2

    # Work out the format of the jobs table
    header = ['Job']
    if self._separate_experiments: header.append('Exp\nid')
    if self._separate_panels: header.append('Panel\nid')
    if self.get_block_width() is not None: header.append('Block range\n(deg)')
    header.extend(['Nref', 'Nout', '%out'])
    rows = []

    # now loop over the lowest level of splits
    for i, job in enumerate(jobs3):

      data = job['data']
      indices = job['indices']
      iexp = job['id']
      ipanel = job['panel']
      nref = len(indices)

      if nref >= self._min_num_obs:

        # get the subset of data as a list of columns
        cols = [data[col] for col in self._cols]

        # determine the position of outliers on this sub-dataset
        outliers = self._detect_outliers(cols)

        # get positions of outliers from the original matches
        ioutliers = indices.select(outliers)

      elif nref > 0:
        # too few reflections in the job
        msg = "For job {0}, fewer than {1} reflections are present.".format(
          i + 1, self._min_num_obs)
        msg += " All reflections flagged as possible outliers."
        logger.debug(msg)
        ioutliers = indices

      else:
        # no reflections in the job
        ioutliers = indices

      # set the centroid_outlier flag in the original reflection table
      nout = len(ioutliers)
      if nout > 0:
        reflections.set_flags(ioutliers,
          reflections.flags.centroid_outlier)
        self.nreject += nout

      # Add job data to the table
      row = [str(i + 1)]
      if self._separate_experiments: row.append(str(iexp))
      if self._separate_panels: row.append(str(ipanel))
      if self.get_block_width() is not None:
        try:
          row.append('{phi_start:.2f} - {phi_end:.2f}'.format(**job))
        except KeyError:
          row.append('{0:.2f} - {1:.2f}'.format(0.0,0.0))
      if nref == 0:
        p100 = 0
      else:
        p100 = nout / nref * 100.0
        if p100 > 30.0:
          msg = ("{0:3.1f}% of reflections were flagged as outliers from job"
                 " {1}").format(p100, i + 1)
      row.extend([str(nref), str(nout), '%3.1f' % p100])
      rows.append(row)

    if self.nreject == 0: return False
    logger.info("{0} reflections have been flagged as outliers".format(self.nreject))
    logger.debug("Outlier rejections per job:")
    st = simple_table(rows, header)
    logger.debug(st.format())

    return True

# The phil scope for outlier rejection
phil_str = '''
outlier
  .help = "Outlier rejection after initial reflection prediction."
{
  algorithm = null *auto mcd tukey sauter_poon
    .help = "Outlier rejection algorithm. If auto is selected, the algorithm is"
            "chosen automatically"
    .type = choice

  minimum_number_of_reflections = 20
    .help = "The minimum number of input observations per outlier rejection"
            "job below which all reflections in the job will be rejected as"
            "potential outliers."
    .type = int(value_min=0)

  separate_experiments = True
    .help = "If true, outlier rejection will be performed on each experiment"
            "separately. Otherwise, the data from all experiments will be"
            "combined for outlier rejection."
    .type = bool

  separate_panels = Auto
    .help = "If true, outlier rejection will be performed separately for each"
            "panel of a multi-panel detector model. Otherwise data from across"
            "all panels will be combined for outlier rejection."
    .type = bool

  separate_blocks = True
    .help = "If true, for scans outlier rejection will be performed separately"
            "in equal-width blocks of phi, controlled by the parameter"
            "outlier.block_width."
    .type = bool

  block_width = Auto
    .help = "If separate_blocks, a scan will be divided into equal-sized blocks"
            "with width (in degrees) close to this value for outlier rejection."
            "If Auto, a width of at least 18 degrees will be determined,"
            "such that each block contains enough reflections to perform"
            "outlier rejection."
    .type = float(value_min=1.0)

  tukey
    .help = "Options for the tukey outlier rejector"
    .expert_level = 1
  {
    iqr_multiplier = 1.5
      .help = "The IQR multiplier used to detect outliers. A value of 1.5"
              "gives Tukey's rule for outlier detection"
      .type = float(value_min = 0.)
  }

  mcd
    .help = "Options for the mcd outlier rejector, which uses an algorithm"
            "based on FAST-MCD by Rousseeuw and van Driessen. See"
            "doi.org/10.1080/00401706.1999.10485670."
    .expert_level = 1
  {
     alpha = 0.5
       .help = "Decimal fraction controlling the size of subsets over which the"
               "covariance matrix determinant is minimised."
       .type = float(value_min = 0., value_max = 1.0)

     max_n_groups=5
       .help = "The maximum number of groups to split the dataset into if the"
               "dataset is 'large' (more observations than twice the"
               "min_group_size)."
       .type = int(value_min = 1)

     min_group_size=300
       .help = "The smallest sub-dataset size when splitting the dataset into"
               "a number of groups, maximally max_n_groups."
       .type = int(value_min = 100)

     n_trials=500
       .help = "The number of samples used for initial estimates to seed the"
               "search within each sub-dataset."
       .type = int(value_min = 1)

     k1=2
       .help = "The number of concentration steps to take after initial estimates."
       .type = int(value_min = 1)

     k2=2
       .help = "If the dataset is 'large', the number of concentration steps to"
               "take after applying the best subset estimates to the merged"
               "group."
       .type = int(value_min = 1)

     k3=100
       .help = "If the dataset is 'small', the number of concentration steps to"
               "take after selecting the best of the initial estimates, applied"
               "to the whole dataset."
       .type = int(value_min = 1)

     threshold_probability=0.975
       .help = "Quantile probability from the Chi-squared distribution with"
               "number of degrees of freedom equal to the number of dimensions"
               "of the data data (e.g. 3 for X, Y and Phi residuals)."
               "Observations whose robust Mahalanobis distances are larger than"
               "the obtained quantile will be flagged as outliers."
       .type = float(value_min = 0., value_max = 1.0)
  }

  sauter_poon
    .help = "Options for the outlier rejector described in Sauter & Poon (2010)"
            "(https://doi.org/10.1107/S0021889810010782)"
    .expert_level = 1
  {
    px_sz = Auto
      .help = "X, Y pixel size in mm. If Auto, this will be taken from"
              "the first panel of the first experiment."
      .type = floats(size = 2, value_min = 0.001)

    verbose = False
      .help = "Verbose output."
      .type = bool
      .multiple = False

    pdf=None
      .help = "Output file name for making graphs of |dr| vs spot number and dy vs dx."
      .type = str
      .multiple = False
  }
}

'''

phil_scope = parse(phil_str)

class CentroidOutlierFactory(object):

  @classmethod
  def from_parameters_and_colnames(cls, params, colnames, verbosity=0):

    # id the relevant scope for the requested method
    method = params.outlier.algorithm
    if method == "null": return None
    elif method == "tukey":
      from dials.algorithms.refinement.outlier_detection.tukey import Tukey as outlier_detector
      algo_params = params.outlier.tukey
    elif method == "mcd":
      from dials.algorithms.refinement.outlier_detection.mcd import MCD as outlier_detector
      algo_params = params.outlier.mcd
    elif method == "sauter_poon":
      from dials.algorithms.refinement.outlier_detection.sauter_poon import SauterPoon as outlier_detector
      algo_params = params.outlier.sauter_poon
    else:
      raise RuntimeError("outlier.algorithm not recognised")

    # construct kwargs from the algo_params scope
    kwargs = dict((k, v) for k, v in algo_params.__dict__.items() \
      if not k.startswith('_'))

    if not params.outlier.separate_blocks:
      params.outlier.block_width = None
    od = outlier_detector(
      cols=colnames,
      min_num_obs=params.outlier.minimum_number_of_reflections,
      separate_experiments=params.outlier.separate_experiments,
      separate_panels=params.outlier.separate_panels,
      block_width=params.outlier.block_width,
      **kwargs)
    od.set_verbosity(verbosity)
    return od

if __name__ == "__main__":

  # test construction
  params=phil_scope.extract()
  params.outlier.algorithm="tukey"
  print(CentroidOutlierFactory.from_parameters_and_colnames(params, [1, 2, 3]))
  params.outlier.algorithm="mcd"
  print(CentroidOutlierFactory.from_parameters_and_colnames(params, [1, 2, 3]))
