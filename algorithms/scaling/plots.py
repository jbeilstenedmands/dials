"""
Make plotly plots for html output by dials.scale, dials.report or xia2.report.
"""
import math as pymath
import numpy as np
from scitbx import math as scitbxmath
from scitbx.math import distributions
from dials.array_family import flex
from dials.algorithms.scaling.model.model import PhysicalScalingModel

def plot_scaling_models(scaling_model_dict):
  d = {}
  if scaling_model_dict['__id__'] == 'physical':
    model = PhysicalScalingModel.from_dict(scaling_model_dict)
    d.update(_plot_smooth_scales(model))
    if 'absorption' in model.components:
      d.update(plot_absorption_surface(model))
  return d

def _get_smooth_plotting_data_from_model(physical_model, component='scale'):
  """Return a tuple of phis, parameters, parameter esds,
  sample positions for plotting and sample scale values."""
  configdict = physical_model.configdict
  valid_osc = configdict['valid_osc_range']
  sample_values = flex.double(np.linspace(valid_osc[0], valid_osc[1],
    ((valid_osc[1]-valid_osc[0])/0.1)+1, endpoint=True)) # Make a grid of
    #points with 10 points per degree.
  if component == 'scale':
    if 'scale' in configdict['corrections']:
      scale_SF = physical_model.components['scale']
      norm_vals = sample_values / physical_model.configdict['scale_rot_interval']
      scale_SF.data = {'x' : norm_vals}
      scale_SF.update_reflection_data()
      s = scale_SF.calculate_scales()
      smoother_phis = [(i * configdict['scale_rot_interval']) + valid_osc[0]
        for i in scale_SF.smoother.positions()]
    return smoother_phis, scale_SF.parameters, scale_SF.parameter_esds, sample_values, s
  if component == 'decay':
    if 'decay' in configdict['corrections']:
      decay_SF = physical_model.components['decay']
      norm_vals = sample_values / physical_model.configdict['decay_rot_interval']
      decay_SF.data = {'x' : norm_vals, 'd' : flex.double(norm_vals.size(), 1.0)}
      decay_SF.update_reflection_data()
      s = decay_SF.calculate_scales()
      smoother_phis = [(i * configdict['decay_rot_interval']) + valid_osc[0]
        for i in decay_SF._smoother.positions()]
    return smoother_phis, decay_SF.parameters, decay_SF.parameter_esds, sample_values, s

def _plot_smooth_scales(physical_model):
  """Plot smooth scale factors for the physical model."""

  d = {'smooth_scale_model': {
    'data' : [],
    'layout' : {
      'title' : 'Smoothly varying corrections',
      'xaxis' : {
        'domain' :[0, 1],
        'anchor' : 'y',
        'title' : 'rotation angle (degrees)'},
      'yaxis' : {
        'domain' :[0, 0.45],
        'anchor' : 'x',
        'title' : 'relative B-factor'},
      'yaxis2' : {
        'domain' :[0.5, 0.95],
        'anchor' : 'x',
        'title' : 'inverse <br> scale factor'}
      }
    }
  }

  data = []

  if 'scale' in physical_model.components:
    smoother_phis, parameters, parameter_esds, sample_values, sample_scales = \
      _get_smooth_plotting_data_from_model(physical_model, component='scale')

    data.append({
      'x' : list(sample_values),
      'y' : list(sample_scales),
      'type' : 'line',
      'name' : 'smooth scale term',
      'xaxis' : 'x',
      'yaxis' : 'y2'
    })
    data.append({
      'x' : list(smoother_phis),
      'y' : list(parameters),
      'type' : 'scatter',
      'mode' : 'markers',
      'name' : 'scale term parameters',
      'xaxis' : 'x',
      'yaxis' : 'y2'
    })
    if parameter_esds:
      data[-1]['error_y'] = {'type':'data', 'array':list(parameter_esds)}

  if 'decay' in physical_model.components:
    smoother_phis, parameters, parameter_esds, sample_values, sample_scales = \
      _get_smooth_plotting_data_from_model(physical_model, component='decay')

    data.append({
      'x' : list(sample_values),
      'y' : list(np.log(sample_scales)*2.0),
      'type' : 'line',
      'name' : 'smooth decay term',
      'xaxis' : 'x',
      'yaxis' : 'y'
    })
    data.append({
      'x' : list(smoother_phis),
      'y' : list(parameters),
      'type' : 'scatter',
      'mode' : 'markers',
      'name' : 'decay term parameters',
      'xaxis' : 'x',
      'yaxis' : 'y'
    })
    if parameter_esds:
      data[-1]['error_y'] = {'type':'data', 'array':list(parameter_esds)}
  d['smooth_scale_model']['data'].extend(data)
  return d

def plot_absorption_surface(physical_model):
    """Plot an absorption surface for a physical scaling model."""

    d = {'absorption_surface': {
      'data' : [],
      'layout' : {
        'title' : 'Absorption correction surface',
        'xaxis' : {
          'domain' :[0, 1],
          'anchor' : 'y',
          'title' : 'theta (degrees)'},
        'yaxis' : {
          'domain' :[0, 1],
          'anchor' : 'x',
          'title' : 'phi (degrees)'}
        }
      }
    }

    params = physical_model.components['absorption'].parameters

    order = int(-1.0 + ((1.0 + len(params))**0.5))
    lfg = scitbxmath.log_factorial_generator(2 * order + 1)
    STEPS = 50
    phi = np.linspace(0, 2 * np.pi, 2*STEPS)
    theta = np.linspace(0, np.pi, STEPS)
    THETA, _ = np.meshgrid(theta, phi)
    lmax = int(-1.0 + ((1.0 + len(params))**0.5))
    Intensity = np.ones(THETA.shape)
    counter = 0
    sqrt2 = pymath.sqrt(2)
    nsssphe = scitbxmath.nss_spherical_harmonics(order, 50000, lfg)
    for l in range(1, lmax+1):
      for m in range(-l, l+1):
        for it, t in enumerate(theta):
          for ip, p in enumerate(phi):
            Ylm = nsssphe.spherical_harmonic(l, abs(m), t, p)
            if m < 0:
              r = sqrt2 * ((-1) ** m) * Ylm.imag
            elif m == 0:
              assert Ylm.imag == 0.0
              r = Ylm.real
            else:
              r = sqrt2 * ((-1) ** m) * Ylm.real
            Intensity[ip, it] += params[counter] * r
        counter += 1
    d['absorption_surface']['data'].append({
      'x' : list(theta*180.0/np.pi),
      'y' : list(phi*180.0/np.pi),
      'z' : list(Intensity.T.tolist()),
      'type' : 'heatmap',
      'colorscale' : 'Viridis',
      'colorbar' : {'title' : 'inverse <br>scale factor'},
      'name' : 'absorption surface',
      'xaxis' : 'x',
      'yaxis' : 'y'
    })
    return d

def statistics_tables(dataset_statistics):
  result = dataset_statistics
  resolution_binned_table = [('d_max', 'd_min', 'n_obs', 'n_uniq', 'mult', 'comp',
    '&ltI&gt', '&ltI/sI&gt', 'r_merge', 'r_meas', 'r_pim', 'cc1/2', 'cc_anom')]
  for bin_stats in result.bins:
    resolution_binned_table.append(tuple(bin_stats.format().split()))
  result = result.overall
  summary_table = [('Resolution',
    '{0:.3f} - {1:.3f}'.format(result.d_max, result.d_min)),
    ('Observations', result.n_obs), ('Unique Reflections', result.n_uniq),
    ('Redundancy', '{:.2f}'.format(result.mean_redundancy)),
    ('Completeness', '{:.2f}'.format(result.completeness*100)),
    ('Mean intensity', '{:.1f}'.format(result.i_mean)),
    ('Mean I/sigma(I)', '{:.1f}'.format(result.i_over_sigma_mean)),
    ('R-merge', '{:.4f}'.format(result.r_merge)),
    ('R-meas', '{:.4f}'.format(result.r_meas)),
    ('R-pim', '{:.4f}'.format(result.r_pim))]
  return (summary_table, resolution_binned_table)

def plot_outliers(data):
  '''plots positions of outliers'''

  if not data['z']:
    return {'outlier_xy_positions' : {}, 'outliers_vs_z' : {}}

  hist = flex.histogram(flex.double(data['z']), n_slots=min(100, int(len(data['z'])*10)))

  d = {'outlier_xy_positions': {
      'data' : [{
        'x' : data['x'],
        'y' : data['y'],
        'type' : 'scatter',
        'mode' : 'markers',
        'xaxis' : 'x',
        'yaxis' : 'y',
      }],
      'layout' : {
        'title' : 'Outlier x-y positions',
        'xaxis' : {
          'anchor' : 'y',
          'title' : 'x (px)',
          'range' : [0, data['image_size'][0]]},
        'yaxis' : {
          'anchor' : 'x',
          'title' : 'y (px)',
          'range' : [0, data['image_size'][1]]}
      }
    },
    'outliers_vs_z': {
      'data' : [{
        'x': list(hist.slot_centers()),
        'y': list(hist.slots()),
        'type': 'bar',
        'name': 'outliers vs rotation',
      }],
      'layout': {
        'title': 'Outlier distribution across frames',
        'xaxis': {'title': 'frame'},
        'yaxis': {'title': 'count'},
        'bargap': 0,
      }
    }
  }

  return d

def normal_probability_plot(data):
  ##FIXME this plot become very large/slow for a big dataset - can
  ##we make this just a static image?
  norm = distributions.normal_distribution()

  n = len(data['delta_hl'])
  if n <= 10:
    a = 3/8
  else:
    a = 0.5

  sorted_data = flex.sorted(flex.double(data['delta_hl']))
  rankits = [norm.quantile((i+1-a)/(n+1-(2*a))) for i in xrange(n)]

  d = {'normal_distribution_plot': {
      'data' : [{
        'x' : rankits,
        'y' : list(sorted_data),
        'type' : 'scatter',
        'mode' : 'markers',
        'xaxis' : 'x',
        'yaxis' : 'y',
        'name' : 'normalised deviations'
      },
      {
          'x'   : [-5, 5],
          'y'   : [-5, 5],
          'type': 'scatter',
          'mode': 'lines',
          'name': 'z = m',
      }],
      'layout' : {
        'title' : 'Normal probability plot with error model applied',
        'xaxis' : {
          'anchor' : 'y',
          'title' : 'Order statistic medians, m'},
        'yaxis' : {
          'anchor' : 'x',
          'title' : 'Ordered responses, z'}
      }
    }
  }

  return d
