/*
 * observation.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dials/model/data/observation.h>

namespace dials { namespace model { namespace boost_python {

  using namespace boost::python;
  using scitbx::vec2;
  using scitbx::vec3;

  static
  vec3<double> position_data_get_position(const Position::PositionData &obj) {
    return obj.position;
  }

  static
  void position_data_set_position(Position::PositionData &obj, vec3<double> v) {
    obj.position = v;
  }

  static
  vec3<double> position_data_get_variance(const Position::PositionData &obj) {
    return obj.variance;
  }

  static
  void position_data_set_variance(Position::PositionData &obj, vec3<double> v) {
    obj.variance = v;
  }
  
  static
  vec3<double> position_data_get_std_err_sq(const Position::PositionData &obj) {
    return obj.std_err_sq;
  }
 
  static
  void position_data_set_std_err_sq(Position::PositionData &obj, vec3<double> v) {
    obj.std_err_sq = v;
  }
  
  static
  vec2<double> position_get_px_coord(const Position &obj) {
    return vec2<double>(obj.px.position[0], obj.px.position[1]);
  }
  
  static
  void position_set_px_coord(Position &obj, vec2<double> v) {
    obj.px.position[0] = v[0];
    obj.px.position[1] = v[1];
  }
  
  static
  double position_get_frame(const Position &obj) {
    return obj.px.position[2];
  }
  
  static
  void position_set_frame(Position &obj, double v) {
    obj.px.position[2] = v;
  }
  
  static
  vec2<double> position_get_mm_coord(const Position &obj) {
    return vec2<double>(obj.mm.position[0], obj.mm.position[1]);
  }
  
  static
  void position_set_mm_coord(Position &obj, vec2<double> v) {
    obj.mm.position[0] = v[0];
    obj.mm.position[1] = v[1];
  }
  
  static
  double position_get_angle(const Position &obj) {
    return obj.mm.position[2];
  }
  
  static
  void position_set_angle(Position &obj, double v) {
    obj.mm.position[2] = v;
  }
  
  void export_observation()
  {
    class_<Intensity::IntensityData>("IntensityData")
      .def(init<double, double>((
        arg("value"),
        arg("variance"))))
      .def_readwrite("value", &Intensity::IntensityData::value)
      .def_readwrite("variance", &Intensity::IntensityData::variance);
    
    class_<Intensity>("Intensity")
      .def(init<double, double>((
        arg("observed_value"),
        arg("observed_variance"))))
      .def(init<double, double, double, double>((
        arg("observed_value"),
        arg("observed_variance"),
        arg("corrected_value"),
        arg("corrected_variance"))))
      .def(init<const Intensity::IntensityData&>((
        arg("observed"))))
      .def(init<
          const Intensity::IntensityData&, 
          const Intensity::IntensityData&>((
        arg("observed"),
        arg("corrected"))))
      .def_readwrite("observed", &Intensity::observed)
      .def_readwrite("corrected", &Intensity::corrected);

    class_<Position::PositionData>("PositionData")
      .def(init<vec3<double>, vec3<double>, vec3<double> >((
        arg("position"),
        arg("variance"),
        arg("std_err_sq"))))
      .add_property("position", 
        &position_data_get_position,
        &position_data_set_position)
      .add_property("variance", 
        &position_data_get_variance,
        &position_data_set_variance)
      .add_property("std_err_sq",
        &position_data_get_std_err_sq,
        &position_data_set_std_err_sq);
      
    class_<Position>("Position")
      .def(init<vec3<double>, vec3<double>, vec3<double> >((
        arg("px_position"),
        arg("px_variance"),
        arg("px_std_err_sq"))))
      .def(init<vec3<double>, vec3<double>, vec3<double>,
                vec3<double>, vec3<double>, vec3<double> >((
        arg("px_position"),
        arg("px_variance"),
        arg("px_std_err_sq"),
        arg("mm_position"),
        arg("mm_variance"),
        arg("mm_std_err_sq"))))   
      .def(init<const Position::PositionData&>((
        arg("px"))))
      .def(init<const Position::PositionData&, const Position::PositionData&>((
        arg("px"),
        arg("mm"))))    
      .def_readwrite("px", &Position::px)
      .def_readwrite("mm", &Position::mm)
      .add_property("px_coord",
        &position_get_px_coord,
        &position_set_px_coord)
      .add_property("frame",
        &position_get_frame,
        &position_set_frame)
      .add_property("mm_coord",
        &position_get_mm_coord,
        &position_set_mm_coord)
      .add_property("angle",
        &position_get_angle,
        &position_set_angle);
  
    class_<Observation>("Observation")
      .def(init<const Position&>((
        arg("centroid"))))
      .def(init<const Intensity&>((
        arg("intensity"))))
      .def(init<const Position&, const Intensity&>((
        arg("centroid"),
        arg("intensity"))))
      .def_readwrite("centroid", &Observation::centroid)
      .def_readwrite("intensity", &Observation::intensity);
  }

}}} // namespace dials::model::boost_python
