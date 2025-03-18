/*
 * processor.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_INTEGRATION_PROCESSOR_H
#define DIALS_ALGORITHMS_INTEGRATION_PROCESSOR_H
#include <scitbx/vec3.h>
#include <scitbx/vec2.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <list>
#include <vector>
#include <ctime>
#include <dials/model/data/image.h>
#include <dials/model/data/shoebox.h>
#include <dials/array_family/reflection_table.h>
#include <dxtbx/array_family/flex_table_suite.h>
#include <dials/array_family/boost_python/reflection_table_suite.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/scan.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>
#include <iomanip>
namespace dials { namespace algorithms {

  using dials::model::Foreground;
  using model::Image;
  using model::Shoebox;
  using model::Valid;

  /**
   * The cctbx build system is too messed up to figure out how to build
   * boost::system need by boost::chrono. Therefore just use system clock
   * to get a timestamp in ms
   */
  double timestamp() {
    return ((double)clock()) / ((double)CLOCKS_PER_SEC);
  }

  /**
   * A base class for executor callbacks
   */
  class Executor {
  public:
    virtual void process(int, af::reflection_table) = 0;
  };

  class ShoeboxProcessorV2 {
  public:
    ShoeboxProcessorV2(af::reflection_table data,
                       std::size_t npanels,
                       int frame0,
                       int frame1,
                       bool save,
                       const dxtbx::model::Scan& scan,
                       const dxtbx::model::BeamBase& beam,
                       const dxtbx::model::Goniometer& gonio,
                       const dxtbx::model::Detector& detector,
                       const double delta_b,
                       const double delta_m)
        : data_(data),
          extract_time_(0.0),
          process_time_(0.0),
          save_(save),
          npanels_(npanels),
          frame0_(frame0),
          frame1_(frame1),
          frame_(frame0),
          nframes_(frame1 - frame0),
          phi0_(scan.get_oscillation()[0]),
          dphi_(scan.get_oscillation()[1]),
          s0_(beam.get_s0()),
          m2_(gonio.get_rotation_axis()),
          detector_(detector),
          index0_(scan.get_array_range()[0]),
          index1_(scan.get_array_range()[1]) {
      delta_b_r2 = 1.0 / (std::pow(delta_b, 2));
      delta_m_r2 = 1.0 / (std::pow(delta_m, 2));
      DIALS_ASSERT(frame0_ < frame1_);
      DIALS_ASSERT(npanels_ > 0);
      // DIALS_ASSERT(data.is_consistent());
      DIALS_ASSERT(data.contains("shoebox"));
      DIALS_ASSERT(data.size() > 0);
      af::const_ref<Shoebox<>> shoebox = data["shoebox"];
      // af::shared<int> total_intensity(data.size());
      // data["intensity_sum_value"] = total_intensity;
      std::size_t size = nframes_ * npanels_;
      std::vector<std::size_t> num(size, 0);
      std::vector<std::size_t> count(size, 0);
      flatten_ = shoebox[0].flat;
      for (std::size_t i = 0; i < shoebox.size(); ++i) {
        DIALS_ASSERT(shoebox[i].flat == flatten_);
        // DIALS_ASSERT(shoebox[i].is_allocated() == false);
        DIALS_ASSERT(shoebox[i].bbox[1] > shoebox[i].bbox[0]);
        DIALS_ASSERT(shoebox[i].bbox[3] > shoebox[i].bbox[2]);
        DIALS_ASSERT(shoebox[i].bbox[5] > shoebox[i].bbox[4]);
        for (int z = shoebox[i].bbox[4]; z < shoebox[i].bbox[5]; ++z) {
          std::size_t j = shoebox[i].panel + (z - frame0_) * npanels_;
          /*if (j >= num.size()) {
            std::cout << j << " " << num.size() << std::endl;
          }*/
          DIALS_ASSERT(j < num.size());
          num[j]++;
        }
      }
      offset_.push_back(0);
      std::partial_sum(num.begin(), num.end(), std::back_inserter(offset_));
      indices_.resize(offset_.back());
      for (std::size_t i = 0; i < shoebox.size(); ++i) {
        for (int z = shoebox[i].bbox[4]; z < shoebox[i].bbox[5]; ++z) {
          std::size_t j = shoebox[i].panel + (z - frame0_) * npanels_;
          std::size_t k = offset_[j] + count[j];
          DIALS_ASSERT(j < count.size());
          DIALS_ASSERT(k < indices_.size());
          indices_[k] = i;
          count[j]++;
        }
      }
      DIALS_ASSERT(count == num);
    }

    template <typename T>
    void next(const Image<T>& image) {
      using dials::af::boost_python::reflection_table_suite::select_rows_index;
      using dxtbx::af::flex_table_suite::set_selected_rows_index;
      typedef Shoebox<>::float_type float_type;
      typedef af::ref<float_type, af::c_grid<3>> sbox_data_type;
      typedef af::ref<int, af::c_grid<3>> sbox_mask_type;
      DIALS_ASSERT(frame_ >= frame0_ && frame_ < frame1_);
      DIALS_ASSERT(image.npanels() == npanels_);

      // Get the initial time
      double start_time = timestamp();

      // For each image, extract shoeboxes of reflections recorded.
      // Allocate data where necessary
      af::ref<Shoebox<>> shoebox = data_["shoebox"];
      af::ref<vec3<double>> s1_vec = data_["s1"];
      af::ref<vec3<double>> xyzcal_px = data_["xyzcal.px"];
      double s0_length = s0_.length();

      // af::shared<std::size_t> process_indices;
      for (std::size_t p = 0; p < image.npanels(); ++p) {
        af::const_ref<std::size_t> ind = indices(frame_, p);
        af::const_ref<T, af::c_grid<2>> data = image.data(p);
        af::const_ref<bool, af::c_grid<2>> mask = image.mask(p);
        DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
        for (std::size_t i = 0; i < ind.size(); ++i) {
          DIALS_ASSERT(ind[i] < shoebox.size());
          Shoebox<>& sbox = shoebox[ind[i]];
          /*if (frame_ == sbox.bbox[4]) {
            DIALS_ASSERT(sbox.is_allocated() == false);
            sbox.allocate();
          }*/ // Don't allocate
          int6 b = sbox.bbox;
          // int sbox_intensity = 0;
          //  sbox_data_type sdata = sbox.data.ref();
          //sbox_mask_type smask = sbox.mask.ref();
          DIALS_ASSERT(b[1] > b[0]);
          DIALS_ASSERT(b[3] > b[2]);
          DIALS_ASSERT(b[5] > b[4]);
          DIALS_ASSERT(frame_ >= b[4] && frame_ < b[5]);
          int x0 = b[0];
          int x1 = b[1];
          int y0 = b[2];
          int y1 = b[3];
          int z0 = b[4];
          int xs = x1 - x0;
          int ys = y1 - y0;
          int z = frame_ - z0;
          int yi = (int)data.accessor()[0];
          int xi = (int)data.accessor()[1];
          int xb = x0 >= 0 ? 0 : std::abs(x0);
          int yb = y0 >= 0 ? 0 : std::abs(y0);
          int xe = x1 <= xi ? xs : xs - (x1 - xi);
          int ye = y1 <= yi ? ys : ys - (y1 - yi);
          /*if (yb >= ye || xb >= xe) {
            continue;
          }*/
          DIALS_ASSERT(yb >= 0 && ye <= ys);
          DIALS_ASSERT(xb >= 0 && xe <= xs);
          DIALS_ASSERT(yb + y0 >= 0 && ye + y0 <= yi);
          DIALS_ASSERT(xb + x0 >= 0 && xe + x0 <= xi);
          // DIALS_ASSERT(sbox.is_consistent());
          /*if (flatten_) {
            for (std::size_t y = yb; y < ye; ++y) {
              for (std::size_t x = xb; x < xe; ++x) {
                sdata(0, y, x) += data(y + y0, x + x0);
                bool sv = smask(0, y, x) & Valid;
                bool mv = mask(y + y0, x + x0);
                smask(0, y, x) = (mv && (z == 0 ? true : sv) ? Valid : 0);
              }
            }
          }*/
          // std::cout << "Processing " << b[0] << " " << b[1] << " " << b[2] << " " <<
          // b[3] << " " << b[4] << " " << b[5] <<std::endl; std::cout << frame_ << " "
          // << frame0_ << std::endl;
          const dxtbx::model::Panel& panel = detector_[p];
          vec3<double> s1 = s1_vec[ind[i]];
          vec3<double> xyzcal = xyzcal_px[ind[i]];
          double phi = phi0_ + (xyzcal[2] - index0_) * dphi_;
          profile_model::gaussian_rs::CoordinateSystem cs(m2_, s0_, s1, phi);
          vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1);
          double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
          bool print_out = false;
          if ((b[0] == 213) && (b[1] ==227) && (b[2] == 1219) && (b[3] == 1229) &&(b[4] == 284) && (b[5] == 626)){
            std::cout << "attenuation length ,phi " << attenuation_length << " " << phi << std::endl;
            //print_out = true;
          }
          af::versa<double, af::c_grid<3>> dxyz_array(af::c_grid<3>(2, ys + 1, xs + 1));
          for (std::size_t k = 0; k < 2; ++k) {
            int j = 0;
            for (std::size_t y = yb; y <= ye; ++y, ++j) {
              int m = 0;
              for (std::size_t x = xb; x <= xe; ++x, ++m) {
                // double x = x0 + i;  // + 0.5;
                // double y = y0 + j;  // + 0.5;
                //  int z = z0 + k;
                vec3<double> s1dash =
                  panel
                    .get_pixel_lab_coord(vec2<double>(x + x0, y + y0),
                                         attenuation_length)
                    .normalize()
                  * s0_length;
                /*if (print_out){
                  std::cout << "s1dash" << std::endl;
                  std::cout << std::setprecision(12) << s1dash[0] << " " <<s1dash[1] << " " <<s1dash[2] << std::endl;
                }*/
                // nned to get epsilon 1.
                // s1_dash = box.beam_vectors
                // double phidash = phi0_ + (z0 + k - frame0_) * dphi_;
                double phidash = phi0_ + (frame_ + k - frame0_) * dphi_;
                vec3<double> epsilon_coords = cs.coords_from_s1vector(s1dash, phidash);
                dxyz_array(k, j, m) =
                  ((epsilon_coords[0] * epsilon_coords[0]
                    + epsilon_coords[1] * epsilon_coords[1])
                   * delta_b_r2)
                  + ((epsilon_coords[2] * epsilon_coords[2]) * delta_m_r2);
                if (print_out){
                  std::cout << std::setprecision(12) << dxyz_array(k, j, m) << std::endl;
                }
                // int mask_value = (d <= 1.0) ? Foreground : Background;
                // mask(k, j, i) |= mask_value;
              }
            }
            /*    //for (int j = 0; j <= ys; ++j) {
                //for (int i = 0; i <= xs; ++i) {
                double x = x0 + i;  // + 0.5;
                double y = y0 + j;  // + 0.5;
                // int z = z0 + k;
                vec3<double> s1dash =
                  panel.get_pixel_lab_coord(vec2<double>(x, y), attenuation_length)
                    .normalize()
                  * s0_length;
                // nned to get epsilon 1.
                // s1_dash = box.beam_vectors
                //double phidash = phi0_ + (z0 + k - frame0_) * dphi_;
                double phidash = phi0_ + (frame_ + k - frame0_) * dphi_;
                vec3<double> epsilon_coords = cs.coords_from_s1vector(s1dash, phidash);
                dxyz_array(k, j, i) =
                  ((epsilon_coords[0] * epsilon_coords[0]
                    + epsilon_coords[1] * epsilon_coords[1])
                  * delta_b_r2)
                  + ((epsilon_coords[2] * epsilon_coords[2]) * delta_m_r2);
                // int mask_value = (d <= 1.0) ? Foreground : Background;
                // mask(k, j, i) |= mask_value;
              }
            }*/
          }
          int j = 0;
          for (std::size_t y = yb; y < ye; ++y, ++j) {
            int m = 0;
            for (std::size_t x = xb; x < xe; ++x, ++m) {
              
              double d1 = dxyz_array(0, j, m);
              double d2 = dxyz_array(0, j + 1, m);
              double d3 = dxyz_array(0, j, m + 1);
              double d4 = dxyz_array(0, j + 1, m + 1);
              double d5 = dxyz_array(1, j, m);
              double d6 = dxyz_array(1, j + 1, m);
              double d7 = dxyz_array(1, j, m+1);
              double d8 = dxyz_array(1, j + 1, m + 1);
              double d = std::min(std::min(std::min(d1, d2), std::min(d3, d4)),
                                  std::min(std::min(d5, d6), std::min(d7, d8)));
              // std::cout << "d = " << d << std::endl;
              // std::cout << "Coord " << x << " " << y << std::endl;
              if (d <= 1.0) { // Is foreground
                if (mask(y + y0, x + x0)){ // If pixel not masked on image
                  sbox.total_intensity += data(y + y0, x + x0);
                  sbox.n_valid_fg += 1;
                }
                else {
                  sbox.masked_image_pixel = true;
                  sbox.n_invalid_fg += 1;
                }
              } else {
                // to make more efficient, fill in histogram between a min and max?
                // stop once got to n entries?
                if (mask(y + y0, x + x0)){
                  sbox.n_valid_bg += 1;
                  int this_pixel = data(y + y0, x + x0);
                  if (auto search = sbox.background_hist.find(this_pixel);
                      search != sbox.background_hist.end()) {
                    sbox.background_hist[this_pixel] += 1;
                  } else {
                    sbox.background_hist[this_pixel] = 1;
                  }
                }
                else {
                  sbox.n_invalid_bg += 1;
                }
              }
              
              // sbox.total_intensity += data(y + y0, x + x0);
              // std::cout << "Total I " << sbox.total_intensity << std::endl;
              /*if (mask(y + y0, x + x0)) {
                // FIXME add test on foreground/background.
                //if ((smask(z, y, x) & Foreground) == Foreground){
                //sbox.total_intensity += data(y + y0, x + x0);
                //}

              }*/
              // sdata(z, y, x) = data(y + y0, x + x0);
              // smask(z, y, x) = mask(y + y0, x + x0) ? Valid : 0;
            }
          }
          // af::shared<int> total_intensity = data_["intensity_sum_value"];
          // total_intensity[ind[i]] += sbox_intensity;
          // if (frame_ == sbox.bbox[5] - 1) {
          //   process_indices.push_back(ind[i]);
          // }
        }
      }

      // Update timing info
      double end_time = timestamp();
      extract_time_ += end_time - start_time;

      // Process all the reflections and set the reflections
      /*if (process_indices.size() > 0) {
        double start_time = timestamp();
        af::const_ref<std::size_t> ind = process_indices.const_ref();
        af::reflection_table reflections = select_rows_index(data_, ind);
        // For now, just set the total intensity

        //executor.process(frame_, reflections);
        set_selected_rows_index(data_, ind, reflections);
        if (!save_) {
          for (std::size_t i = 0; i < ind.size(); ++i) {
            shoebox[ind[i]].deallocate();
          }
        } // Didn't allocate, so don't need to deallocate.
        double end_time = timestamp();
        process_time_ += end_time - start_time;
      }*/

      // Update the frame counter
      frame_++;
    }

    template <typename T>
    af::shared<int> finalise(af::reflection_table data) {
      af::shared<int> total_intensity(data.size());
      af::const_ref<Shoebox<>> shoebox = data["shoebox"];
      af::shared<bool> success = data["summation_success"];
      af::shared<int> nbg = data["num_pixels.background"];
      af::shared<int> bg_used = data["num_pixels.background_used"];
      af::shared<int> foreground = data["num_pixels.foreground"];
      af::shared<int> valid = data["num_pixels.valid"];
      for (int i = 0; i < data.size(); i++) {
        total_intensity[i] = shoebox[i].total_intensity;
        if (shoebox[i].n_invalid_fg > 0){
          success[i] = false;
        }
        nbg[i] = shoebox[i].n_valid_bg;
        //bg_used[i] = shoebox[i].n_valid_bg;
        foreground[i] = shoebox[i].n_valid_fg;
        valid[i] = shoebox[i].n_valid_bg + shoebox[i].n_valid_fg;

        /*int bg_size = shoebox[i].background_hist.size();
        std::cout << "Number of elements in bg hist: " << bg_size << std::endl;
        std::cout << "Number of elements in shoebox: " << ((shoebox[i].bbox[1] -
        shoebox[i].bbox[0]) *(shoebox[i].bbox[3] - shoebox[i].bbox[2]) *
        (shoebox[i].bbox[5] - shoebox[i].bbox[4]))<< std::endl; int total_bg_count = 0;
        for (auto& it: shoebox[i].background_hist){
          std::cout << "bg " << it.first << " " << it.second << std::endl;
          total_bg_count += it.second;
        }
        std::cout << "Total background: " << total_bg_count<< std::endl;*/
      }
      return total_intensity;
      // data["intensity_sum_value"] = total_intensity;
    }

    template <typename T>
    void next_data_only(const Image<T>& image) {
      using dials::af::boost_python::reflection_table_suite::select_rows_index;
      using dxtbx::af::flex_table_suite::set_selected_rows_index;
      typedef Shoebox<>::float_type float_type;
      typedef af::ref<float_type, af::c_grid<3>> sbox_data_type;
      typedef af::ref<int, af::c_grid<3>> sbox_mask_type;
      DIALS_ASSERT(frame_ >= frame0_ && frame_ < frame1_);
      DIALS_ASSERT(image.npanels() == npanels_);
      next(image);
    }

    /** @returns The first frame.  */
    int frame0() const {
      return frame0_;
    }

    /** @returns The last frame */
    int frame1() const {
      return frame1_;
    }

    /** @returns The current frame. */
    int frame() const {
      return frame_;
    }

    /** @returns The number of frames  */
    std::size_t nframes() const {
      return nframes_;
    }

    /** @returns The number of panels */
    std::size_t npanels() const {
      return npanels_;
    }

    /**
     * @returns Is the extraction finished.
     */
    bool finished() const {
      return frame_ == frame1_;
    }

    /**
     * @returns The extract time
     */
    double extract_time() const {
      return extract_time_;
    }

    /**
     * @returns The process time
     */
    double process_time() const {
      return process_time_;
    }

  private:
    /**
     * Get an index array specifying which reflections are recorded on a given
     * frame and panel.
     * @param frame The frame number
     * @param panel The panel number
     * @returns An array of indices
     */
    af::const_ref<std::size_t> indices(int frame, std::size_t panel) const {
      std::size_t j0 = panel + (frame - frame0_) * npanels_;
      DIALS_ASSERT(offset_.size() > 0);
      DIALS_ASSERT(j0 < offset_.size() - 1);
      std::size_t i0 = offset_[j0];
      std::size_t i1 = offset_[j0 + 1];
      DIALS_ASSERT(i1 >= i0);
      std::size_t off = i0;
      std::size_t num = i1 - off;
      DIALS_ASSERT(off + num <= indices_.size());
      return af::const_ref<std::size_t>(&indices_[off], num);
    }

    af::reflection_table data_;
    double extract_time_;
    double process_time_;
    bool flatten_;
    bool save_;
    std::size_t npanels_;
    int frame0_;
    int frame1_;
    int frame_;
    std::size_t nframes_;
    std::vector<std::size_t> indices_;
    std::vector<std::size_t> offset_;
    double phi0_;
    double dphi_;
    vec3<double> s0_;
    vec3<double> m2_;
    dxtbx::model::Detector detector_;
    double delta_b_r2;
    double delta_m_r2;
    double index0_;
    double index1_;
  };

  /**
   * A class to extract shoebox pixels from images
   */
  class ShoeboxProcessor {
  public:
    /**
     * Initialise the index array. Determine which reflections are recorded on
     * each frame and panel ahead of time to enable quick lookup of the
     * reflections to be written to when processing each image.
     */
    ShoeboxProcessor(af::reflection_table data,
                     std::size_t npanels,
                     int frame0,
                     int frame1,
                     bool save)
        : data_(data),
          extract_time_(0.0),
          process_time_(0.0),
          save_(save),
          npanels_(npanels),
          frame0_(frame0),
          frame1_(frame1),
          frame_(frame0),
          nframes_(frame1 - frame0) {
      DIALS_ASSERT(frame0_ < frame1_);
      DIALS_ASSERT(npanels_ > 0);
      DIALS_ASSERT(data.is_consistent());
      DIALS_ASSERT(data.contains("shoebox"));
      DIALS_ASSERT(data.size() > 0);
      af::const_ref<Shoebox<>> shoebox = data["shoebox"];
      std::size_t size = nframes_ * npanels_;
      std::vector<std::size_t> num(size, 0);
      std::vector<std::size_t> count(size, 0);
      flatten_ = shoebox[0].flat;
      for (std::size_t i = 0; i < shoebox.size(); ++i) {
        DIALS_ASSERT(shoebox[i].flat == flatten_);
        DIALS_ASSERT(shoebox[i].is_allocated() == false);
        DIALS_ASSERT(shoebox[i].bbox[1] > shoebox[i].bbox[0]);
        DIALS_ASSERT(shoebox[i].bbox[3] > shoebox[i].bbox[2]);
        DIALS_ASSERT(shoebox[i].bbox[5] > shoebox[i].bbox[4]);
        for (int z = shoebox[i].bbox[4]; z < shoebox[i].bbox[5]; ++z) {
          std::size_t j = shoebox[i].panel + (z - frame0_) * npanels_;
          DIALS_ASSERT(j < num.size());
          num[j]++;
        }
      }
      offset_.push_back(0);
      std::partial_sum(num.begin(), num.end(), std::back_inserter(offset_));
      indices_.resize(offset_.back());
      for (std::size_t i = 0; i < shoebox.size(); ++i) {
        for (int z = shoebox[i].bbox[4]; z < shoebox[i].bbox[5]; ++z) {
          std::size_t j = shoebox[i].panel + (z - frame0_) * npanels_;
          std::size_t k = offset_[j] + count[j];
          DIALS_ASSERT(j < count.size());
          DIALS_ASSERT(k < indices_.size());
          indices_[k] = i;
          count[j]++;
        }
      }
      DIALS_ASSERT(count == num);
    }

    /**
     * Extract the pixels from the image and copy to the relevant shoeboxes.
     * @param image The image to process
     * @param frame The current image frame
     */
    template <typename T>
    void next(const Image<T>& image, Executor& executor) {
      using dials::af::boost_python::reflection_table_suite::select_rows_index;
      using dxtbx::af::flex_table_suite::set_selected_rows_index;
      typedef Shoebox<>::float_type float_type;
      typedef af::ref<float_type, af::c_grid<3>> sbox_data_type;
      typedef af::ref<int, af::c_grid<3>> sbox_mask_type;
      DIALS_ASSERT(frame_ >= frame0_ && frame_ < frame1_);
      DIALS_ASSERT(image.npanels() == npanels_);

      // Get the initial time
      double start_time = timestamp();

      // For each image, extract shoeboxes of reflections recorded.
      // Allocate data where necessary
      af::ref<Shoebox<>> shoebox = data_["shoebox"];
      af::shared<std::size_t> process_indices;
      for (std::size_t p = 0; p < image.npanels(); ++p) {
        af::const_ref<std::size_t> ind = indices(frame_, p);
        af::const_ref<T, af::c_grid<2>> data = image.data(p);
        af::const_ref<bool, af::c_grid<2>> mask = image.mask(p);
        DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
        for (std::size_t i = 0; i < ind.size(); ++i) {
          DIALS_ASSERT(ind[i] < shoebox.size());
          Shoebox<>& sbox = shoebox[ind[i]];
          if (frame_ == sbox.bbox[4]) {
            DIALS_ASSERT(sbox.is_allocated() == false);
            sbox.allocate();
          }
          int6 b = sbox.bbox;
          sbox_data_type sdata = sbox.data.ref();
          sbox_mask_type smask = sbox.mask.ref();
          DIALS_ASSERT(b[1] > b[0]);
          DIALS_ASSERT(b[3] > b[2]);
          DIALS_ASSERT(b[5] > b[4]);
          DIALS_ASSERT(frame_ >= b[4] && frame_ < b[5]);
          int x0 = b[0];
          int x1 = b[1];
          int y0 = b[2];
          int y1 = b[3];
          int z0 = b[4];
          int xs = x1 - x0;
          int ys = y1 - y0;
          int z = frame_ - z0;
          int yi = (int)data.accessor()[0];
          int xi = (int)data.accessor()[1];
          int xb = x0 >= 0 ? 0 : std::abs(x0);
          int yb = y0 >= 0 ? 0 : std::abs(y0);
          int xe = x1 <= xi ? xs : xs - (x1 - xi);
          int ye = y1 <= yi ? ys : ys - (y1 - yi);
          if (yb >= ye || xb >= xe) {
            continue;
          }
          DIALS_ASSERT(yb >= 0 && ye <= ys);
          DIALS_ASSERT(xb >= 0 && xe <= xs);
          DIALS_ASSERT(yb + y0 >= 0 && ye + y0 <= yi);
          DIALS_ASSERT(xb + x0 >= 0 && xe + x0 <= xi);
          DIALS_ASSERT(sbox.is_consistent());
          if (flatten_) {
            for (std::size_t y = yb; y < ye; ++y) {
              for (std::size_t x = xb; x < xe; ++x) {
                sdata(0, y, x) += data(y + y0, x + x0);
                bool sv = smask(0, y, x) & Valid;
                bool mv = mask(y + y0, x + x0);
                smask(0, y, x) = (mv && (z == 0 ? true : sv) ? Valid : 0);
              }
            }
          } else {
            for (std::size_t y = yb; y < ye; ++y) {
              for (std::size_t x = xb; x < xe; ++x) {
                sdata(z, y, x) = data(y + y0, x + x0);
                smask(z, y, x) = mask(y + y0, x + x0) ? Valid : 0;
              }
            }
          }
          if (frame_ == sbox.bbox[5] - 1) {
            process_indices.push_back(ind[i]);
          }
        }
      }

      // Update timing info
      double end_time = timestamp();
      extract_time_ += end_time - start_time;

      // Process all the reflections and set the reflections
      if (process_indices.size() > 0) {
        double start_time = timestamp();
        af::const_ref<std::size_t> ind = process_indices.const_ref();
        af::reflection_table reflections = select_rows_index(data_, ind);
        executor.process(frame_, reflections);
        set_selected_rows_index(data_, ind, reflections);
        if (!save_) {
          for (std::size_t i = 0; i < ind.size(); ++i) {
            shoebox[ind[i]].deallocate();
          }
        }
        double end_time = timestamp();
        process_time_ += end_time - start_time;
      }

      // Update the frame counter
      frame_++;
    }

    /**
     * Extract the pixels from the image and copy to the relevant shoeboxes.
     * @param image The image to process
     * @param frame The current image frame
     */
    template <typename T>
    void next_data_only(const Image<T>& image) {
      using dials::af::boost_python::reflection_table_suite::select_rows_index;
      using dxtbx::af::flex_table_suite::set_selected_rows_index;
      typedef Shoebox<>::float_type float_type;
      typedef af::ref<float_type, af::c_grid<3>> sbox_data_type;
      typedef af::ref<int, af::c_grid<3>> sbox_mask_type;
      DIALS_ASSERT(frame_ >= frame0_ && frame_ < frame1_);
      DIALS_ASSERT(image.npanels() == npanels_);

      // Get the initial time
      double start_time = timestamp();

      // For each image, extract shoeboxes of reflections recorded.
      // Allocate data where necessary
      af::ref<Shoebox<>> shoebox = data_["shoebox"];
      af::shared<std::size_t> process_indices;
      for (std::size_t p = 0; p < image.npanels(); ++p) {
        af::const_ref<std::size_t> ind = indices(frame_, p);
        af::const_ref<T, af::c_grid<2>> data = image.data(p);
        af::const_ref<bool, af::c_grid<2>> mask = image.mask(p);
        DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
        for (std::size_t i = 0; i < ind.size(); ++i) {
          DIALS_ASSERT(ind[i] < shoebox.size());
          Shoebox<>& sbox = shoebox[ind[i]];
          if (frame_ == sbox.bbox[4]) {
            DIALS_ASSERT(sbox.is_allocated() == false);
            sbox.allocate();
          }
          int6 b = sbox.bbox;
          sbox_data_type sdata = sbox.data.ref();
          sbox_mask_type smask = sbox.mask.ref();
          DIALS_ASSERT(b[1] > b[0]);
          DIALS_ASSERT(b[3] > b[2]);
          DIALS_ASSERT(b[5] > b[4]);
          DIALS_ASSERT(frame_ >= b[4] && frame_ < b[5]);
          int x0 = b[0];
          int x1 = b[1];
          int y0 = b[2];
          int y1 = b[3];
          int z0 = b[4];
          int xs = x1 - x0;
          int ys = y1 - y0;
          int z = frame_ - z0;
          int yi = (int)data.accessor()[0];
          int xi = (int)data.accessor()[1];
          // std::cout << yi << " " << xi << " " << b[1] - b[0] << " " << b[3] - b[2] <<
          // " " << b[5] - b[4] << std::endl;

          int xb = x0 >= 0 ? 0 : std::abs(x0);
          int yb = y0 >= 0 ? 0 : std::abs(y0);
          int xe = x1 <= xi ? xs : xs - (x1 - xi);
          int ye = y1 <= yi ? ys : ys - (y1 - yi);
          if (yb >= ye || xb >= xe) {
            continue;
          }
          DIALS_ASSERT(yb >= 0 && ye <= ys);
          DIALS_ASSERT(xb >= 0 && xe <= xs);
          DIALS_ASSERT(yb + y0 >= 0 && ye + y0 <= yi);
          DIALS_ASSERT(xb + x0 >= 0 && xe + x0 <= xi);
          DIALS_ASSERT(sbox.is_consistent());
          if (flatten_) {
            for (std::size_t y = yb; y < ye; ++y) {
              for (std::size_t x = xb; x < xe; ++x) {
                sdata(0, y, x) += data(y + y0, x + x0);
                bool sv = smask(0, y, x) & Valid;
                bool mv = mask(y + y0, x + x0);
                smask(0, y, x) = (mv && (z == 0 ? true : sv) ? Valid : 0);
              }
            }
          } else {
            for (std::size_t y = yb; y < ye; ++y) {
              for (std::size_t x = xb; x < xe; ++x) {
                // std::cout << data(y + y0, x + x0) << std::endl;
                sdata(z, y, x) = data(y + y0, x + x0);
                smask(z, y, x) = mask(y + y0, x + x0) ? Valid : 0;
              }
            }
          }
          if (frame_ == sbox.bbox[5] - 1) {
            process_indices.push_back(ind[i]);
          }
        }
      }

      // Update timing info
      double end_time = timestamp();
      extract_time_ += end_time - start_time;

      // Update the frame counter
      frame_++;
    }
    /** @returns The first frame.  */
    int frame0() const {
      return frame0_;
    }

    /** @returns The last frame */
    int frame1() const {
      return frame1_;
    }

    /** @returns The current frame. */
    int frame() const {
      return frame_;
    }

    /** @returns The number of frames  */
    std::size_t nframes() const {
      return nframes_;
    }

    /** @returns The number of panels */
    std::size_t npanels() const {
      return npanels_;
    }

    /**
     * @returns Is the extraction finished.
     */
    bool finished() const {
      return frame_ == frame1_;
    }

    /**
     * @returns The extract time
     */
    double extract_time() const {
      return extract_time_;
    }

    /**
     * @returns The process time
     */
    double process_time() const {
      return process_time_;
    }

  private:
    /**
     * Get an index array specifying which reflections are recorded on a given
     * frame and panel.
     * @param frame The frame number
     * @param panel The panel number
     * @returns An array of indices
     */
    af::const_ref<std::size_t> indices(int frame, std::size_t panel) const {
      std::size_t j0 = panel + (frame - frame0_) * npanels_;
      DIALS_ASSERT(offset_.size() > 0);
      DIALS_ASSERT(j0 < offset_.size() - 1);
      std::size_t i0 = offset_[j0];
      std::size_t i1 = offset_[j0 + 1];
      DIALS_ASSERT(i1 >= i0);
      std::size_t off = i0;
      std::size_t num = i1 - off;
      DIALS_ASSERT(off + num <= indices_.size());
      return af::const_ref<std::size_t>(&indices_[off], num);
    }

    af::reflection_table data_;
    double extract_time_;
    double process_time_;
    bool flatten_;
    bool save_;
    std::size_t npanels_;
    int frame0_;
    int frame1_;
    int frame_;
    std::size_t nframes_;
    std::vector<std::size_t> indices_;
    std::vector<std::size_t> offset_;
  };

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_INTEGRATION_PROCESSOR_H
