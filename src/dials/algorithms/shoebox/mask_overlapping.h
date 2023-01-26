/*
 * mask_overlapping.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_INTEGRATION_MASK_OVERLAPPING_H
#define DIALS_ALGORITHMS_INTEGRATION_MASK_OVERLAPPING_H

#include <memory>
#include <scitbx/vec3.h>
#include <scitbx/array_family/tiny_types.h>
#include <dials/model/data/adjacency_list.h>
#include <dials/algorithms/shoebox/mask_code.h>
#include <dials/error.h>

namespace dials { namespace algorithms { namespace shoebox {

  using dials::model::AdjacencyList;
  using dials::model::Shoebox;
  using scitbx::vec3;
  using scitbx::af::int3;
  using scitbx::af::int6;

  /** Class to calculate the shoebox masks for all reflections */
  class MaskOverlapping {
  public:
    // Useful typedefs
    typedef AdjacencyList::edge_iterator edge_iterator;
    typedef AdjacencyList::edge_iterator_range edge_iterator_range;

    /**
     * Initialise the algorithm
     */
    MaskOverlapping() {}

    /**
     * The entry point of the functor. The list of reflections is queried
     * against the adjacency list to find overlapping reflections. Those
     * that have no overlaps their whole mask set to 1 to indicate that
     * the reflection owns all its pixels. Those that do overlap are
     * then compared pixel-by-pixel in the overlapping region and the
     * reflection whose predicted central location is closer to the pixel
     * will gain ownership of the pixel (mask value 1).
     *
     * @param shoeboxes The list of shoeboxes
     * @param coords The pixel coordinate
     * @param adjacency_list The adjacency_list
     */
    void operator()(af::ref<Shoebox<> > shoeboxes,
                    const af::const_ref<vec3<double> > &coords,
                    const std::shared_ptr<AdjacencyList> &adjacency_list) const {
      // Loop through all the reflections
      if (adjacency_list) {
        for (std::size_t i = 0; i < shoeboxes.size(); ++i) {
          // Get a reference to the reflection
          Shoebox<> &s = shoeboxes[i];
          vec3<double> c = coords[i];

          // Get the list of overlapping shoeboxes
          edge_iterator_range range = adjacency_list->edges(i);
          for (edge_iterator it = range.first; it != range.second; ++it) {
            std::size_t index1 = it->first;
            std::size_t index2 = it->second;
            DIALS_ASSERT(index1 == i);
            if (index1 < index2) {
              assign_ownership(s, c, shoeboxes[index2], coords[index2]);
            }
          }
        }
      }
    }

  private:
    /**
     * The distance between two points
     * @param a Point a
     * @param b Point b
     * @returns The distance from a to b
     */
    double distance(vec3<double> a, vec3<double> b) const {
      return (b - a).length_sq();
    }

    /**
     * Get a coordinate from the voxel.
     * @param i The voxel x index
     * @param j The voxel y index
     * @param k THe voxel z index
     * @returns An (x, y, z) coordinate
     */
    vec3<double> voxel_coord(double i, double j, double k) const {
      return vec3<double>(i + 0.5, j + 0.5, k + 0.5);
    }

    /**
     * Assign ownership of all the pixels in the overlapping range.
     * @param a Shoebox a
     * @param coord_a The coordinate of a
     * @param b Shoebox b
     * @param coord_b The coordinate of b
     * @throws RuntimeError if reflections to do overlap.
     */
    void assign_ownership(Shoebox<> &a,
                          vec3<double> coord_a,
                          Shoebox<> &b,
                          vec3<double> coord_b) const {
      // Get the reflection mask arrays
      af::ref<int, af::c_grid<3> > mask_a = a.mask.ref();
      af::ref<int, af::c_grid<3> > mask_b = b.mask.ref();

      // Get the sizes of the masks
      af::c_grid<3> size_a = mask_a.accessor();
      af::c_grid<3> size_b = mask_b.accessor();

      // Get the bounding boxes
      int6 bbox_a = a.bbox;
      int6 bbox_b = b.bbox;

      // Get range to iterate over
      int i0 = std::max(bbox_a[0], bbox_b[0]);
      int i1 = std::min(bbox_a[1], bbox_b[1]);
      int j0 = std::max(bbox_a[2], bbox_b[2]);
      int j1 = std::min(bbox_a[3], bbox_b[3]);
      int k0 = std::max(bbox_a[4], bbox_b[4]);
      int k1 = std::min(bbox_a[5], bbox_b[5]);

      // Ensure ranges are valid
      DIALS_ASSERT(k1 > k0 && j1 > j0 && i1 > i0);

      DIALS_ASSERT(i0 - bbox_a[0] >= 0 && i1 - bbox_a[0] <= size_a[2]);
      DIALS_ASSERT(j0 - bbox_a[2] >= 0 && j1 - bbox_a[2] <= size_a[1]);
      DIALS_ASSERT(k0 - bbox_a[4] >= 0 && k1 - bbox_a[4] <= size_a[0]);

      DIALS_ASSERT(i0 - bbox_b[0] >= 0 && i1 - bbox_b[0] <= size_b[2]);
      DIALS_ASSERT(j0 - bbox_b[2] >= 0 && j1 - bbox_b[2] <= size_b[1]);
      DIALS_ASSERT(k0 - bbox_b[4] >= 0 && k1 - bbox_b[4] <= size_b[0]);

      // Iterate over range of indices
      for (int k = k0; k < k1; ++k) {
        for (int j = j0; j < j1; ++j) {
          for (int i = i0; i < i1; ++i) {
            // Set the coordinate
            vec3<double> coord_c(voxel_coord(i, j, k));

            // Set the mask indices
            int ka = k - bbox_a[4], kb = k - bbox_b[4];
            int ja = j - bbox_a[2], jb = j - bbox_b[2];
            int ia = i - bbox_a[0], ib = i - bbox_b[0];

            // If the distance from a to c is less than b to c then
            // set the mask for a to 1 and b to 0, otherwise set
            // b to 1 and a to 0.
            if (distance(coord_a, coord_c) < distance(coord_b, coord_c)) {
              mask_b(kb, jb, ib) = 0;
            } else {
              mask_a(ka, ja, ia) = 0;
            }
          }
        }
      }
    }
  };

}}}  // namespace dials::algorithms::shoebox

#endif /* DIALS_ALGORITHMS_INTEGRATION_MASK_OVERLAPPING_H */
