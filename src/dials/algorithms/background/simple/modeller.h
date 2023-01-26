/*
 * modeller.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *  Author: Luis Fuentes-Montero
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_BACKGROUND_MODELLER_H
#define DIALS_ALGORITHMS_BACKGROUND_MODELLER_H

#include <cmath>
#include <memory>
#include <scitbx/matrix/inversion.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/error.h>

namespace dials { namespace algorithms { namespace background {

  using scitbx::matrix::inversion_in_place;

  /**
   * An abstract class for the background model
   */
  class Model {
  public:
    virtual ~Model() {}

    // The value of the background model at shoebox position x, y, z
    virtual double value(double z, double y, double x) const = 0;

    // The parameters of the background model
    virtual af::shared<double> params() const = 0;

    // The variances of the parameters from the model fit (not variances of the
    // background values)
    virtual af::shared<double> variances() const = 0;
  };

  /**
   * An abstract class for the background modeller
   */
  class Modeller {
  public:
    virtual ~Modeller() {}

    virtual std::shared_ptr<Model> create(
      const af::const_ref<double, af::c_grid<3> > &data,
      const af::const_ref<bool, af::c_grid<3> > &mask) const = 0;
  };

  /**
   * A background model that is constant per image.
   */
  class Constant2dModel : public Model {
  public:
    Constant2dModel(af::shared<double> a_, af::shared<double> va_) : a(a_), va(va_) {
      DIALS_ASSERT(a.size() == va.size());
    }

    virtual double value(double z, double y, double x) const {
      int index = (int)std::floor(z);
      DIALS_ASSERT(index >= 0 && index < a.size());
      return a[index];
    }

    virtual af::shared<double> params() const {
      return af::shared<double>(a.begin(), a.end());
    }

    virtual af::shared<double> variances() const {
      return af::shared<double>(va.begin(), va.end());
    }

  private:
    af::shared<double> a;
    af::shared<double> va;
  };

  /**
   * Create a background that is constant per image.
   */
  class Constant2dModeller : public Modeller {
  public:
    virtual std::shared_ptr<Model> create(
      const af::const_ref<double, af::c_grid<3> > &data,
      const af::const_ref<bool, af::c_grid<3> > &mask) const {
      DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
      af::shared<double> mean(data.accessor()[0], 0.0);
      af::shared<double> sq_sem(data.accessor()[0], 0.0);

      for (std::size_t k = 0; k < data.accessor()[0]; ++k) {
        // Welford's one-pass mean and variance calculation
        double M = 0;
        double S = 0;
        std::size_t count = 0;
        for (std::size_t j = 0; j < data.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < data.accessor()[2]; ++i) {
            if (mask(k, j, i)) {
              count++;
              double x = data(k, j, i);
              double oldM = M;
              M = M + (x - M) / count;
              S = S + (x - M) * (x - oldM);
            }
          }
        }
        DIALS_ASSERT(count > 1);
        mean[k] = M;
        // variance is S/(count-1), SEM^2 is variance/count
        sq_sem[k] = S / (count * count - count);
      }
      return std::make_shared<Constant2dModel>(mean, sq_sem);
    }
  };

  /**
   * A background model that is constant over the reflection.
   */
  class Constant3dModel : public Model {
  public:
    Constant3dModel(double a_, double va_) : a(a_), va(va_) {}

    virtual double value(double z, double y, double x) const {
      return a;
    }

    virtual af::shared<double> params() const {
      af::shared<double> result(1, a);
      return result;
    }

    virtual af::shared<double> variances() const {
      return af::shared<double>(1, va);
    }

  private:
    double a, va;
  };

  /**
   * Create a background model that is constant over the reflection.
   */
  class Constant3dModeller : public Modeller {
  public:
    virtual std::shared_ptr<Model> create(
      const af::const_ref<double, af::c_grid<3> > &data,
      const af::const_ref<bool, af::c_grid<3> > &mask) const {
      DIALS_ASSERT(data.size() == mask.size());

      // Welford's one-pass mean and variance calculation
      double M = 0;
      double S = 0;
      std::size_t count = 0;
      for (std::size_t i = 0; i < data.size(); ++i) {
        if (mask[i]) {
          count++;
          double x = data[i];
          double oldM = M;
          M = M + (x - M) / count;
          S = S + (x - M) * (x - oldM);
        }
      }
      DIALS_ASSERT(count > 1);
      // variance is S/(count-1), SEM^2 is variance/count
      return std::make_shared<Constant3dModel>(M, S / (count * count - count));
    }
  };

  /**
   * A background model that is a plane per image.
   */
  class Linear2dModel : public Model {
  public:
    Linear2dModel(af::shared<double> a_,
                  af::shared<double> b_,
                  af::shared<double> c_,
                  af::shared<double> va_,
                  af::shared<double> vb_,
                  af::shared<double> vc_)
        : a(a_), b(b_), c(c_), va(va_), vb(vb_), vc(vc_) {
      DIALS_ASSERT(a.size() == b.size());
      DIALS_ASSERT(a.size() == c.size());
      DIALS_ASSERT(a.size() == va.size());
      DIALS_ASSERT(va.size() == vb.size());
      DIALS_ASSERT(va.size() == vc.size());
    }

    virtual double value(double z, double y, double x) const {
      int index = (int)std::floor(z);
      DIALS_ASSERT(index >= 0 && index < c.size());
      return a[index] + b[index] * x + c[index] * y;
    }

    virtual af::shared<double> params() const {
      af::shared<double> result(a.size() * 3);
      for (std::size_t i = 0; i < a.size(); ++i) {
        result[3 * i + 0] = a[i];
        result[3 * i + 1] = b[i];
        result[3 * i + 2] = c[i];
      }
      return result;
    }

    virtual af::shared<double> variances() const {
      af::shared<double> result(a.size() * 3);
      for (std::size_t i = 0; i < a.size(); ++i) {
        result[3 * i + 0] = va[i];
        result[3 * i + 1] = vb[i];
        result[3 * i + 2] = vc[i];
      }
      return result;
    }

  private:
    af::shared<double> a, b, c;
    af::shared<double> va, vb, vc;
  };

  /**
   * Create a background model that is a plane per image.
   */
  class Linear2dModeller : public Modeller {
  public:
    virtual std::shared_ptr<Model> create(
      const af::const_ref<double, af::c_grid<3> > &data,
      const af::const_ref<bool, af::c_grid<3> > &mask) const {
      DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
      af::shared<double> a(mask.accessor()[0], 0);
      af::shared<double> b(mask.accessor()[0], 0);
      af::shared<double> c(mask.accessor()[0], 0);
      af::shared<double> va(mask.accessor()[0], 0);
      af::shared<double> vb(mask.accessor()[0], 0);
      af::shared<double> vc(mask.accessor()[0], 0);
      for (std::size_t k = 0; k < mask.accessor()[0]; ++k) {
        af::shared<double> A(3 * 3, 0);
        af::shared<double> B(3, 0);
        for (std::size_t j = 0; j < mask.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < mask.accessor()[2]; ++i) {
            if (mask(k, j, i)) {
              double x = (i + 0.5);
              double y = (j + 0.5);
              double p = data(k, j, i);
              A[0] += 1;
              A[1] += x;
              A[2] += y;
              A[4] += x * x;
              A[5] += x * y;
              A[8] += y * y;
              B[0] += p;
              B[1] += x * p;
              B[2] += y * p;
            }
          }
        }
        A[3] = A[1];
        A[6] = A[2];
        A[7] = A[5];
        inversion_in_place(A.begin(), 3, B.begin(), 1);
        a[k] = B[0];
        b[k] = B[1];
        c[k] = B[2];
        double S = 0.0;
        int count = 0;
        for (std::size_t j = 0; j < mask.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < mask.accessor()[2]; ++i) {
            if (mask(k, j, i)) {
              double x = (i + 0.5);
              double y = (j + 0.5);
              double p = data(k, j, i);
              double s = (p - a[k] - b[k] * x - c[k] * y);
              S += s * s;
              count++;
            }
          }
        }
        DIALS_ASSERT(count > 3);
        va[k] = S * A[0] / (count - 3);
        vb[k] = S * A[4] / (count - 3);
        vc[k] = S * A[8] / (count - 3);
      }
      return std::make_shared<Linear2dModel>(a, b, c, va, vb, vc);
    }
  };

  /**
   * A background model that is a 3d hyper-plane.
   */
  class Linear3dModel : public Model {
  public:
    Linear3dModel(double a_,
                  double b_,
                  double c_,
                  double d_,
                  double va_,
                  double vb_,
                  double vc_,
                  double vd_)
        : a(a_), b(b_), c(c_), d(d_), va(va_), vb(vb_), vc(vc_), vd(vd_) {}

    virtual double value(double z, double y, double x) const {
      return a + b * x + c * y + d * z;
    }

    virtual af::shared<double> params() const {
      af::shared<double> result(4);
      result[0] = a;
      result[1] = b;
      result[2] = c;
      result[3] = d;
      return result;
    }

    virtual af::shared<double> variances() const {
      af::shared<double> result(4);
      result[0] = va;
      result[1] = vb;
      result[2] = vc;
      result[3] = vd;
      return result;
    }

  private:
    double a, b, c, d;
    double va, vb, vc, vd;
  };

  /**
   * Create a background model that is a 3d hyper-plane.
   */
  class Linear3dModeller : public Modeller {
  public:
    virtual std::shared_ptr<Model> create(
      const af::const_ref<double, af::c_grid<3> > &data,
      const af::const_ref<bool, af::c_grid<3> > &mask) const {
      DIALS_ASSERT(data.accessor().all_eq(mask.accessor()));
      af::shared<double> A(4 * 4, 0);
      af::shared<double> B(4, 0);
      for (std::size_t k = 0; k < mask.accessor()[0]; ++k) {
        for (std::size_t j = 0; j < mask.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < mask.accessor()[2]; ++i) {
            if (mask(k, j, i)) {
              double x = (i + 0.5);
              double y = (j + 0.5);
              double z = (k + 0.5);
              double p = data(k, j, i);
              A[0] += 1;
              A[1] += x;
              A[2] += y;
              A[3] += z;
              A[5] += x * x;
              A[6] += x * y;
              A[7] += x * z;
              A[10] += y * y;
              A[11] += y * z;
              A[15] += z * z;
              B[0] += p;
              B[1] += x * p;
              B[2] += y * p;
              B[3] += z * p;
            }
          }
        }
      }
      A[4] = A[1];
      A[8] = A[2];
      A[9] = A[6];
      A[12] = A[3];
      A[13] = A[7];
      A[14] = A[11];
      inversion_in_place(A.begin(), 4, B.begin(), 1);
      double S = 0.0;
      int count = 0;
      for (std::size_t k = 0; k < mask.accessor()[0]; ++k) {
        for (std::size_t j = 0; j < mask.accessor()[1]; ++j) {
          for (std::size_t i = 0; i < mask.accessor()[2]; ++i) {
            if (mask(k, j, i)) {
              double x = (i + 0.5);
              double y = (j + 0.5);
              double z = (k + 0.5);
              double p = data(k, j, i);
              double s = (p - B[0] - B[1] * x - B[2] * y - B[3] * z);
              S += s * s;
              count++;
            }
          }
        }
      }
      DIALS_ASSERT(count > 4);
      double va = S * A[0] / (count - 4);
      double vb = S * A[5] / (count - 4);
      double vc = S * A[10] / (count - 4);
      double vd = S * A[15] / (count - 4);
      return std::make_shared<Linear3dModel>(B[0], B[1], B[2], B[3], va, vb, vc, vd);
    }
  };

}}}  // namespace dials::algorithms::background

#endif  // DIALS_ALGORITHMS_BACKGROUND_MODELLER_H
