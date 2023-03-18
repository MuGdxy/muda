#pragma once
#include <Eigen/Core>
#include <muda/muda_def.h>
#include <muda/tools/debug_log.h>
#include <muda/math/math.h>
#include "distance/CCD.h"

namespace muda
{
template <typename T>
class accd
{
  public:
    /* CCD */
    MUDA_GENERIC accd() {}

    MUDA_GENERIC bool Point_Point_CCD(Eigen::Matrix<T, 3, 1> p0,
                                      Eigen::Matrix<T, 3, 1> p1,
                                      Eigen::Matrix<T, 3, 1> dp0,
                                      Eigen::Matrix<T, 3, 1> dp1,
                                      T                      eta,
                                      T                      thickness,
                                      int                    max_iter,
                                      T&                     toc)
    {
        return JGSL::Point_Point_CCD(p0, p1, dp0, dp1, eta, thickness, max_iter, toc);
    }


    MUDA_GENERIC bool Point_Edge_CCD(const Eigen::Matrix<T, 2, 1>& x0,
                                     const Eigen::Matrix<T, 2, 1>& x1,
                                     const Eigen::Matrix<T, 2, 1>& x2,
                                     const Eigen::Matrix<T, 2, 1>& d0,
                                     const Eigen::Matrix<T, 2, 1>& d1,
                                     const Eigen::Matrix<T, 2, 1>& d2,
                                     T                             eta,
                                     T&                            toc)
    {
        return JGSL::Point_Edge_CCD(x0, x1, x2, d0, d1, d2, eta, toc);
    }

    template <class T>
    MUDA_GENERIC bool Point_Edge_CCD(Eigen::Matrix<T, 3, 1> p,
                                     Eigen::Matrix<T, 3, 1> e0,
                                     Eigen::Matrix<T, 3, 1> e1,
                                     Eigen::Matrix<T, 3, 1> dp,
                                     Eigen::Matrix<T, 3, 1> de0,
                                     Eigen::Matrix<T, 3, 1> de1,
                                     T                      eta,
                                     T                      thickness,
                                     int                    max_iter,
                                     T&                     toc)
    {
        return JGSL::Point_Edge_CCD(p, e0, e1, dp, de0, de1, eta, max_iter, toc);
    }

    MUDA_GENERIC bool Point_Triangle_CCD(Eigen::Matrix<T, 3, 1> p,
                                         Eigen::Matrix<T, 3, 1> t0,
                                         Eigen::Matrix<T, 3, 1> t1,
                                         Eigen::Matrix<T, 3, 1> t2,
                                         Eigen::Matrix<T, 3, 1> dp,
                                         Eigen::Matrix<T, 3, 1> dt0,
                                         Eigen::Matrix<T, 3, 1> dt1,
                                         Eigen::Matrix<T, 3, 1> dt2,
                                         T                      eta,
                                         T                      thickness,
                                         int                    max_iter,
                                         T&                     toc)
    {
        return JGSL::Point_Triangle_CCD(
            p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, max_iter, toc);
    }

    template <class T>
    MUDA_GENERIC bool Edge_Edge_CCD(Eigen::Matrix<T, 3, 1> ea0,
                                    Eigen::Matrix<T, 3, 1> ea1,
                                    Eigen::Matrix<T, 3, 1> eb0,
                                    Eigen::Matrix<T, 3, 1> eb1,
                                    Eigen::Matrix<T, 3, 1> dea0,
                                    Eigen::Matrix<T, 3, 1> dea1,
                                    Eigen::Matrix<T, 3, 1> deb0,
                                    Eigen::Matrix<T, 3, 1> deb1,
                                    T                      eta,
                                    T                      thickness,
                                    int                    max_iter,
                                    T&                     toc)
    {
        return JGSL::Edge_Edge_CCD(
            ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness, max_iter, toc);
    }


    template <int dim>
    MUDA_GENERIC bool Point_Edge_CD_Broadphase(const Eigen::Matrix<T, dim, 1>& x0,
                                               const Eigen::Matrix<T, dim, 1>& x1,
                                               const Eigen::Matrix<T, dim, 1>& x2,
                                               T dist)
    {
        return JGSL::Point_Edge_CD_Broadphase(x0, x1, x2, dist);
    }


    MUDA_GENERIC bool Point_Edge_CCD_Broadphase(const Eigen::Matrix<T, 2, 1>& p,
                                                const Eigen::Matrix<T, 2, 1>& e0,
                                                const Eigen::Matrix<T, 2, 1>& e1,
                                                const Eigen::Matrix<T, 2, 1>& dp,
                                                const Eigen::Matrix<T, 2, 1>& de0,
                                                const Eigen::Matrix<T, 2, 1>& de1,
                                                T dist)
    {
        return JGSL::Point_Edge_CCD_Broadphase(p, e0, e1, dp, de0, de1, dist);
    }


    MUDA_GENERIC bool Point_Triangle_CD_Broadphase(const Eigen::Matrix<T, 3, 1>& p,
                                                   const Eigen::Matrix<T, 3, 1>& t0,
                                                   const Eigen::Matrix<T, 3, 1>& t1,
                                                   const Eigen::Matrix<T, 3, 1>& t2,
                                                   T dist)
    {
        return JGSL::Point_Triangle_CD_Broadphase(p, t0, t1, t2, dist);
    }

    MUDA_GENERIC bool Edge_Edge_CD_Broadphase(const Eigen::Matrix<T, 3, 1>& ea0,
                                              const Eigen::Matrix<T, 3, 1>& ea1,
                                              const Eigen::Matrix<T, 3, 1>& eb0,
                                              const Eigen::Matrix<T, 3, 1>& eb1,
                                              T dist)
    {
        return JGSL::Edge_Edge_CD_Broadphase(ea0, ea1, eb0, eb1, dist);
    }


    MUDA_GENERIC bool Point_Triangle_CCD_Broadphase(const Eigen::Matrix<T, 3, 1>& p,
                                                    const Eigen::Matrix<T, 3, 1>& t0,
                                                    const Eigen::Matrix<T, 3, 1>& t1,
                                                    const Eigen::Matrix<T, 3, 1>& t2,
                                                    const Eigen::Matrix<T, 3, 1>& dp,
                                                    const Eigen::Matrix<T, 3, 1>& dt0,
                                                    const Eigen::Matrix<T, 3, 1>& dt1,
                                                    const Eigen::Matrix<T, 3, 1>& dt2,
                                                    T dist)
    {
        return JGSL::Point_Triangle_CCD_Broadphase(p, t0, t1, t2, dp, dt0, dt1, dt2, dist);
    }

    template <class T>
    MUDA_GENERIC bool Edge_Edge_CCD_Broadphase(const Eigen::Matrix<T, 3, 1>& ea0,
                                               const Eigen::Matrix<T, 3, 1>& ea1,
                                               const Eigen::Matrix<T, 3, 1>& eb0,
                                               const Eigen::Matrix<T, 3, 1>& eb1,
                                               const Eigen::Matrix<T, 3, 1>& dea0,
                                               const Eigen::Matrix<T, 3, 1>& dea1,
                                               const Eigen::Matrix<T, 3, 1>& deb0,
                                               const Eigen::Matrix<T, 3, 1>& deb1,
                                               T dist)
    {
        return JGSL::Edge_Edge_CCD_Broadphase(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, dist);
    }

    MUDA_GENERIC bool Point_Edge_CCD_Broadphase(const Eigen::Matrix<T, 3, 1>& p,
                                                const Eigen::Matrix<T, 3, 1>& e0,
                                                const Eigen::Matrix<T, 3, 1>& e1,
                                                const Eigen::Matrix<T, 3, 1>& dp,
                                                const Eigen::Matrix<T, 3, 1>& de0,
                                                const Eigen::Matrix<T, 3, 1>& de1,
                                                T dist)
    {
        return JGSL::Point_Edge_CCD_Broadphase(p, e0, e1, dp, de0, de1, dist);
    }

    MUDA_GENERIC bool Point_Point_CCD_Broadphase(const Eigen::Matrix<T, 3, 1>& p0,
                                                 const Eigen::Matrix<T, 3, 1>& p1,
                                                 const Eigen::Matrix<T, 3, 1>& dp0,
                                                 const Eigen::Matrix<T, 3, 1>& dp1,
                                                 T dist)
    {
        return JGSL::Point_Point_CCD_Broadphase(p0, p1, dp0, dp1, dist);
    }
};
}  // namespace muda

#include "impl/accd.inl"