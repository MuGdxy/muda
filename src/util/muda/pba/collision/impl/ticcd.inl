#ifdef __INTELLISENSE__
#include "../ticcd.h"
#endif

namespace muda
{
template <typename T, typename Alloc>
inline MUDA_GENERIC bool ticcd<T, Alloc>::edgeEdgeCCD(const Vector3& a0_start,
                                                      const Vector3& a1_start,
                                                      const Vector3& b0_start,
                                                      const Vector3& b1_start,
                                                      const Vector3& a0_end,
                                                      const Vector3& a1_end,
                                                      const Vector3& b0_end,
                                                      const Vector3& b1_end,
                                                      const Array3&  err_in,
                                                      const Scalar   ms_in,
                                                      Scalar&        toi,
                                                      const Scalar tolerance_in,
                                                      const Scalar t_max_in,
                                                      const int    max_itr,
                                                      Scalar& output_tolerance,
                                                      bool    no_zero_toi)
{
    // unsigned so can be larger than MAX_NO_ZERO_TOI_ITER
    unsigned int no_zero_toi_iter = 0;

    bool is_impacting, tmp_is_impacting;

    // Mutable copies for no_zero_toi
    Scalar t_max     = t_max_in;
    Scalar tolerance = tolerance_in;
    Scalar ms        = ms_in;

    Array3 tol = compute_edge_edge_tolerances(
        a0_start, a1_start, b0_start, b1_start, a0_end, a1_end, b0_end, b1_end, tolerance_in);
    // printf("compute_edge_edge_tolerances\n");
    //////////////////////////////////////////////////////////
    // this should be the error of the whole mesh
    Array3 err;
    // if error[0]<0, means we need to calculate error here
    if(err_in[0] < 0)
    {
        Vector3 vlist[] = {a0_start, a1_start, b0_start, b1_start, a0_end, a1_end, b0_end, b1_end};
        bool use_ms = ms > 0;
        err         = get_numerical_error(vlist, 8, false, use_ms);
    }
    else
    {
        err = err_in;
    }
    // printf("get_numerical_error\n");
    //////////////////////////////////////////////////////////


    do
    {
        muda_kernel_check(t_max >= 0 && t_max <= 1, "t_max=%f\n", t_max);
        tmp_is_impacting = edge_edge_interval_root_finder_BFS(a0_start,
                                                              a1_start,
                                                              b0_start,
                                                              b1_start,
                                                              a0_end,
                                                              a1_end,
                                                              b0_end,
                                                              b1_end,
                                                              tol,
                                                              tolerance,
                                                              err,
                                                              ms,
                                                              t_max,
                                                              max_itr,
                                                              toi,
                                                              output_tolerance);
        // printf("edge_edge_interval_root_finder_BFS\n");
        muda_kernel_check(!tmp_is_impacting || toi >= 0,
                          "tmp_is_impacting=%d, toi=%f\n",
                          tmp_is_impacting,
                          toi);

        if(t_max == t_max_in)
        {
            // This will be the final output because we might need to
            // perform CCD again if the toi is zero. In which case we will
            // use a smaller t_max for more time resolution.
            is_impacting = tmp_is_impacting;
        }
        else
        {
            toi = tmp_is_impacting ? toi : t_max;
        }

        // This modification is for CCD-filtered line-search (e.g., IPC)
        // strategies for dealing with toi = 0:
        // 1. shrink t_max (when reaches max_itr),
        // 2. shrink tolerance (when not reach max_itr and tolerance is big) or
        // ms (when tolerance is too small comparing with ms)
        if(tmp_is_impacting && toi == 0 && no_zero_toi)
        {
            if(output_tolerance > tolerance)
            {
                // reaches max_itr, so shrink t_max to return a more accurate result to reach target tolerance.
                t_max *= 0.9;
            }
            else if(10 * tolerance < ms)
            {
                ms *= 0.5;  // ms is too large, shrink it
            }
            else
            {
                tolerance *= 0.5;  // tolerance is too large, shrink it

                tol = compute_edge_edge_tolerances(
                    a0_start, a1_start, b0_start, b1_start, a0_end, a1_end, b0_end, b1_end, tolerance);
            }
        }
        // Only perform a second iteration if toi == 0.
        // WARNING: This option assumes the initial distance is not zero.
    } while(no_zero_toi && ++no_zero_toi_iter < MAX_NO_ZERO_TOI_ITER
            && tmp_is_impacting && toi == 0);

    muda_kernel_check(!no_zero_toi || !is_impacting || toi != 0,
                      "no_zero_toi=%d, is_impacting=%d, toi=%f\n",
                      no_zero_toi,
                      is_impacting,
                      toi);

    return is_impacting;
}
template <typename T, typename Alloc>
inline MUDA_GENERIC bool ticcd<T, Alloc>::vertexFaceCCD(const Vector3& vertex_start,
                                                        const Vector3& face_vertex0_start,
                                                        const Vector3& face_vertex1_start,
                                                        const Vector3& face_vertex2_start,
                                                        const Vector3& vertex_end,
                                                        const Vector3& face_vertex0_end,
                                                        const Vector3& face_vertex1_end,
                                                        const Vector3& face_vertex2_end,
                                                        const Array3& err_in,
                                                        const Scalar  ms_in,
                                                        Scalar&       toi,
                                                        const Scalar tolerance_in,
                                                        const Scalar t_max_in,
                                                        const int    max_itr,
                                                        Scalar& output_tolerance,
                                                        bool no_zero_toi)
{
    // unsigned so can be larger than MAX_NO_ZERO_TOI_ITER
    unsigned int no_zero_toi_iter = 0;

    bool is_impacting, tmp_is_impacting;

    // Mutable copies for no_zero_toi
    Scalar t_max     = t_max_in;
    Scalar tolerance = tolerance_in;
    Scalar ms        = ms_in;

    Array3 tol = compute_face_vertex_tolerances(vertex_start,
                                                face_vertex0_start,
                                                face_vertex1_start,
                                                face_vertex2_start,
                                                vertex_end,
                                                face_vertex0_end,
                                                face_vertex1_end,
                                                face_vertex2_end,
                                                tolerance);

    //////////////////////////////////////////////////////////
    // this is the error of the whole mesh
    Array3 err;
    // if error[0]<0, means we need to calculate error here
    if(err_in[0] < 0)
    {
        Vector3 vlist[] = {vertex_start,
                           face_vertex0_start,
                           face_vertex1_start,
                           face_vertex2_start,
                           vertex_end,
                           face_vertex0_end,
                           face_vertex1_end,
                           face_vertex2_end};
        bool    use_ms  = ms > 0;
        err             = get_numerical_error(vlist, 8, true, use_ms);
    }
    else
    {
        err = err_in;
    }
    //////////////////////////////////////////////////////////

    do
    {
        muda_kernel_assert(t_max >= 0 && t_max <= 1, "t_max=%f\n", t_max);
        tmp_is_impacting = vertex_face_interval_root_finder_BFS(vertex_start,
                                                                face_vertex0_start,
                                                                face_vertex1_start,
                                                                face_vertex2_start,
                                                                vertex_end,
                                                                face_vertex0_end,
                                                                face_vertex1_end,
                                                                face_vertex2_end,
                                                                tol,
                                                                tolerance,
                                                                err,
                                                                ms,
                                                                t_max,
                                                                max_itr,
                                                                toi,
                                                                output_tolerance);
        muda_kernel_assert(!tmp_is_impacting || toi >= 0,
                           "tmp_is_impacting=%d, toi=%d\n",
                           tmp_is_impacting,
                           toi);

        if(t_max == t_max_in)
        {
            // This will be the final output because we might need to
            // perform CCD again if the toi is zero. In which case we will
            // use a smaller t_max for more time resolution.
            is_impacting = tmp_is_impacting;
        }
        else
        {
            toi = tmp_is_impacting ? toi : t_max;
        }

        // This modification is for CCD-filtered line-search (e.g., IPC)
        // strategies for dealing with toi = 0:
        // 1. shrink t_max (when reaches max_itr),
        // 2. shrink tolerance (when not reach max_itr and tolerance is big) or
        // ms (when tolerance is too small comparing with ms)
        if(tmp_is_impacting && toi == 0 && no_zero_toi)
        {
            if(output_tolerance > tolerance)
            {
                // reaches max_itr, so shrink t_max to return a more accurate result to reach target tolerance.
                t_max *= 0.9;
            }
            else if(10 * tolerance < ms)
            {
                ms *= 0.5;  // ms is too large, shrink it
            }
            else
            {
                tolerance *= 0.5;  // tolerance is too large, shrink it

                // recompute this
                tol = compute_face_vertex_tolerances(vertex_start,
                                                     face_vertex0_start,
                                                     face_vertex1_start,
                                                     face_vertex2_start,
                                                     vertex_end,
                                                     face_vertex0_end,
                                                     face_vertex1_end,
                                                     face_vertex2_end,
                                                     tolerance);
            }
        }

        // Only perform a second iteration if toi == 0.
        // WARNING: This option assumes the initial distance is not zero.
    } while(no_zero_toi && ++no_zero_toi_iter < MAX_NO_ZERO_TOI_ITER
            && tmp_is_impacting && toi == 0);
    muda_kernel_assert(!no_zero_toi || !is_impacting || toi != 0,
                       "no_zero_toi=%d, is_impacting=%d, toi=%f\n",
                       no_zero_toi,
                       is_impacting,
                       toi);
    return is_impacting;
}

template <typename T, typename Alloc>
template <bool check_vf>
MUDA_GENERIC bool ticcd<T, Alloc>::interval_root_finder_BFS(const Vector3& a0s,
                                                            const Vector3& a1s,
                                                            const Vector3& b0s,
                                                            const Vector3& b1s,
                                                            const Vector3& a0e,
                                                            const Vector3& a1e,
                                                            const Vector3& b0e,
                                                            const Vector3& b1e,
                                                            const Interval3& iset,
                                                            const Array3& tol,
                                                            const Scalar co_domain_tolerance,
                                                            const Array3& err,
                                                            const Scalar  ms,
                                                            const Scalar max_time,
                                                            const int max_itr,
                                                            Scalar&   toi,
                                                            Scalar& output_tolerance)
{
    long queue_size = 0;
    // if max_itr <0, output_tolerance= co_domain_tolerance;
    // else, output_tolearancewill be the precision after iteration time > max_itr
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    Scalar temp_output_tolerance = co_domain_tolerance;

    // Stack of intervals and the last split dimension
    // std::stack<std::pair<Interval3,int>> istack;

    istack.get_container().clear();
    istack.push({iset, -1});

    // current intervals
    Interval3 current;
    int       refine       = 0;
    Scalar    impact_ratio = 1;

    toi = INFINITY;  //set toi as infinate
    // temp_toi is to catch the toi of each level
    Scalar temp_toi = toi;
    // set TOI to 4. this is to record the impact time of this level
    NumCCD TOI(4, 0);
    // this is to record the element that already small enough or contained in eps-box
    NumCCD TOI_SKIP      = TOI;
    bool   use_skip      = false;  // this is to record if TOI_SKIP is used.
    int    rnbr          = 0;
    int    current_level = -2;  // in the begining, current_level != level
    int    box_in_level  = -2;  // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller than tolerance or eps-box
    bool   this_level_less_tol = true;
    bool   find_level_root     = false;
    Scalar t_upper_bound       = max_time;  // 2*tol make it more conservative
    while(!istack.empty())
    {

        if(MAX_QUEUE_SIZE > 0 && istack.size() >= MAX_QUEUE_SIZE)
        {
            muda_kernel_debug_info(DEBUG_TICCD, "ticcd istack full so just return, toi=%f", toi);
            return true;
        }

        current   = istack.top().first;
        int level = istack.top().second;
        istack.pop();

        // if this box is later than TOI_SKIP in time, we can skip this one.
        // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
        if(current[0].lower >= TOI_SKIP)
        {
            continue;
        }
        // before check a new level, set this_level_less_tol=true
        if(box_in_level != level)
        {
            box_in_level        = level;
            this_level_less_tol = true;
        }

        refine++;
        bool   zero_in, box_in;
        Array3 true_tol;
        {
            TIGHT_INCLUSION_SCOPED_TIMER(time_predicates);
            // #ifdef TIGHT_INCLUSION_USE_GMP // this is defined in the begining of this file
            // Array3 ms_3d = Array3::Constant(ms);
            // zero_in = origin_in_function_bounding_box_rational_return_tolerance<
            //     check_vf>(
            //     current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, ms_3d, box_in,
            //     true_tol);
            // #else
            zero_in = origin_in_function_bounding_box_vector<check_vf>(
                current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, err, box_in, ms, &true_tol);
            // #endif
        }

        if(!zero_in)
            continue;

        Array3 widths;
        {
            TIGHT_INCLUSION_SCOPED_TIMER(time_width);
            widths = width(current);
        }

        bool tol_condition = (true_tol <= co_domain_tolerance).all();

        // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we have condition 2
        bool condition1 = (widths <= tol).all();

        // Condition 2, zero_in = true, box inside eps-box and in this level,
        // no box whose zero_in is true but box size larger than tolerance, can return
        bool condition2 = box_in && this_level_less_tol;
        if(!tol_condition)
        {
            this_level_less_tol = false;
            // this level has at least one box whose size > tolerance, thus we
            // cannot directly return if find one box whose size < tolerance or box-in
            // TODO: think about it. maybe we can return even if this value is false, so we can terminate earlier.
        }

        // Condition 3, in this level, we find a box that zero-in and size < tolerance.
        // and no other boxes whose zero-in is true in this level before this one is larger than tolerance, can return
        bool condition3 = this_level_less_tol;
        if(condition1 || condition2 || condition3)
        {
            TOI = current[0].lower;
            rnbr++;
            // continue;
            toi = TOI.value() * impact_ratio;
            // we don't need to compare with TOI_SKIP because we already
            // continue when t >= TOI_SKIP
            return true;
        }

        if(max_itr > 0)
        {  // if max_itr <= 0  -> unlimited iterations
            if(current_level != level)
            {
                // output_tolerance=current_tolerance;
                // current_tolerance=0;
                current_level   = level;
                find_level_root = false;
            }

            if(!find_level_root)
            {
                TOI = current[0].lower;
                // collision=true;
                rnbr++;
                // continue;
                temp_toi = TOI.value() * impact_ratio;

                // if the real tolerance is larger than input, use the real one;
                // if the real tolerance is smaller than input, use input
                temp_output_tolerance = muda::max(
                    Vector4(true_tol[0], true_tol[1], true_tol[2], co_domain_tolerance));
                // this ensures always find the earlist root
                find_level_root = true;
            }
            if(refine > max_itr)
            {
                toi              = temp_toi;
                output_tolerance = temp_output_tolerance;
                return true;
            }
            // get the time of impact down here
        }

        // if this box is small enough, or inside of eps-box, then just continue,
        // but we need to record the collision time
        if(tol_condition || box_in)
        {
            if(current[0].lower < TOI_SKIP)
            {
                TOI_SKIP = current[0].lower;
            }
            use_skip = true;
            continue;
        }

        // find the next dimension to split
        int split_i = find_next_split(widths, tol);

        bool overflow =
            split_and_push(current, split_i, istack, level + 1, check_vf, t_upper_bound);

        if(overflow)
        {
            muda_kernel_printf("OVERFLOW HAPPENS WHEN SPLITTING INTERVALS\n");
            return true;
        }
    }

    if(use_skip)
    {
        toi = TOI_SKIP.value() * impact_ratio;
        return true;
    }

    return false;
}

template <typename T, typename Alloc>
MUDA_GENERIC bool ticcd<T, Alloc>::split_and_push(
    const Interval3& tuv, int split_i, IStack& istack, int level, bool check_vf, Scalar t_upper_bound)
{
    auto halves = tuv[split_i].bisect();
    if(halves[0].lower >= halves[0].upper || halves[1].lower >= halves[1].upper)
    {
        muda_kernel_printf("OVERFLOW HAPPENS WHEN SPLITTING INTERVALS\n");
        return true;
    }

    Interval3 tmp = tuv;

    if(split_i == 0)
    {
        if(t_upper_bound == 1 || halves[1].overlaps(0, t_upper_bound))
        {
            tmp[split_i] = halves[1];
            istack.push({tmp, level});
        }
        if(t_upper_bound == 1 || halves[0].overlaps(0, t_upper_bound))
        {
            tmp[split_i] = halves[0];
            istack.push({tmp, level});
        }
    }
    else if(!check_vf)
    {  // edge uv
        tmp[split_i] = halves[1];
        istack.push({tmp, level});
        tmp[split_i] = halves[0];
        istack.push({tmp, level});
    }
    else
    {
        muda_kernel_check(check_vf && split_i != 0, "check_vf=%d, split_i=%d\n", check_vf, split_i);
        // u + v <= 1
        if(split_i == 1)
        {
            const Interval& v = tuv[2];
            if(NumCCD::is_sum_leq_1(halves[1].lower, v.lower))
            {
                tmp[split_i] = halves[1];
                istack.push({tmp, level});
            }
            if(NumCCD::is_sum_leq_1(halves[0].lower, v.lower))
            {
                tmp[split_i] = halves[0];
                istack.push({tmp, level});
            }
        }
        else if(split_i == 2)
        {
            const Interval& u = tuv[1];
            if(NumCCD::is_sum_leq_1(u.lower, halves[1].lower))
            {
                tmp[split_i] = halves[1];
                istack.push({tmp, level});
            }
            if(NumCCD::is_sum_leq_1(u.lower, halves[0].lower))
            {
                tmp[split_i] = halves[0];
                istack.push({tmp, level});
            }
        }
    }
    return false;  // no overflow
}
}  // namespace muda