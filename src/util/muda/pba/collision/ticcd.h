#include <Eigen/Core>
#include <muda/muda_def.h>
#include <muda/tools/debug_log.h>
#include <muda/math/math.h>
#include <muda/thread_only/priority_queue.h>

#define TIGHT_INCLUSION_SCOPED_TIMER(x)
#define CCD_MAX_TIME_TOL INFINITY
#define CCD_MAX_COORD_TOL INFINITY
#define MAX_DENOM_POWER (8 * sizeof(uint64_t) - 1)
#define MAX_NO_ZERO_TOI_ITER INT_MAX

#undef max
#undef min

namespace muda::thread_only
{
namespace details
{
    // calculate a*(2^b)
    // calculate a*(2^b)
    MUDA_GENERIC inline uint64_t power(const uint64_t a, const uint8_t b)
    {
        // The fast bit shifting power trick only works if b is not too larger.
        muda_kernel_check(b < MAX_DENOM_POWER);
        // WARNING: Technically this can still fail with `b < MAX_DENOM_POWER` if `a > 1`.
        return a << b;
    }

    // calculate 2^exponent
    MUDA_GENERIC inline uint64_t pow2(const uint8_t exponent)
    {
        return power(1l, exponent);
    }

    // return power t. n=result*2^t
    // return power t. n=result*2^t
    MUDA_GENERIC inline uint8_t reduction(const uint64_t n, uint64_t& result)
    {
        uint8_t t = 0;
        result    = n;
        while(result != 0 && (result & 1) == 0)
        {
            result >>= 1;
            t++;
        }
        return t;
    }

    MUDA_GENERIC inline float max_linf_4(const Eigen::Vector3f& p1,
                                         const Eigen::Vector3f& p2,
                                         const Eigen::Vector3f& p3,
                                         const Eigen::Vector3f& p4,
                                         const Eigen::Vector3f& p1e,
                                         const Eigen::Vector3f& p2e,
                                         const Eigen::Vector3f& p3e,
                                         const Eigen::Vector3f& p4e)
    {
        float a = (p1e - p1).lpNorm<Eigen::Infinity>();
        float b = (p2e - p2).lpNorm<Eigen::Infinity>();
        float c = (p3e - p3).lpNorm<Eigen::Infinity>();
        float d = (p4e - p4).lpNorm<Eigen::Infinity>();
        return muda::max(Eigen::Vector4f(a, b, c, d));
    }

    MUDA_GENERIC inline double max_linf_4(const Eigen::Vector3d& p1,
                                          const Eigen::Vector3d& p2,
                                          const Eigen::Vector3d& p3,
                                          const Eigen::Vector3d& p4,
                                          const Eigen::Vector3d& p1e,
                                          const Eigen::Vector3d& p2e,
                                          const Eigen::Vector3d& p3e,
                                          const Eigen::Vector3d& p4e)
    {
        double a = (p1e - p1).lpNorm<Eigen::Infinity>();
        double b = (p2e - p2).lpNorm<Eigen::Infinity>();
        double c = (p3e - p3).lpNorm<Eigen::Infinity>();
        double d = (p4e - p4).lpNorm<Eigen::Infinity>();
        return muda::max(Eigen::Vector4d(a, b, c, d));
    }
}  // namespace details

template <typename T = float, typename Alloc = thread_allocator>
class ticcd
{
  public:
    using allocator_type = Alloc;
    MUDA_GENERIC ticcd(const Alloc& alloc, int max_queue_size = -1)
        : MAX_QUEUE_SIZE(max_queue_size)
        , istack(alloc)
    {
        if(MAX_QUEUE_SIZE > 0)
            istack.get_container().reserve(MAX_QUEUE_SIZE);
        else
            istack.get_container().reserve(64);
    }

    MUDA_GENERIC ticcd(int max_queue_size = -1)
        : MAX_QUEUE_SIZE(max_queue_size)
        , istack()
    {
        if(MAX_QUEUE_SIZE > 0)
            istack.get_container().reserve(MAX_QUEUE_SIZE);
        else
            istack.get_container().reserve(64);
    }
    int MAX_QUEUE_SIZE;

    using Scalar  = T;
    using Vector3 = Eigen::Vector<Scalar, 3>;
    using Vector4 = Eigen::Vector<Scalar, 4>;
    using Array3  = Eigen::Array<Scalar, 3, 1>;

    //<k,n> pair present a number k/pow(2,n)
    struct NumCCD
    {
        uint64_t numerator;
        uint8_t  denom_power;

        MUDA_GENERIC NumCCD() {}

        MUDA_GENERIC NumCCD(uint64_t p_numerator, uint8_t p_denom_power)
            : numerator(p_numerator)
            , denom_power(p_denom_power)
        {
        }

        MUDA_GENERIC NumCCD(Scalar x)
        {
            // Use bisection to find an upper bound of x.
            muda_kernel_check(x >= 0 && x <= 1);
            NumCCD low(0, 0), high(1, 0), mid;

            // Hard code these cases for better accuracy.
            if(x == 0)
            {
                *this = low;
                return;
            }
            else if(x == 1)
            {
                *this = high;
                return;
            }

            do
            {
                mid = low + high;
                mid.denom_power++;
                if(mid.denom_power >= MAX_DENOM_POWER)
                    break;
                if(x > mid)
                    low = mid;
                else if(x < mid)
                    high = mid;
                else
                    break;
            } while(mid.denom_power < MAX_DENOM_POWER);
            *this = high;
            muda_kernel_check(x <= value());
        }

        MUDA_GENERIC ~NumCCD() {}

        MUDA_GENERIC uint64_t denominator() const
        {
            return details::pow2(denom_power);
        }

        // convert NumCCD to double number
        MUDA_GENERIC Scalar value() const
        {
            return Scalar(numerator) / denominator();
        }

        MUDA_GENERIC operator double() const { return value(); }

        MUDA_GENERIC NumCCD operator+(const NumCCD& other) const
        {
            const uint64_t &k1 = numerator, &k2 = other.numerator;
            const uint8_t & n1 = denom_power, &n2 = other.denom_power;

            NumCCD result;
            if(n1 == n2)
                result.denom_power = n2 - details::reduction(k1 + k2, result.numerator);
            else if(n2 > n1)
            {
                result.numerator = k1 * details::pow2(n2 - n1) + k2;
                muda_kernel_check(result.numerator % 2 == 1);
                result.denom_power = n2;
            }
            else
            {  // n2 < n1
                result.numerator = k1 + k2 * details::pow2(n1 - n2);
                muda_kernel_check(result.numerator % 2 == 1);
                result.denom_power = n1;
            }
            return result;
        }

        MUDA_GENERIC bool operator==(const NumCCD& other) const
        {
            return numerator == other.numerator && denom_power == other.denom_power;
        }

        MUDA_GENERIC bool operator!=(const NumCCD& other) const
        {
            return !(*this == other);
        }

        MUDA_GENERIC bool operator<(const NumCCD& other) const
        {
            const uint64_t &k1 = numerator, &k2 = other.numerator;
            const uint8_t & n1 = denom_power, &n2 = other.denom_power;

            uint64_t tmp_k1 = k1, tmp_k2 = k2;
            if(n1 < n2)
                tmp_k1 = details::pow2(n2 - n1) * k1;
            else if(n1 > n2)
                tmp_k2 = details::pow2(n1 - n2) * k2;
            muda_kernel_check((value() < other.value()) == (tmp_k1 < tmp_k2));
            return tmp_k1 < tmp_k2;
        }

        MUDA_GENERIC bool operator<=(const NumCCD& other) const
        {
            return (*this == other) || (*this < other);
        }

        MUDA_GENERIC bool operator>=(const NumCCD& other) const
        {
            return !(*this < other);
        }

        MUDA_GENERIC bool operator>(const NumCCD& other) const
        {
            return !(*this <= other);
        }

        MUDA_GENERIC bool operator<(const Scalar other) const
        {
            return value() < other;
        }

        MUDA_GENERIC bool operator>(const Scalar other) const
        {
            return value() > other;
        }

        MUDA_GENERIC bool operator==(const Scalar other) const
        {
            return value() == other;
        }

        MUDA_GENERIC static bool is_sum_leq_1(const NumCCD& num1, const NumCCD& num2)
        {
            if(num1.denom_power == num2.denom_power)
            {
                // skip the reduction in num1 + num2
                return num1.numerator + num2.numerator <= num1.denominator();
            }
            NumCCD tmp = num1 + num2;
            return tmp.numerator <= tmp.denominator();
        }
    };


    struct Interval;
    using Interval2 = Eigen::Array<Interval, 2, 1>;
    using Interval3 = Eigen::Array<Interval, 3, 1>;

    // an interval represented by two double numbers
    struct Interval
    {
        NumCCD lower;
        NumCCD upper;

        MUDA_GENERIC Interval() {}

        MUDA_GENERIC Interval(const NumCCD& p_lower, const NumCCD& p_upper)
            : lower(p_lower)
            , upper(p_upper)
        {
        }

        MUDA_GENERIC ~Interval() {}

        MUDA_GENERIC Interval2 bisect() const
        {
            // interval is [k1/pow2(n1), k2/pow2(n2)]
            NumCCD mid = upper + lower;
            mid.denom_power++;  // ÷ 2
            muda_kernel_check(mid.value() > lower.value() && mid.value() < upper.value());
            return Interval2(Interval(lower, mid), Interval(mid, upper));
        }

        MUDA_GENERIC bool overlaps(const Scalar r1, const Scalar r2) const
        {
            return upper.value() >= r1 && lower.value() <= r2;
        }
    };

    struct Interval3Pair
    {
        Interval3 first;
        int       second;
    };

    struct Compare
    {
        MUDA_GENERIC bool operator()(const Interval3Pair& i1, const Interval3Pair& i2)
        {
            if(i1.second != i2.second)
                return i1.second >= i2.second;
            else
                return i1.first[0].lower > i2.first[0].lower;
        }
    };

  public:
    /// @brief This function can give you the answer of continous collision detection with minimum
    /// seperation, and the earlist collision time if collision happens.
    ///
    /// @param[in] err The filters calculated using the bounding box of the simulation scene.
    ///                If you are checking a single query without a scene, please set it as {-1,-1,-1}.
    /// @param[in] ms The minimum seperation. should set: ms < max(abs(x),1), ms < max(abs(y),1), ms < max(abs(z),1) of the QUERY (NOT THE SCENE!).
    /// @param[out] toi The earlist time of collision if collision happens. If there is no collision, toi will be infinate.
    /// @param[in] tolerance A user - input solving precision. We suggest to use 1e-6.
    /// @param[in] t_max The upper bound of the time interval [0,t_max] to be checked. 0<=t_max<=1
    /// @param[in] max_itr A user-defined value to terminate the algorithm earlier, and return a result under current
    ///                    precision. please set max_itr either a big number like 1e7, or -1 which means it will not be terminated
    ///                    earlier and the precision will be user-defined precision -- tolerance.
    /// @param[out] output_tolerance The precision under max_itr ( > 0). if max_itr < 0, output_tolerance = tolerance;
    /// @param[in] no_zero_toi Refine further if a zero toi is produced (assumes not initially in contact).
    /// @return is_impacting
    MUDA_GENERIC bool edgeEdgeCCD(const Vector3& a0_start,
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
                                  const Scalar   tolerance_in,
                                  const Scalar   t_max_in,
                                  const int      max_itr,
                                  Scalar&        output_tolerance,
                                  bool           no_zero_toi = false)
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
            muda_kernel_check(t_max >= 0 && t_max <= 1);
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
            muda_kernel_check(!tmp_is_impacting || toi >= 0);

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

        muda_kernel_check(!no_zero_toi || !is_impacting || toi != 0);

        return is_impacting;
    }

  private:
    MUDA_GENERIC static Array3 width(const Interval3& x)
    {
        return Array3(x[0].upper.value() - x[0].lower.value(),
                      x[1].upper.value() - x[1].lower.value(),
                      x[2].upper.value() - x[2].lower.value());
    }

    MUDA_GENERIC static Array3 compute_edge_edge_tolerances(
        const Vector3& edge0_vertex0_start,
        const Vector3& edge0_vertex1_start,
        const Vector3& edge1_vertex0_start,
        const Vector3& edge1_vertex1_start,
        const Vector3& edge0_vertex0_end,
        const Vector3& edge0_vertex1_end,
        const Vector3& edge1_vertex0_end,
        const Vector3& edge1_vertex1_end,
        const Scalar   distance_tolerance)
    {

        const Vector3 p000 = edge0_vertex0_start - edge1_vertex0_start;
        const Vector3 p001 = edge0_vertex0_start - edge1_vertex1_start;
        const Vector3 p011 = edge0_vertex1_start - edge1_vertex1_start;
        const Vector3 p010 = edge0_vertex1_start - edge1_vertex0_start;
        const Vector3 p100 = edge0_vertex0_end - edge1_vertex0_end;
        const Vector3 p101 = edge0_vertex0_end - edge1_vertex1_end;
        const Vector3 p111 = edge0_vertex1_end - edge1_vertex1_end;
        const Vector3 p110 = edge0_vertex1_end - edge1_vertex0_end;

        Scalar dl = 3 * details::max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
        Scalar edge0_length =
            3 * details::max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
        Scalar edge1_length =
            3 * details::max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

        return Array3(muda::min(distance_tolerance / dl, CCD_MAX_TIME_TOL),
                      muda::min(distance_tolerance / edge0_length, CCD_MAX_COORD_TOL),
                      muda::min(distance_tolerance / edge1_length, CCD_MAX_COORD_TOL));
    }

    MUDA_GENERIC static Array3 get_numerical_error(const Vector3 vertices[],
                                                   uint32_t      size,
                                                   const bool    check_vf,
                                                   const bool using_minimum_separation)
    {
        Scalar eefilter;
        Scalar vffilter;
        if(!using_minimum_separation)
        {
            if constexpr(std::is_same_v<Scalar, double>)
            {
                eefilter = 6.217248937900877e-15;
                vffilter = 6.661338147750939e-15;
            }
            else
            {

                eefilter = 3.337861e-06;
                vffilter = 3.576279e-06;
            }
        }
        else  // using minimum separation
        {
            if constexpr(std::is_same_v<Scalar, double>)
            {
                eefilter = 7.105427357601002e-15;
                vffilter = 7.549516567451064e-15;
            }
            else
            {
                eefilter = 3.814698e-06;
                vffilter = 4.053116e-06;
            }
        }

        Vector3 max = vertices[0].cwiseAbs();
        for(int i = 0; i < size; i++)
        {
            max = max.cwiseMax(vertices[i].cwiseAbs());
        }
        Vector3 delta  = max.cwiseMin(1);
        Scalar  filter = check_vf ? vffilter : eefilter;
        return filter * delta.array().pow(3);
    }

    // find the largest width/tol dimension that is greater than its tolerance
    MUDA_GENERIC static int find_next_split(const Array3& widths, const Array3& tols)
    {
        // muda_kernel_check((widths > tols).any());
        Array3 tmp = (widths > tols).select(widths / tols, -INFINITY);
        int    max_index;
        tmp.maxCoeff(&max_index);
        return max_index;
    }

    //MUDA_GENERIC static bool split_and_push(const Interval3& tuv,
    //                                        int              split_i,
    //                                        std::function<void(const Interval3&)> push,
    //                                        bool   check_vf,
    //                                        Scalar t_upper_bound = 1)
    //{
    //    auto halves = tuv[split_i].bisect();
    //    if(halves[0].lower >= halves[0].upper || halves[1].lower >= halves[1].upper)
    //    {
    //        muda_kernel_printf("OVERFLOW HAPPENS WHEN SPLITTING INTERVALS\n");
    //        return true;
    //    }

    //    Interval3 tmp = tuv;

    //    if(split_i == 0)
    //    {
    //        if(t_upper_bound == 1 || halves[1].overlaps(0, t_upper_bound))
    //        {
    //            tmp[split_i] = halves[1];
    //            push({tmp});
    //        }
    //        if(t_upper_bound == 1 || halves[0].overlaps(0, t_upper_bound))
    //        {
    //            tmp[split_i] = halves[0];
    //            push({tmp});
    //        }
    //    }
    //    else if(!check_vf)
    //    {  // edge uv
    //        tmp[split_i] = halves[1];
    //        push({tmp});
    //        tmp[split_i] = halves[0];
    //        push({tmp});
    //    }
    //    else
    //    {
    //        muda_kernel_check(check_vf && split_i != 0);
    //        // u + v ≤ 1
    //        if(split_i == 1)
    //        {
    //            const Interval& v = tuv[2];
    //            if(NumCCD::is_sum_leq_1(halves[1].lower, v.lower))
    //            {
    //                tmp[split_i] = halves[1];
    //                push({tmp});
    //            }
    //            if(NumCCD::is_sum_leq_1(halves[0].lower, v.lower))
    //            {
    //                tmp[split_i] = halves[0];
    //                push({tmp});
    //            }
    //        }
    //        else if(split_i == 2)
    //        {
    //            const Interval& u = tuv[1];
    //            if(NumCCD::is_sum_leq_1(u.lower, halves[1].lower))
    //            {
    //                tmp[split_i] = halves[1];
    //                push({tmp});
    //            }
    //            if(NumCCD::is_sum_leq_1(u.lower, halves[0].lower))
    //            {
    //                tmp[split_i] = halves[0];
    //                push({tmp});
    //            }
    //        }
    //    }
    //    return false;  // no overflow
    //}

    MUDA_GENERIC static void convert_tuv_to_array(const Interval3& itv,
                                                  Eigen::Array<Scalar, 8, 1>& t_up,
                                                  Eigen::Array<Scalar, 8, 1>& t_dw,
                                                  Eigen::Array<Scalar, 8, 1>& u_up,
                                                  Eigen::Array<Scalar, 8, 1>& u_dw,
                                                  Eigen::Array<Scalar, 8, 1>& v_up,
                                                  Eigen::Array<Scalar, 8, 1>& v_dw)
    {
        // t order: 0,0,0,0,1,1,1,1
        // u order: 0,0,1,1,0,0,1,1
        // v order: 0,1,0,1,0,1,0,1
        const Scalar t0_up = itv[0].lower.numerator,
                     t0_dw = itv[0].lower.denominator(), t1_up = itv[0].upper.numerator,
                     t1_dw = itv[0].upper.denominator(), u0_up = itv[1].lower.numerator,
                     u0_dw = itv[1].lower.denominator(), u1_up = itv[1].upper.numerator,
                     u1_dw = itv[1].upper.denominator(), v0_up = itv[2].lower.numerator,
                     v0_dw = itv[2].lower.denominator(), v1_up = itv[2].upper.numerator,
                     v1_dw = itv[2].upper.denominator();
        t_up << t0_up, t0_up, t0_up, t0_up, t1_up, t1_up, t1_up, t1_up;
        t_dw << t0_dw, t0_dw, t0_dw, t0_dw, t1_dw, t1_dw, t1_dw, t1_dw;
        u_up << u0_up, u0_up, u1_up, u1_up, u0_up, u0_up, u1_up, u1_up;
        u_dw << u0_dw, u0_dw, u1_dw, u1_dw, u0_dw, u0_dw, u1_dw, u1_dw;
        v_up << v0_up, v1_up, v0_up, v1_up, v0_up, v1_up, v0_up, v1_up;
        v_dw << v0_dw, v1_dw, v0_dw, v1_dw, v0_dw, v1_dw, v0_dw, v1_dw;
    }

    // ** this version can return the true tolerance of the co-domain **
    // give the result of if the hex overlaps the input eps box around origin
    // use vectorized hex-vertex-solving function for acceleration
    // box_in_eps shows if this hex is totally inside box. if so, no need to do further bisection
    template <bool check_vf>
    MUDA_GENERIC static bool origin_in_function_bounding_box_vector(
        const Interval3& paras,
        const Vector3&   a0s,
        const Vector3&   a1s,
        const Vector3&   b0s,
        const Vector3&   b1s,
        const Vector3&   a0e,
        const Vector3&   a1e,
        const Vector3&   b0e,
        const Vector3&   b1e,
        const Array3&    eps,
        bool&            box_in_eps,
        const Scalar     ms        = 0,
        Array3*          tolerance = nullptr)
    {
        box_in_eps = false;

        Eigen::Array<Scalar, 8, 1> t_up, t_dw, u_up, u_dw, v_up, v_dw;
        {
            TIGHT_INCLUSION_SCOPED_TIMER(time_eval_origin_tuv);
            convert_tuv_to_array(paras, t_up, t_dw, u_up, u_dw, v_up, v_dw);
        }

        bool box_in[3];
        for(int i = 0; i < 3; i++)
        {
            TIGHT_INCLUSION_SCOPED_TIMER(time_eval_origin_1D);
            Scalar* tol = tolerance == nullptr ? nullptr : &((*tolerance)[i]);
            if(!evaluate_bbox_one_dimension_vector<check_vf>(
                   t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, i, eps[i], box_in[i], ms, tol))
            {
                return false;
            }
        }

        if(box_in[0] && box_in[1] && box_in[2])
        {
            box_in_eps = true;
        }

        return true;
    }


    template <bool check_vf>
    MUDA_GENERIC static bool evaluate_bbox_one_dimension_vector(
        Eigen::Array<Scalar, 8, 1>& t_up,
        Eigen::Array<Scalar, 8, 1>& t_dw,
        Eigen::Array<Scalar, 8, 1>& u_up,
        Eigen::Array<Scalar, 8, 1>& u_dw,
        Eigen::Array<Scalar, 8, 1>& v_up,
        Eigen::Array<Scalar, 8, 1>& v_dw,
        const Vector3&              a0s,
        const Vector3&              a1s,
        const Vector3&              b0s,
        const Vector3&              b1s,
        const Vector3&              a0e,
        const Vector3&              a1e,
        const Vector3&              b0e,
        const Vector3&              b1e,
        const int                   dim,
        const Scalar                eps,
        bool&                       bbox_in_eps,
        const Scalar                ms  = 0,
        Scalar*                     tol = nullptr)
    {
        TIGHT_INCLUSION_SCOPED_TIMER(time_vertex_solving);

        Eigen::Array<Scalar, 8, 1> vs;
        if constexpr(check_vf)
        {
            vs = function_vf(a0s[dim],
                             a1s[dim],
                             b0s[dim],
                             b1s[dim],
                             a0e[dim],
                             a1e[dim],
                             b0e[dim],
                             b1e[dim],
                             t_up,
                             t_dw,
                             u_up,
                             u_dw,
                             v_up,
                             v_dw);
        }
        else
        {
            vs = function_ee(a0s[dim],
                             a1s[dim],
                             b0s[dim],
                             b1s[dim],
                             a0e[dim],
                             a1e[dim],
                             b0e[dim],
                             b1e[dim],
                             t_up,
                             t_dw,
                             u_up,
                             u_dw,
                             v_up,
                             v_dw);
        }

        Scalar minv, maxv;
        min_max_array(vs, minv, maxv);

        if(tol != nullptr)
        {
            *tol = maxv - minv;  // this is the real tolerance
        }

        bbox_in_eps = false;

        const Scalar eps_and_ms = eps + ms;

        if(minv > eps_and_ms || maxv < -eps_and_ms)
        {
            return false;
        }

        if(minv >= -eps_and_ms && maxv <= eps_and_ms)
        {
            bbox_in_eps = true;
        }

        return true;
    }

    MUDA_GENERIC static Eigen::Array<Scalar, 8, 1> function_ee(
        const Scalar&                     a0s,
        const Scalar&                     a1s,
        const Scalar&                     b0s,
        const Scalar&                     b1s,
        const Scalar&                     a0e,
        const Scalar&                     a1e,
        const Scalar&                     b0e,
        const Scalar&                     b1e,
        const Eigen::Array<Scalar, 8, 1>& t_up,
        const Eigen::Array<Scalar, 8, 1>& t_dw,
        const Eigen::Array<Scalar, 8, 1>& u_up,
        const Eigen::Array<Scalar, 8, 1>& u_dw,
        const Eigen::Array<Scalar, 8, 1>& v_up,
        const Eigen::Array<Scalar, 8, 1>& v_dw)
    {
        Eigen::Array<Scalar, 8, 1> rst;
        for(int i = 0; i < 8; i++)
        {
            Scalar edge0_vertex0 = (a0e - a0s) * t_up[i] / t_dw[i] + a0s;
            Scalar edge0_vertex1 = (a1e - a1s) * t_up[i] / t_dw[i] + a1s;
            Scalar edge1_vertex0 = (b0e - b0s) * t_up[i] / t_dw[i] + b0s;
            Scalar edge1_vertex1 = (b1e - b1s) * t_up[i] / t_dw[i] + b1s;

            Scalar edge0_vertex =
                (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i] + edge0_vertex0;
            Scalar edge1_vertex =
                (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i] + edge1_vertex0;
            rst[i] = edge0_vertex - edge1_vertex;
        }
        return rst;
    }

    MUDA_GENERIC static Eigen::Array<Scalar, 8, 1> function_vf(
        const Scalar&                     vs,
        const Scalar&                     t0s,
        const Scalar&                     t1s,
        const Scalar&                     t2s,
        const Scalar&                     ve,
        const Scalar&                     t0e,
        const Scalar&                     t1e,
        const Scalar&                     t2e,
        const Eigen::Array<Scalar, 8, 1>& t_up,
        const Eigen::Array<Scalar, 8, 1>& t_dw,
        const Eigen::Array<Scalar, 8, 1>& u_up,
        const Eigen::Array<Scalar, 8, 1>& u_dw,
        const Eigen::Array<Scalar, 8, 1>& v_up,
        const Eigen::Array<Scalar, 8, 1>& v_dw)
    {
        Eigen::Array<Scalar, 8, 1> rst;
        for(int i = 0; i < 8; i++)
        {
            Scalar v  = (ve - vs) * t_up[i] / t_dw[i] + vs;
            Scalar t0 = (t0e - t0s) * t_up[i] / t_dw[i] + t0s;
            Scalar t1 = (t1e - t1s) * t_up[i] / t_dw[i] + t1s;
            Scalar t2 = (t2e - t2s) * t_up[i] / t_dw[i] + t2s;
            Scalar pt = (t1 - t0) * u_up[i] / u_dw[i] + (t2 - t0) * v_up[i] / v_dw[i] + t0;
            rst[i] = v - pt;
        }
        return rst;
    }

    template <typename T, int N>
    MUDA_GENERIC static void min_max_array(const Eigen::Array<T, N, 1>& arr, T& min, T& max)
    {
        static_assert(N > 0, "no min/max of empty array");
        min = arr[0];
        max = arr[0];
        for(int i = 1; i < N; i++)
        {
            if(min > arr[i])
            {
                min = arr[i];
            }
            if(max < arr[i])
            {
                max = arr[i];
            }
        }
    }

  private:
    MUDA_GENERIC bool edge_edge_interval_root_finder_BFS(const Vector3& a0s,
                                                         const Vector3& a1s,
                                                         const Vector3& b0s,
                                                         const Vector3& b1s,
                                                         const Vector3& a0e,
                                                         const Vector3& a1e,
                                                         const Vector3& b0e,
                                                         const Vector3& b1e,
                                                         const Array3&  tol,
                                                         const Scalar co_domain_tolerance,
                                                         // this is the maximum error on each axis when calculating the vertices, err, aka, filter
                                                         const Array3& err,
                                                         const Scalar  ms,
                                                         const Scalar  max_time,
                                                         const int     max_itr,
                                                         Scalar&       toi,
                                                         Scalar& output_tolerance)
    {
        return interval_root_finder_BFS<false>(
            a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, tol, co_domain_tolerance, err, ms, max_time, max_itr, toi, output_tolerance);
    }

    template <bool check_vf>
    MUDA_GENERIC bool interval_root_finder_BFS(const Vector3& a0s,
                                               const Vector3& a1s,
                                               const Vector3& b0s,
                                               const Vector3& b1s,
                                               const Vector3& a0e,
                                               const Vector3& a1e,
                                               const Vector3& b0e,
                                               const Vector3& b1e,
                                               const Array3&  tol,
                                               const Scalar co_domain_tolerance,
                                               const Array3& err,
                                               const Scalar  ms,
                                               const Scalar  max_time,
                                               const int     max_itr,
                                               Scalar&       toi,
                                               Scalar&       output_tolerance)
    {
        // build interval set [0,t_max]x[0,1]x[0,1]
        const Interval zero_to_one = Interval(NumCCD(0, 0), NumCCD(1, 0));
        Interval3      iset(zero_to_one, zero_to_one, zero_to_one);

        return interval_root_finder_BFS<check_vf>(
            a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, iset, tol, co_domain_tolerance, err, ms, max_time, max_itr, toi, output_tolerance);
    }

    using IStack =
        priority_queue<Interval3Pair, vector<Interval3Pair, allocator_type>, Compare>;
    IStack istack;

    // when check_t_overlap = false, check [0,1]x[0,1]x[0,1]; otherwise, check [0, t_max]x[0,1]x[0,1]
    template <bool check_vf>
    MUDA_GENERIC bool interval_root_finder_BFS(const Vector3&   a0s,
                                               const Vector3&   a1s,
                                               const Vector3&   b0s,
                                               const Vector3&   b1s,
                                               const Vector3&   a0e,
                                               const Vector3&   a1e,
                                               const Vector3&   b0e,
                                               const Vector3&   b1e,
                                               const Interval3& iset,
                                               const Array3&    tol,
                                               const Scalar co_domain_tolerance,
                                               const Array3& err,
                                               const Scalar  ms,
                                               const Scalar  max_time,
                                               const int     max_itr,
                                               Scalar&       toi,
                                               Scalar&       output_tolerance)
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
        Scalar t_upper_bound = max_time;  // 2*tol make it more conservative
        while(!istack.empty())
        {

            if(MAX_QUEUE_SIZE > 0 && istack.size() >= MAX_QUEUE_SIZE)
            {
                //muda_kernel_debug_info(DEBUG_TICCD, "ticcd istack full so just return");
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
            {  // if max_itr <= 0 ⟹ unlimited iterations
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

    MUDA_GENERIC static bool split_and_push(const Interval3& tuv,
                                            int              split_i,
                                            IStack&          istack,
                                            int              level,
                                            bool             check_vf,
                                            Scalar           t_upper_bound = 1)
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
            muda_kernel_check(check_vf && split_i != 0);
            // u + v ≤ 1
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
};


template <typename T>
using ticcd_alloc_elem_type = typename ticcd<T>::Interval3Pair;
}  // namespace muda::thread_only

#undef TIGHT_INCLUSION_SCOPED_TIMER
#undef CCD_MAX_TIME_TOL
#undef CCD_MAX_COORD_TOL
#undef MAX_DENOM_POWER
#undef MAX_NO_ZERO_TOI_ITER