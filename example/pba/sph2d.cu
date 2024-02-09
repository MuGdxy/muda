#include <Eigen/Core>
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <muda/tools/filesystem.h>
using namespace muda;

using Vector2 = Eigen::Vector2f;

// all const data in simulation are stored in this struct
struct ConstData
{
    // "Particle-Based Fluid Simulation for Interactive Applications" by Mueller et al.
    // solver parameters
    Vector2 G         = Vector2(0.f, -10.f);  // external (gravitational) forces
    float   REST_DENS = 300.f;                // rest density
    float   GAS_CONST = 2000.f;               //  for equation of state
    float   H         = 16.f;                 // kernel radius
    float   HSQ       = H * H;                // radius^2 for optimization
    float   MASS      = 2.5f;     // assume all particles have the same mass
    float   VISC      = 200.f;    // viscosity constant
    float   DT        = 0.0007f;  // integration timestep

    // smoothing kernels defined in Mueller and their gradients
    // adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
    float POLY6      = 4.f / (M_PI * pow(H, 8.f));
    float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
    float VISC_LAP   = 40.f / (M_PI * pow(H, 5.f));

    // simulation parameters
    float EPS           = H;  // boundary epsilon
    float BOUND_DAMPING = -0.5f;

    // interaction
    //  int MAX_PARTICLES   = 2500;
    int DAM_PARTICLES = 1000;
    // int BLOCK_PARTICLES = 250;

    // rendering projection parameters
    int    WINDOW_WIDTH  = 800;
    int    WINDOW_HEIGHT = 600;
    double VIEW_WIDTH    = 1.5 * 800.f;
    double VIEW_HEIGHT   = 1.5 * 600.f;
} CONST_DATA;

// particle data structure
// stores position, velocity, and force for integration
// stores density (rho) and pressure values for SPH
struct Particle
{
    MUDA_GENERIC Particle(float _x, float _y, int id)
        : x(_x, _y)
        , v(0.f, 0.f)
        , f(0.f, 0.f)
        , rho(0)
        , p(0.f)
        , id(id)
    {
    }
    MUDA_GENERIC Particle()
        : x(0, 0)
        , v(0.f, 0.f)
        , f(0.f, 0.f)
        , rho(0)
        , p(0.f)
        , id(-1)
    {
    }

    Vector2 x, v, f;
    float   rho, p;
    int     id;

    void to_csv(std::ostream& o) const
    {
        // expand to 3d for latter visualization
        // clang-format off
        o << x(0) << "," << x(1) << "," << 0.0f << ","  
          << v(0) << "," << v(1) << "," << 0.0f << "," 
          << f(0) << "," << f(1) << "," << 0.0f << ","
          << rho << "," << p << "," << id << std::endl;
        // clang-format on
    }

    // create header for csv file
    static void CSVHeader(std::ostream& o)
    {
        o << "x[0],x[1],x[2],v[0],v[1],v[2],f[0],f[1],f[2],rho,p,id" << std::endl;
    }
};

constexpr int BLOCK_DIM = 128;

class SPHSolver
{
    DeviceVector<Particle> particles;
    cudaStream_t           stream;

  public:
    SPHSolver(cudaStream_t stream = nullptr)
        : stream(stream)
    {
    }

    void set_particles(const HostVector<Particle>& p) { particles = p; }

    void solve()
    {
        compute_density_pressure();
        compute_forces();
        integrate();
    }

    void get_particles(HostVector<Particle>& p)
    {
        wait_device();
        // copy the particles from device to host
        p = particles;
    }

    void integrate()
    {
        // using dynamic grid size to cover all the particles
        ParallelFor(BLOCK_DIM, 0, stream)
            .apply(particles.size(),
                   [BOUND_DAMPING = CONST_DATA.BOUND_DAMPING,
                    EPS           = CONST_DATA.EPS,
                    VIEW_WIDTH    = CONST_DATA.VIEW_WIDTH,
                    VIEW_HEIGHT   = CONST_DATA.VIEW_HEIGHT,
                    DT            = CONST_DATA.DT,
                    particles = particles.viewer()] __device__(int i) mutable
                   {
                       auto& p = particles(i);
                       // forward Euler integration
                       p.v += DT * p.f / p.rho;
                       p.x += DT * p.v;

                       // enforce boundary conditions
                       if(p.x(0) - EPS < 0.f)
                       {
                           p.v(0) *= BOUND_DAMPING;
                           p.x(0) = EPS;
                       }
                       if(p.x(0) + EPS > VIEW_WIDTH)
                       {
                           p.v(0) *= BOUND_DAMPING;
                           p.x(0) = VIEW_WIDTH - EPS;
                       }
                       if(p.x(1) - EPS < 0.f)
                       {
                           p.v(1) *= BOUND_DAMPING;
                           p.x(1) = EPS;
                       }
                       if(p.x(1) + EPS > VIEW_HEIGHT)
                       {
                           p.v(1) *= BOUND_DAMPING;
                           p.x(1) = VIEW_HEIGHT - EPS;
                       }
                   });
    }

    void compute_forces()
    {
        // using dynamic grid size to cover all the particles
        ParallelFor(BLOCK_DIM, 0, stream)
            .apply(particles.size(),
                   [H          = CONST_DATA.H,
                    MASS       = CONST_DATA.MASS,
                    SPIKY_GRAD = CONST_DATA.SPIKY_GRAD,
                    VISC       = CONST_DATA.VISC,
                    VISC_LAP   = CONST_DATA.VISC_LAP,
                    G          = CONST_DATA.G,
                    particles  = particles.viewer()] __device__(int i) mutable
                   {
                       auto&   pi = particles(i);
                       Vector2 fpress(0.f, 0.f);
                       Vector2 fvisc(0.f, 0.f);
                       for(int j = 0; j < particles.dim(); ++j)
                       {
                           auto& pj = particles(j);
                           if(pi.id == pj.id)
                           {
                               continue;
                           }

                           Vector2 rij = pj.x - pi.x;
                           float   r   = rij.norm();

                           if(r < H)
                           {
                               // compute pressure force contribution
                               fpress += -rij.normalized() * MASS * (pi.p + pj.p)
                                         / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
                               // compute viscosity force contribution
                               fvisc += VISC * MASS * (pj.v - pi.v) / pj.rho
                                        * VISC_LAP * (H - r);
                           }
                       }
                       Vector2 fgrav = G * MASS / pi.rho;
                       pi.f          = fpress + fvisc + fgrav;
                   });
    }

    void compute_density_pressure()
    {
        // using dynamic grid size to cover all the particles
        ParallelFor(BLOCK_DIM, 0, stream)
            .apply(particles.size(),
                   [HSQ       = CONST_DATA.HSQ,
                    MASS      = CONST_DATA.MASS,
                    POLY6     = CONST_DATA.POLY6,
                    GAS_CONST = CONST_DATA.GAS_CONST,
                    REST_DENS = CONST_DATA.REST_DENS,
                    particles = particles.viewer()] __device__(int i) mutable
                   {
                       auto& pi = particles(i);
                       pi.rho   = 0.f;
                       for(int j = 0; j < particles.dim(); ++j)
                       {
                           auto&   pj  = particles(j);
                           Vector2 rij = pj.x - pi.x;
                           float   r2  = rij.squaredNorm();

                           if(r2 < HSQ)
                           {
                               // this computation is symmetric
                               pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
                           }
                       }
                       pi.p = GAS_CONST * (pi.rho - REST_DENS);
                   });
    }
};

void init_sph(HostVector<Particle>& particles)
{
    int i = 0;
    for(float y = CONST_DATA.EPS; y < CONST_DATA.VIEW_HEIGHT - CONST_DATA.EPS * 2.f;
        y += CONST_DATA.H)
    {
        for(float x = CONST_DATA.VIEW_WIDTH / 4; x <= CONST_DATA.VIEW_WIDTH / 2;
            x += CONST_DATA.H)
        {
            if(particles.size() < CONST_DATA.DAM_PARTICLES)
            {
                float jitter =
                    static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                particles.push_back(Particle(x + jitter, y, i++));
            }
            else
            {
                return;
            }
        }
    }
}

void export_csv(const std::string& folder, int idx, HostVector<Particle>& particles)
{
    std::ofstream     f;
    std::stringstream ss;
    f.open(folder + "/" + std::to_string(idx) + ".csv");
    Particle::CSVHeader(ss);
    for(const auto& p : particles)
        p.to_csv(ss);
    f << ss.str();
    f.close();
}

void sph2d(int particle_count)
{
    example_desc(
        "a simple 2d Smoothed Particle Hydrodynamics exmaple.\n"
        "ref: https://lucasschuermann.com/writing/implementing-sph-in-2d\n"
        "you could modify the particle count to get better result.\n");

    // create particles on host
    HostVector<Particle> particles;
    CONST_DATA.DAM_PARTICLES = particle_count;
    particles.reserve(CONST_DATA.DAM_PARTICLES);
    // generate particles randomly
    init_sph(particles);
    std::cout << "initializing dam break with " << CONST_DATA.DAM_PARTICLES
              << " particles" << std::endl;

    std::cout << "delta time =" << CONST_DATA.DT << std::endl;
    std::cout << "particle count =" << CONST_DATA.DAM_PARTICLES << std::endl;

    // create a stream (RAII style, no need to destroy manually)
    Stream s;

    SPHSolver solver(s);
    // set the particles in solver
    solver.set_particles(particles);

    // create a folder for frame data output
    filesystem::path folder("sph2d/");
    if(!filesystem::exists(folder))
        filesystem::create_directory(folder);

    auto abspath = filesystem::absolute(folder);
    std::cout << "solve frames to folder: " << abspath << std::endl;

    int bar    = 77;
    int nframe = 1000;
    for(int i = 0; i < nframe; i++)
    {
        solver.solve();
        solver.get_particles(particles);
        make_progress_bar((i + 1.0f) / nframe, bar);
        export_csv("sph2d", i, particles);
    }
}

TEST_CASE("sph2d-light", "[pba]")
{
    sph2d(100);
}

TEST_CASE("sph2d-full", "[.pba]")
{
    sph2d(1000);
}